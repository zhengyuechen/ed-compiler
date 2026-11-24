#!/usr/bin/env python3
"""edc — a small operator compiler for exact diagonalisation (spin-1/2).

The user describes a Hamiltonian as a list of terms on sites.  The compiler
  * splits the bonds into "families" in which every site is the first member of at most one bond,
  * writes each family as a few shift-and-mask operations on the bit string (the "structure"),
  * does the same for symmetry operations (site permutations),
  * emits C with OpenMP, compiles it with clang -O3, and binds it with ctypes.
The hot loops contain no Python.  Python only describes the model and drives the Lanczos iteration
(one C call per Lanczos step).

Terms (spin-1/2 operators S = sigma/2):
  exchange(i, j, J)   J (Sx_i Sx_j + Sy_i Sy_j) = J/2 (S+_i S-_j + S-_i S+_j)
  zz(i, j, J)         J Sz_i Sz_j
  heisenberg(i, j, J) both of the above
  field(i, h)         h Sz_i
Symmetry: total Sz (number of up spins) and, optionally, a group of site permutations at momentum zero
(trivial representation), handled with a lookup table over the full Sz sector.
"""
import os, sys, math, time, ctypes, hashlib, subprocess
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the OpenMP runtime for macOS: vendor/libomp of a checkout, edc_vendor/libomp next to the modules after `pip install`
OMP = next((d for d in (os.path.join(HERE, v, 'libomp') for v in ('vendor', 'edc_vendor')) if os.path.isdir(d)), os.path.join(HERE, 'vendor', 'libomp'))
OMP_INC = os.path.join(OMP, 'include')
OMP_DYLIB = os.path.join(OMP, 'lib', 'libomp.dylib')
CACHE = os.environ.get('EDC_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'edc'))
if sys.platform == 'darwin':
    # one shared OpenMP runtime for every generated library in this process
    _OMP = ctypes.CDLL(OMP_DYLIB, mode=ctypes.RTLD_GLOBAL)
    CC = ['clang', '-O3', '-mcpu=native', '-ffp-contract=fast', '-Xpreprocessor', '-fopenmp', f'-I{OMP_INC}',
          '-shared', '-fPIC', '-undefined', 'dynamic_lookup']
else:
    CC = [os.environ.get('EDC_CC', 'gcc'), '-O3', '-march=native', '-ffp-contract=fast', '-fopenmp', '-shared', '-fPIC']


# ------------------------------------------------------------------ model description
class SpinHalfModel:
    def __init__(self, n_sites):
        self.n = n_sites
        self.bonds = defaultdict(lambda: [0.0, 0.0])   # (i, j) with i < j  ->  [Jxy, Jz]
        self.orient = {}                                # first orientation given by the user
        self.h = np.zeros(n_sites)

    def _key(self, i, j):
        k = (min(i, j), max(i, j))
        self.orient.setdefault(k, (i, j))
        return k

    def exchange(self, i, j, J):
        self.bonds[self._key(i, j)][0] += J

    def zz(self, i, j, J):
        self.bonds[self._key(i, j)][1] += J

    def heisenberg(self, i, j, J):
        self.exchange(i, j, J); self.zz(i, j, J)

    def field(self, i, h):
        self.h[i] += h


def ring(L, J1=1.0, J2=0.0):
    m = SpinHalfModel(L)
    for i in range(L):
        m.heisenberg(i, (i+1) % L, J1)
        if J2:
            m.heisenberg(i, (i+2) % L, J2)
    return m


def square(Lx, Ly, J=1.0):
    m = SpinHalfModel(Lx*Ly)
    for y in range(Ly):
        for x in range(Lx):
            i = x+Lx*y
            m.heisenberg(i, (x+1) % Lx+Lx*y, J)
            m.heisenberg(i, x+Lx*((y+1) % Ly), J)
    return m


def ring_translations(L):
    return [[(i+k) % L for i in range(L)] for k in range(L)]


def torus_translations(Lx, Ly):
    return [[((i % Lx+dx) % Lx)+Lx*(((i//Lx)+dy) % Ly) for i in range(Lx*Ly)]
            for dy in range(Ly) for dx in range(Lx)]


def square_space_group(L):
    """All 8 L^2 symmetry operations of the L x L periodic square lattice (translations x D4), as site permutations."""
    pts = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
           lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (y, x), lambda x, y: (-y, -x)]
    ops = []
    for P in pts:
        for dy in range(L):
            for dx in range(L):
                perm = []
                for i in range(L*L):
                    X, Y = P(i % L, i // L)
                    perm.append(((X+dx) % L)+L*((Y+dy) % L))
                ops.append(perm)
    assert len({tuple(p) for p in ops}) == len(ops)
    return ops


def rect_point_ops(Lx, Ly):
    """Point operations of the Lx x Ly periodic rectangle about the origin: D4 if Lx == Ly, otherwise D2."""
    pts = [lambda x, y: (x, y), lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (-x, -y)]
    if Lx == Ly:
        pts += [lambda x, y: (-y, x), lambda x, y: (y, -x), lambda x, y: (y, x), lambda x, y: (-y, -x)]
    return [[(P(i % Lx, i // Lx)[0] % Lx)+Lx*(P(i % Lx, i // Lx)[1] % Ly) for i in range(Lx*Ly)] for P in pts]


def rect_space_group(Lx, Ly):
    """Point operation first, then translation (dx, dy); index = p * Lx*Ly + dy * Lx + dx."""
    ops = []
    for P in rect_point_ops(Lx, Ly):
        for dy in range(Ly):
            for dx in range(Lx):
                ops.append([((P[i] % Lx+dx) % Lx)+Lx*((P[i]//Lx+dy) % Ly) for i in range(Lx*Ly)])
    assert len({tuple(p) for p in ops}) == len(ops)
    return ops


# ------------------------------------------------------------------ structure extraction
def families(model):
    """Orient bonds and split them so that each site is the first member of at most one bond per family."""
    fams = []
    for key, (jxy, jz) in model.bonds.items():
        if jxy == 0 and jz == 0 and not getattr(model, 'keep_zero', False):    # keep_zero: scans through zero keep the bond
            continue
        i, j = model.orient[key]
        # prefer the orientation that follows the lattice (i -> j, or the wrap j -> i)
        placed = False
        for a, b in ((i, j), (j, i)):
            for f in fams:
                if a not in f:
                    f[a] = (b, jxy, jz); placed = True; break
            if placed:
                break
        if not placed:
            fams.append({i: (j, jxy, jz)})
    return fams


def shift_groups(pairs):
    """pairs: list of (src_bit, dst_bit).  Returns {shift: mask_of_src_bits}."""
    g = defaultdict(int)
    for a, b in pairs:
        g[b-a] |= 1 << a
    return dict(g)


# ------------------------------------------------------------------ code generation
class Gen:
    def __init__(self, model, nup, group, K, schedule):
        self.m, self.nup, self.group, self.K, self.schedule = model, nup, group, K, schedule
        L = model.n
        self.L, self.LA = L, L//2
        self.T = 'uint32_t' if L <= 32 else 'uint64_t'
        self.POP = '__builtin_popcount' if L <= 32 else '__builtin_popcountll'
        self.CTZ = '__builtin_ctz' if L <= 32 else '__builtin_ctzll'
        self.U = 'u' if L <= 32 else 'ull'
        self.JAT = 'uint32_t' if math.comb(L, nup) < 2**32 else 'uint64_t'
        self.fams = families(model)

    def c(self, v):
        return f'{hex(v)}{self.U}'

    def u_expr(self, fam):
        g = shift_groups([(a, b) for a, (b, _, _) in fam.items()])
        parts = []
        for d, msk in sorted(g.items()):
            sh = f'(s >> {d})' if d > 0 else f'(s << {-d})'
            parts.append(f'((s ^ {sh}) & {self.c(msk)})')
        return ' | '.join(parts)

    def apply_expr(self, perm, var):
        g = shift_groups([(i, perm[i]) for i in range(self.L)])
        parts = []
        for d, msk in sorted(g.items()):
            if d > 0:
                parts.append(f'(({var} & {self.c(msk)}) << {d})')
            elif d < 0:
                parts.append(f'(({var} & {self.c(msk)}) >> {-d})')
            else:
                parts.append(f'({var} & {self.c(msk)})')
        return ' | '.join(parts)

    def source(self):
        L, LA, T, K, U = self.L, self.LA, self.T, self.K, self.U
        sym = self.group is not None
        low = (1 << LA)-1
        o = ['#include <stdint.h>', '#include <math.h>', '#include <omp.h>',
             f'typedef {T} st_t;',
             f'typedef {self.JAT} ja_t;',
             'static const ja_t *JA; static const uint32_t *JB, *TAB; static const st_t *ST; static const double *SQ, *ISQ;',
             'static const int64_t *PTR; static const uint32_t *IDXS; static double *XS; static int64_t NROW;',
             'void edc_bind(const ja_t* ja, const uint32_t* jb, const uint32_t* tab, const st_t* st, const double* sq,',
             '              const double* isq, double* xs, int64_t nrow){ JA=ja; JB=jb; TAB=tab; ST=st; SQ=sq; ISQ=isq; XS=xs; NROW=nrow; }',
             'void edc_bind_stored(const int64_t* ptr, const uint32_t* idx){ PTR=ptr; IDXS=idx; }',
             'void edc_threads(int n){ omp_set_num_threads(n); }',
             f'#define FULLIDX(t) ((int64_t)JA[(t) >> {LA}] + JB[(t) & {self.c(low)}])']
        o.append(f'#define IDX(t) ({"(int64_t)TAB[FULLIDX(t)]" if sym else "FULLIDX(t)"})')
        # constants of the diagonal
        dconst = 0.0
        exch = []
        for fi, f in enumerate(self.fams):
            flip = [0]*L
            cxy = [0.0]*L
            wz = [0.0]*L
            for a, (b, jxy, jz) in f.items():
                flip[a] = (1 << a) | (1 << b)
                cxy[a] = jxy/2
                wz[a] = jz/4
                dconst += jz/4
            o.append(f'static const st_t FLIP{fi}[{L}] = {{{", ".join(self.c(v) for v in flip)}}};')
            mxy = sum(1 << a for a, (b, jxy, jz) in f.items() if jxy != 0)
            mz = sum(1 << a for a, (b, jxy, jz) in f.items() if jz != 0)
            fm = sum(1 << a for a in f)
            uxy = {cxy[a] for a in f if cxy[a] != 0}
            uz = {wz[a] for a in f if wz[a] != 0}
            if len(uxy) > 1:
                o.append(f'static const double CXY{fi}[{L}] = {{{", ".join(repr(v) for v in cxy)}}};')
            if len(uz) > 1:
                o.append(f'static const double WZ{fi}[{L}] = {{{", ".join(repr(v) for v in wz)}}};')
            exch += list(uxy)
            f['_meta'] = dict(mxy=mxy, mz=mz, fm=fm, cxy=(uxy.pop() if len(uxy) == 1 else None) if uxy else 0.0,
                              wz=(uz.pop() if len(uz) == 1 else None) if uz else 0.0)
        h = self.m.h
        dconst = float(dconst) - float(h.sum())/2
        hu = {float(v) for v in h if v != 0}
        huni = next(iter(hu)) if len(hu) == 1 else None
        self.uniform_coef = (len(set(exch)) <= 1)
        self.coef = exch[0] if exch else 0.0
        # symmetry operations
        if sym:
            for gi, perm in enumerate(self.group):
                o.append(f'static inline st_t G{gi}(st_t s){{ return {self.apply_expr(perm, "s")}; }}')
            body = ' '.join(f'{{ st_t r = G{gi}(s); if (r < s) return 0; if (r == s) st++; }}'
                            for gi in range(1, len(self.group)))
            o.append(f'static inline int is_rep(st_t s, int* stab){{ int st = 1; {body} *stab = st; return 1; }}')

        def diag_lines(with_offdiag, X, XIN, out):
            r = ['st_t s = ST[i];', f'double d = {dconst!r};']
            if with_offdiag:
                r.append(f'double acc[{K}]; for (int q = 0; q < {K}; q++) acc[q] = 0.0;')
            for fi, f in enumerate(self.fams):
                meta = f['_meta']
                r.append(f'{{ st_t u = {self.u_expr({a: v for a, v in f.items() if a != "_meta"})};')
                if meta['mz']:
                    uz = 'u' if meta['mz'] == meta['fm'] else f'(u & {self.c(meta["mz"])})'
                    if meta['wz'] is not None:
                        r.append(f'd -= {2*meta["wz"]!r} * (double){self.POP}({uz});')
                    else:
                        r.append(f'{{ st_t z = {uz}; while (z) {{ int a = {self.CTZ}(z); z &= z - 1; d -= 2.0 * WZ{fi}[a]; }} }}')
                if with_offdiag and meta['mxy']:
                    ux = 'u' if meta['mxy'] == meta['fm'] else f'(u & {self.c(meta["mxy"])})'
                    cexpr = repr(meta['cxy']) if meta['cxy'] is not None else f'CXY{fi}[a]'
                    r.append(f'{{ st_t fl = {ux}; while (fl) {{ int a = {self.CTZ}(fl); fl &= fl - 1; '
                             f'st_t t = s ^ FLIP{fi}[a]; int64_t j = IDX(t); double c = {cexpr}; '
                             f'for (int q = 0; q < {K}; q++) acc[q] += c * {X}[j*{K}+q]; }} }}')
                r.append('}')
            if huni is not None:
                r.append(f'd += {huni!r} * (double){self.POP}(s);')
            elif len(hu) > 1:
                r += [f'd += {float(h[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if h[i] != 0]
            if with_offdiag:
                scale = 'SQ[i] * ' if sym else ''
                r.append(f'for (int q = 0; q < {K}; q++) {out}[i*{K}+q] = d * {XIN}[i*{K}+q] + {scale}acc[q];')
            return r

        sched = self.schedule
        # (1) matrix-free kernel
        X = 'XS' if sym else 'x'
        o.append('void mv_free(const double* restrict x, double* restrict y){')
        if sym:
            o.append(f'  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW*{K}; i++) XS[i] = x[i] * ISQ[i/{K}];')
        o.append(f'  #pragma omp parallel for schedule({sched})\n  for (int64_t i = 0; i < NROW; i++) {{')
        o += ['    '+line for line in diag_lines(True, X, 'x', 'y')]
        o.append('  }\n}')
        # (2) diagonal function and stored-index kernel
        o.append('static inline double diag_of(st_t s, int64_t i){')
        o += ['  '+line for line in diag_lines(False, None, None, None)[1:]]
        o.append('  (void)i; return d; }')
        o.append('int64_t stored_count(int64_t* cnt){ int64_t tot = 0;')
        o.append(f'  #pragma omp parallel for schedule({sched}) reduction(+:tot)\n  for (int64_t i = 0; i < NROW; i++) {{ st_t s = ST[i]; int64_t c = 0;')
        for f in self.fams:
            meta = f['_meta']
            if meta['mxy']:
                ue = self.u_expr({a: v for a, v in f.items() if a != '_meta'})
                o.append(f'    c += {self.POP}(({ue}) & {self.c(meta["mxy"])});')
        o.append('    cnt[i] = c; tot += c; }\n  return tot; }')
        o.append('void stored_fill(uint32_t* restrict idx){')
        o.append(f'  #pragma omp parallel for schedule({sched})\n  for (int64_t i = 0; i < NROW; i++) {{ st_t s = ST[i]; int64_t p = PTR[i];')
        for fi, f in enumerate(self.fams):
            meta = f['_meta']
            if meta['mxy']:
                ue = self.u_expr({a: v for a, v in f.items() if a != '_meta'})
                o.append(f'    {{ st_t fl = ({ue}) & {self.c(meta["mxy"])}; while (fl) {{ int a = {self.CTZ}(fl); fl &= fl - 1; '
                         f'st_t t = s ^ FLIP{fi}[a]; idx[p++] = (uint32_t)IDX(t); }} }}')
        o.append('  }\n}')
        o.append('void mv_stored(const double* restrict x, double* restrict y){')
        if sym:
            o.append(f'  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW*{K}; i++) XS[i] = x[i] * ISQ[i/{K}];')
        o.append(f'  #pragma omp parallel for schedule({sched})\n  for (int64_t i = 0; i < NROW; i++) {{')
        o.append(f'    double d = diag_of(ST[i], i); double acc[{K}]; for (int q = 0; q < {K}; q++) acc[q] = 0.0;')
        o.append(f'    for (int64_t p = PTR[i]; p < PTR[i+1]; p++) {{ int64_t j = IDXS[p]; for (int q = 0; q < {K}; q++) acc[q] += {X}[j*{K}+q]; }}')
        scale = 'SQ[i] * ' if sym else ''
        o.append(f'    for (int q = 0; q < {K}; q++) y[i*{K}+q] = d * x[i*{K}+q] + {scale}{self.coef!r} * acc[q];')
        o.append('  }\n}')
        # (2b) row-blocked kernel: the bit string is split into high half (rows) and low half (columns).
        #      Bonds inside the high half move whole rows -> streaming row updates; the rest stays per element.
        if not sym:
            HHm = [0]*len(self.fams)
            for fi, f in enumerate(self.fams):
                for a, v in f.items():
                    if a == '_meta':
                        continue
                    if a >= LA and v[0] >= LA:
                        HHm[fi] |= 1 << a
            o.append('static const st_t *LOWS; static const int64_t *OFF;')
            o.append('void edc_bind_rows(const st_t* lows, const int64_t* off){ LOWS = lows; OFF = off; }')
            o.append('void mv_rows(const double* restrict x, double* restrict y){')
            o.append(f'  #pragma omp parallel for schedule(dynamic, 1)\n  for (int64_t h = 0; h < {1 << (L-LA)}; h++) {{')
            o.append(f'    int p = {self.nup} - {self.POP}((st_t)h); if (p < 0 || p > {LA}) continue;')
            o.append(f'    const int64_t base = JA[h], n = OFF[p+1] - OFF[p]; const st_t* lw = LOWS + OFF[p];')
            o.append(f'    double* restrict yr = y + base*{K}; const double* restrict xr = x + base*{K};')
            o.append(f'    for (int64_t k = 0; k < n; k++) {{ st_t s = ((st_t)h << {LA}) | lw[k]; double d = {dconst!r};')
            o.append(f'      double acc[{K}]; for (int q = 0; q < {K}; q++) acc[q] = 0.0;')
            for fi, f in enumerate(self.fams):
                meta = f['_meta']
                ff = {a: v for a, v in f.items() if a != '_meta'}
                o.append(f'      {{ st_t u = {self.u_expr(ff)};')
                if meta['mz']:
                    uz = 'u' if meta['mz'] == meta['fm'] else f'(u & {self.c(meta["mz"])})'
                    if meta['wz'] is not None:
                        o.append(f'        d -= {2*meta["wz"]!r} * (double){self.POP}({uz});')
                    else:
                        o.append(f'        {{ st_t z = {uz}; while (z) {{ int a = {self.CTZ}(z); z &= z - 1; d -= 2.0 * WZ{fi}[a]; }} }}')
                rest = meta['mxy'] & ~HHm[fi]
                if rest:
                    cexpr = repr(meta['cxy']) if meta['cxy'] is not None else f'CXY{fi}[a]'
                    o.append(f'        {{ st_t fl = u & {self.c(rest)}; while (fl) {{ int a = {self.CTZ}(fl); fl &= fl - 1; '
                             f'st_t t = s ^ FLIP{fi}[a]; int64_t j = FULLIDX(t); double c = {cexpr}; '
                             f'for (int q = 0; q < {K}; q++) acc[q] += c * x[j*{K}+q]; }} }}')
                o.append('      }')
            if huni is not None:
                o.append(f'      d += {huni!r} * (double){self.POP}(s);')
            elif len(hu) > 1:
                o += [f'      d += {float(h[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if h[i] != 0]
            o.append(f'      for (int q = 0; q < {K}; q++) yr[k*{K}+q] = d * xr[k*{K}+q] + acc[q]; }}')
            # whole-row updates for bonds inside the high half
            o.append(f'    st_t s = (st_t)h << {LA};')
            for fi, f in enumerate(self.fams):
                meta = f['_meta']
                hh = meta['mxy'] & HHm[fi]
                if not hh:
                    continue
                ff = {a: v for a, v in f.items() if a != '_meta'}
                cexpr = repr(meta['cxy']) if meta['cxy'] is not None else f'CXY{fi}[a]'
                o.append(f'    {{ st_t u = ({self.u_expr(ff)}) & {self.c(hh)}; while (u) {{ int a = {self.CTZ}(u); u &= u - 1;')
                o.append(f'      const double c = {cexpr}; const double* restrict xr2 = x + (int64_t)JA[(s ^ FLIP{fi}[a]) >> {LA}]*{K};')
                o.append(f'      for (int64_t k = 0; k < n*{K}; k++) yr[k] += c * xr2[k]; }} }}')
            o.append('  }\n}')
        # (3) basis construction
        o.append(f'void fill_basis(const st_t* lows, const int64_t* off, st_t* out){{')
        o.append(f'  #pragma omp parallel for schedule(dynamic, 64)\n  for (int64_t h = 0; h < {1 << (L-LA)}; h++) {{')
        o.append(f'    int p = {self.nup} - {self.POP}((st_t)h); if (p < 0 || p > {LA}) continue; int64_t o = JA[h];')
        o.append(f'    for (int64_t q = off[p]; q < off[p+1]; q++) out[o++] = ((st_t)h << {LA}) | lows[q]; }}\n}}')
        if sym:
            o.append(f'void count_reps(const st_t* lows, const int64_t* off, int64_t* cnt){{')
            o.append(f'  #pragma omp parallel for schedule(dynamic, 16)\n  for (int64_t h = 0; h < {1 << (L-LA)}; h++) {{ int64_t c = 0;')
            o.append(f'    int p = {self.nup} - {self.POP}((st_t)h); if (p >= 0 && p <= {LA})')
            o.append(f'      for (int64_t q = off[p]; q < off[p+1]; q++) {{ int stb; c += is_rep(((st_t)h << {LA}) | lows[q], &stb); }}')
            o.append('    cnt[h] = c; }\n}')
            o.append(f'void fill_reps(const st_t* lows, const int64_t* off, const int64_t* start, st_t* reps, double* sq, double* isq){{')
            o.append(f'  #pragma omp parallel for schedule(dynamic, 16)\n  for (int64_t h = 0; h < {1 << (L-LA)}; h++) {{ int64_t o = start[h];')
            o.append(f'    int p = {self.nup} - {self.POP}((st_t)h); if (p < 0 || p > {LA}) continue;')
            o.append(f'    for (int64_t q = off[p]; q < off[p+1]; q++) {{ st_t s = ((st_t)h << {LA}) | lows[q]; int stb;')
            o.append(f'      if (is_rep(s, &stb)) {{ reps[o] = s; double n = {len(self.group)}.0 / stb; sq[o] = sqrt(n); isq[o] = 1.0 / sqrt(n); o++; }} }} }}\n}}')
            o.append('void fill_tab(uint32_t* tab){')
            o.append(f'  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW; i++) {{ st_t s = ST[i];')
            o += [f'    {{ st_t t = G{gi}(s); tab[FULLIDX(t)] = (uint32_t)i; }}' for gi in range(len(self.group))]
            o.append('  }\n}')
        # (4) one Lanczos step: w = H v - alpha v - beta_prev vp, normalised; ab = (alpha, beta)
        o.append('void lanczos_step(int stored, const double* restrict v, const double* restrict vp, double* restrict w, double bprev, double* ab){')
        o.append('  if (stored == 1) mv_stored(v, w); else if (stored == 2) mv_rows(v, w); else mv_free(v, w);' if not sym else
                 '  if (stored == 1) mv_stored(v, w); else mv_free(v, w);')
        o.append('  double al = 0.0;\n  #pragma omp parallel for schedule(static) reduction(+:al)\n  for (int64_t i = 0; i < NROW; i++) al += v[i] * w[i];')
        o.append('  double nr = 0.0;\n  #pragma omp parallel for schedule(static) reduction(+:nr)\n  for (int64_t i = 0; i < NROW; i++) { double t = w[i] - al * v[i] - bprev * vp[i]; w[i] = t; nr += t * t; }')
        o.append('  double be = sqrt(nr), ib = 1.0 / be;\n  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW; i++) w[i] *= ib;')
        o.append('  ab[0] = al; ab[1] = be; }')
        return '\n'.join(o)+'\n'


def compile_source(src, tag='k'):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(src.encode()).hexdigest()[:16]
    so = os.path.join(CACHE, f'{tag}_{key}.so')
    t0 = time.perf_counter()
    if not os.path.exists(so):
        # private file names: several processes may compile the same source at once (jobs that share the cache)
        c = os.path.join(CACHE, f'{tag}_{key}.{os.getpid()}.c'); tmp = f'{so}.{os.getpid()}.tmp'
        open(c, 'w').write(src)
        subprocess.check_call(CC+['-o', tmp, c])
        os.replace(tmp, so); os.replace(c, os.path.join(CACHE, f'{tag}_{key}.c'))
    return ctypes.CDLL(so), time.perf_counter()-t0, so


def lin_tables(L, nup):
    import math
    LA = L//2
    v = np.arange(1 << LA, dtype=np.int64)
    pl = np.zeros(1 << LA, np.int64)
    for b in range(LA):
        pl += (v >> b) & 1
    order = np.lexsort((v, pl))
    lows = v[order]
    off = np.zeros(LA+2, np.int64); off[1:] = np.cumsum(np.bincount(pl, minlength=LA+1))
    jb = np.zeros(1 << LA, np.uint32); jb[order] = (np.arange(1 << LA)-off[pl[order]]).astype(np.uint32)
    vh = np.arange(1 << (L-LA), dtype=np.int64)
    ph = np.zeros(1 << (L-LA), np.int64)
    for b in range(L-LA):
        ph += (vh >> b) & 1
    need = nup-ph
    comb = np.array([math.comb(LA, int(q)) if 0 <= q <= LA else 0 for q in range(-L, L+1)])
    cnt = np.where((need >= 0) & (need <= LA), comb[need+L], 0)
    ja = np.zeros(1 << (L-LA), np.int64); ja[1:] = np.cumsum(cnt)[:-1]
    return lows, off, ja, jb, int(cnt.sum())



# ------------------------------------------------------------------ runtime
PD = ctypes.POINTER(ctypes.c_double)
PI64 = ctypes.POINTER(ctypes.c_int64)
PU32 = ctypes.POINTER(ctypes.c_uint32)


def ptr(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


class Engine:
    def __init__(self, model, nup, group=None, K=1, schedule='dynamic, 4096', threads=None):
        t0 = time.perf_counter()
        self.gen = Gen(model, nup, group, K, schedule)
        self.src = self.gen.source()
        self.lib, self.compile_seconds, self.so_path = compile_source(self.src)
        self.L, self.K, self.sym = model.n, K, group is not None
        st_c = ctypes.c_uint32 if self.L <= 32 else ctypes.c_uint64
        st_np = np.uint32 if self.L <= 32 else np.uint64
        self.st_c, self.st_np = st_c, st_np
        if threads:
            self.lib.edc_threads(threads)
        L, LA = self.L, self.L//2
        # Lin tables (low half: rank within popcount class; high half: offsets)
        v = np.arange(1 << LA, dtype=np.int64)
        pl = np.zeros(1 << LA, np.int64)
        for b in range(LA):
            pl += (v >> b) & 1
        order = np.lexsort((v, pl))
        lows = v[order].astype(st_np)
        off = np.zeros(LA+2, np.int64)
        off[1:] = np.cumsum(np.bincount(pl, minlength=LA+1))
        jb = np.zeros(1 << LA, np.uint32)
        jb[order] = (np.arange(1 << LA)-off[pl[order]]).astype(np.uint32)
        vh = np.arange(1 << (L-LA), dtype=np.int64)
        ph = np.zeros(1 << (L-LA), np.int64)
        for b in range(L-LA):
            ph += (vh >> b) & 1
        need = nup-ph
        cnt = np.where((need >= 0) & (need <= LA), np.array([math.comb(LA, int(q)) if 0 <= q <= LA else 0 for q in range(-L, L+1)])[need+L], 0)
        self.ja_np = np.uint32 if math.comb(L, nup) < 2**32 else np.uint64
        self.ja_c = ctypes.c_uint32 if self.ja_np is np.uint32 else ctypes.c_uint64
        ja = np.zeros(1 << (L-LA), self.ja_np)
        ja[1:] = np.cumsum(cnt)[:-1].astype(self.ja_np)
        self.full_dim = int(cnt.sum())
        self.ja, self.jb, self.lows, self.off = ja, jb, lows, off
        self.lib.edc_bind.argtypes = [ctypes.POINTER(self.ja_c), PU32, PU32, ctypes.POINTER(st_c), PD, PD, PD, ctypes.c_int64]
        dummy = np.zeros(1, np.uint32)
        if not self.sym:
            self.st = np.empty(self.full_dim, st_np)
            self.lib.edc_bind(ptr(ja, self.ja_c), ptr(jb, ctypes.c_uint32), ptr(dummy, ctypes.c_uint32),
                              ptr(self.st, st_c), None, None, None, self.full_dim)
            self.lib.fill_basis.argtypes = [ctypes.POINTER(st_c), PI64, ctypes.POINTER(st_c)]
            self.lib.fill_basis(ptr(lows, st_c), ptr(off, ctypes.c_int64), ptr(self.st, st_c))
            self.dim = self.full_dim
            self.tab = dummy
            self.sq = self.isq = self.xs = None
        else:
            self.lib.edc_bind(ptr(ja, self.ja_c), ptr(jb, ctypes.c_uint32), ptr(dummy, ctypes.c_uint32),
                              None, None, None, None, 0)
            ch = np.zeros(1 << (L-LA), np.int64)
            self.lib.count_reps.argtypes = [ctypes.POINTER(st_c), PI64, PI64]
            self.lib.count_reps(ptr(lows, st_c), ptr(off, ctypes.c_int64), ptr(ch, ctypes.c_int64))
            start = np.zeros_like(ch); start[1:] = np.cumsum(ch)[:-1]
            self.dim = int(ch.sum())
            self.st = np.empty(self.dim, st_np)
            self.sq = np.empty(self.dim); self.isq = np.empty(self.dim)
            self.lib.fill_reps.argtypes = [ctypes.POINTER(st_c), PI64, PI64, ctypes.POINTER(st_c), PD, PD]
            self.lib.fill_reps(ptr(lows, st_c), ptr(off, ctypes.c_int64), ptr(start, ctypes.c_int64),
                               ptr(self.st, st_c), ptr(self.sq, ctypes.c_double), ptr(self.isq, ctypes.c_double))
            self.tab = np.empty(self.full_dim, np.uint32)
            self.xs = np.empty(self.dim*K)
            self.lib.edc_bind(ptr(ja, self.ja_c), ptr(jb, ctypes.c_uint32), ptr(self.tab, ctypes.c_uint32),
                              ptr(self.st, st_c), ptr(self.sq, ctypes.c_double), ptr(self.isq, ctypes.c_double),
                              ptr(self.xs, ctypes.c_double), self.dim)
            self.lib.fill_tab.argtypes = [PU32]
            self.lib.fill_tab(ptr(self.tab, ctypes.c_uint32))
        for name in ('mv_free', 'mv_stored') + (() if self.sym else ('mv_rows',)):
            getattr(self.lib, name).argtypes = [PD, PD]
        if not self.sym:
            self.lib.edc_bind_rows.argtypes = [ctypes.POINTER(st_c), PI64]
            self.lib.edc_bind_rows(ptr(self.lows, st_c), ptr(self.off, ctypes.c_int64))
        self.lib.lanczos_step.argtypes = [ctypes.c_int, PD, PD, PD, ctypes.c_double, PD]
        self.lib.stored_count.argtypes = [PI64]; self.lib.stored_count.restype = ctypes.c_int64
        self.lib.stored_fill.argtypes = [PU32]
        self.lib.edc_bind_stored.argtypes = [PI64, PU32]
        self.setup_seconds = time.perf_counter()-t0
        self.stored_ready = False

    def threads(self, n):
        self.lib.edc_threads(n)

    def build_stored(self):
        if not self.gen.uniform_coef:
            raise NotImplementedError('stored kernel needs one exchange coefficient')
        t0 = time.perf_counter()
        cnt = np.empty(self.dim, np.int64)
        tot = self.lib.stored_count(ptr(cnt, ctypes.c_int64))
        self.ptr = np.zeros(self.dim+1, np.int64)
        np.cumsum(cnt, out=self.ptr[1:]); del cnt
        self.idx = np.empty(tot, np.uint32)
        self.lib.edc_bind_stored(ptr(self.ptr, ctypes.c_int64), ptr(self.idx, ctypes.c_uint32))
        self.lib.stored_fill(ptr(self.idx, ctypes.c_uint32))
        self.stored_ready = True
        self.stored_seconds = time.perf_counter()-t0
        self.stored_gib = (self.ptr.nbytes+self.idx.nbytes)/2**30
        return self.stored_seconds

    def autotune(self, mem_budget_gib=4.0, expected_iterations=100, reps=2, verbose=False):
        """Time every applicable kernel and keep the one with the smallest build + expected Lanczos cost."""
        x = np.random.default_rng(1).standard_normal(self.dim*self.K); y = np.empty_like(x)
        cand = ['free'] + ([] if self.sym else ['rows'])
        if self.gen.uniform_coef:
            cnt = np.empty(self.dim, np.int64)
            nnz = self.lib.stored_count(ptr(cnt, ctypes.c_int64)); del cnt
            if (4*nnz+8*(self.dim+1))/2**30 <= mem_budget_gib:
                cand.append('stored')
        cost, self.tune = {}, {}
        for k in cand:
            build = self.build_stored() if k == 'stored' and not self.stored_ready else 0.0
            self.matvec(x, y, kernel=k)
            t = min((lambda t0: (self.matvec(x, y, kernel=k), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(reps))
            self.tune[k] = dict(matvec_seconds=t, build_seconds=build)
            cost[k] = build + expected_iterations*t
        self.best_kernel = min(cost, key=cost.get)
        if self.best_kernel != 'stored' and self.stored_ready:
            del self.idx; self.stored_ready = False          # give the memory back
        if verbose:
            print('autotune:', {k: round(v['matvec_seconds']*1e3, 2) for k, v in self.tune.items()}, '->', self.best_kernel)
        return self.best_kernel

    def matvec(self, x, y=None, stored=False, kernel=None):
        if y is None:
            y = np.empty_like(x)
        kernel = kernel or ('stored' if stored else 'free')
        getattr(self.lib, 'mv_'+kernel)(ptr(x, ctypes.c_double), ptr(y, ctypes.c_double))
        return y

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, stored=False, check_every=1):
        from scipy.linalg import eigvalsh_tridiagonal
        if stored == 'auto':
            stored = getattr(self, 'best_kernel', None) or self.autotune()
        assert self.K == 1
        n = self.dim
        v = np.random.default_rng(seed).standard_normal(n); v /= np.linalg.norm(v)
        vp = np.zeros(n); w = np.empty(n)
        ab = np.zeros(2)
        alphas, betas = [], []
        bprev, e_old = 0.0, None
        t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.lib.lanczos_step({False: 0, True: 1, 'stored': 1, 'free': 0, 'rows': 2}[stored], ptr(v, ctypes.c_double), ptr(vp, ctypes.c_double), ptr(w, ctypes.c_double),
                                  bprev, ptr(ab, ctypes.c_double))
            alphas.append(ab[0]); betas.append(ab[1])
            if it % check_every == 0 or it == maxit:
                e = eigvalsh_tridiagonal(np.array(alphas), np.array(betas[:-1]), select='i', select_range=(0, 0))[0] \
                    if it > 1 else alphas[0]
                if e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e)):
                    return e, it, time.perf_counter()-t0
                e_old = e
            if ab[1] < 1e-14:
                break
            bprev = ab[1]
            vp, v, w = v, w, vp
        return e_old, it, time.perf_counter()-t0
