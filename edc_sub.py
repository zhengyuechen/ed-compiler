#!/usr/bin/env python3
"""Spin-1/2 models with a large symmetry group: representatives through a sublattice table (CPU C/OpenMP or CUDA from
one generated source).

The representative of a basis state s is the minimum of its group orbit (integer order, top bit most significant).
Sites are relabelled so that the top w bits form a set T of sites ("sublattice") whose set-stabiliser K in the group
is large.  For g = k c, with c one representative of the right coset K g, the top part of g s is k acting on the
w-bit string x_c = (c s)|_T.  A table over all 2^w strings gives the K-orbit minimum of x and one element k0
reaching it; the representative's top part is the smallest of these minima over the few cosets, and only the
stabiliser of that minimum (usually the identity) has to be searched for the rest of the string.  The cost per matrix
element is (number of cosets) table look-ups plus one or two full permutations, instead of a walk over the group.
Spin flip (S_z = 0) is part of K (it maps T to itself).

Sector conventions as edc_gpu_lean: projector P = (1/|G|) sum chi(g)^* U(g); pull form
(H x)_r = d_r x_r + sum_flips J/2 chi(g) sqrt(|Stab r'| / |Stab r|) x_r'."""
import time, math, ctypes, itertools, re
import numpy as np
import edc
from edc_hubbard_sym import lanczos, _popcount64


_POPC = None
INCREMENTAL = False      # default for SubEngine(incremental=None)
STORED = 'auto'          # default for SubEngine(stored=None): build the stored connectivity when it fits
STORED_MIN_DIM = 20000
_BOUND = {}     # library or module key -> id of the engine whose tables are bound there (identical sources share one
                # compiled library, and with it the global table pointers)


def _hex(v):
    return f'{hex(v)}ull'


def choose_top(cl, perms, wmax=24):
    """Candidate site sets: each basis sublattice of the cluster, and sublattices of index 2, 3, 4 of the Bravais
    lattice (cells with (a x + b y) = 0 mod d) within one or all basis sublattices.  Returns the set T (w <= wmax) with
    the largest set-stabiliser in the group."""
    N = len(perms[0])
    cand = []
    pos = [cl.where(t) if hasattr(cl, 'where') else (0,)+tuple(cl.coord[t]) for t in range(N)]
    nb = getattr(cl, 'nb', 1)
    subl = [frozenset(t for t in range(N) if pos[t][0] == s) for s in range(nb)]
    cand += subl
    for d in (2, 3, 4):
        for a, b in itertools.product(range(d), repeat=2):
            if (a, b) == (0, 0):
                continue
            Sd = frozenset(t for t in range(N) if (a*pos[t][1]+b*pos[t][2]) % d == 0)
            cand.append(Sd)
            for S in subl:
                cand.append(Sd & S)
    best = None
    for T in cand:
        if not 0 < len(T) <= wmax or len(T) == N:
            continue
        K = sum(1 for p in perms if {p[a] for a in T} == T)
        score = (K, len(T))
        if best is None or score > best[0]:
            best = (score, T)
    return sorted(best[1])


def cpl_array(model):
    """Run-time coupling table matching source(): cxy[f*L+a] = Jxy/2, wz[nf*L+f*L+a] = Jz/4, the constant part of the
    diagonal, and the fields h[i] (families in edc.families order)."""
    L = model.n
    fams = edc.families(model)
    nf = len(fams)
    c = np.zeros(2*nf*L+1+L)
    dconst = 0.0
    for fi, f in enumerate(fams):
        for a, (b, jxy, jz) in f.items():
            if a == '_meta':
                continue
            c[fi*L+a] = jxy/2; c[(nf+fi)*L+a] = jz/4; dconst += jz/4
    c[2*nf*L] = dconst-float(np.sum(model.h))/2
    c[2*nf*L+1:] = model.h
    return c


def structure_key(model):
    """What the generated code depends on: the bond pairs per family and whether a field is present."""
    return ([sorted((a, v[0]) for a, v in f.items() if a != '_meta') for f in edc.families(model)], bool(np.any(model.h != 0)))


def source(L, w, model, nup, cosets, nK, nbytes, real, flip, target, rank64=False, batch=0, incremental=False, stored=0,
           batch_fp32=False, batch_lanes=False, batch_fp16=False):
    """Generated code.  cosets: list of site permutations c (relabelled); tables bound at run time.  Representatives are
    indexed by (K-orbit minimum q of the top w bits, Lin rank of the other L - w bits among strings with the remaining
    popcount).  STOREF (run time): use the list of representative states, otherwise a row decodes its state from its index.
    batch = R > 0 adds mvb: H applied to R vectors stored interleaved (x[j R + r]).  Every matrix element (representative
    search, index, phase) is computed once per row and used for all R vectors: one thread per row with R accumulators.
    batch_fp32: the R vectors are single precision (sums in double).  batch_fp16 (CUDA): they are stored in half precision
    (complex: pairs of halves).  The product is linear, so a common scale factor of the stored vectors passes through it.  batch_lanes (CUDA, double precision only): the older
    variant with R adjacent threads per row, one vector each, every thread repeating the matrix-element work.
    incremental: a site permutation is linear over XOR and a bond flip is an XOR with a two-bit mask, so the coset images of
    a flipped state are (coset images of the row state, computed once per row) XOR (images of the mask, tabulated): one XOR
    per coset instead of a full permutation.
    stored = M > 0 adds the stored-connectivity kernels: a count pass and a fill pass write every matrix element once as
    (column index u32, code u32 = phase index m (12 bits, phase exp(2 pi i m / M)) | bond id (10 bits) | target stabiliser
    (10 bits)), with the diagonal and the row stabiliser per row; spmv then reads these entries and never searches."""
    NR = L-w; LB = NR//2
    o = []
    if target == 'cuda':
        o += ['#include <cupy/complex.cuh>', '#define DEV __device__ __forceinline__', '#define GLOB __device__',
              '#define POPC64(x) __popcll(x)', '#define FFS64(x) __ffsll(x)', 'typedef unsigned long long st_t;',
              'typedef long long i64;', 'typedef unsigned long long u64;']
        o += ['typedef double vt;' if real else 'typedef complex<double> vt;']
    else:
        o += ['#include <stdint.h>', '#include <math.h>', '#include <complex.h>', '#include <omp.h>',
              '#define DEV static inline', '#define GLOB static', '#define POPC64(x) __builtin_popcountll(x)',
              '#define FFS64(x) __builtin_ffsll(x)', 'typedef uint64_t st_t;', 'typedef int64_t i64;', 'typedef uint64_t u64;']
        o += ['typedef double vt;' if real else 'typedef double _Complex vt;']
    o.append(f'#define NS {L}\n#define W {w}\n#define SH {L-w}\n#define NR {NR}\n#define LB {LB}\n#define NH {1 << (NR-LB)}\n'
             f'#define NUPC {nup}\n#define LOWB {_hex((1 << LB)-1)}\n#define RESTM {_hex((1 << NR)-1)}\n#define NB {nbytes}\n'
             f'#define FULL {_hex((1 << L)-1)}')
    tabs = [('unsigned', 'MINT'), ('unsigned short', 'KARG'), ('int', 'SPTR'), ('unsigned short', 'SIDX'),
            ('unsigned long long', 'PTK'), ('double', 'CKR'), ('double', 'CKI'), ('double', 'CCR'), ('double', 'CCI'),
            ('long long', 'JAP'), ('unsigned', 'JB'), ('unsigned long long', 'LOWS'), ('long long', 'OFF'),
            ('int', 'QIDX'), ('unsigned', 'QMIN'), ('long long', 'OFFQ'),
            ('unsigned long long', 'BITS'), ('unsigned long long' if rank64 else 'unsigned', 'RANK'), ('unsigned long long', 'ST'),
            ('double', 'CPL')]
    for ty, nm in tabs:
        o.append(f'GLOB const {ty}* {nm};')
    o.append('GLOB long long NQ; GLOB long long NWD; GLOB int STOREF;')
    if target == 'c':
        o.append('void bind(void** p, long long nq, long long nwd, int store) { ' + ' '.join(f'{nm} = (const {ty}*)p[{i}];' for i, (ty, nm) in enumerate(tabs))
                 + ' NQ = nq; NWD = nwd; STOREF = store; }')
    # full permutation of an element of K (with its flip bit) through byte tables
    o.append('DEV st_t appk(int k, st_t s) { const unsigned long long* t = PTK + (i64)(k >> 1) * NB * 256; st_t r = '
             + ' | '.join(f't[{c*256} + ((s >> {8*c}) & 255)]' for c in range(nbytes)) + '; return (k & 1) ? (r ^ FULL) : r; }')
    if real:
        o.append('DEV vt chik(int k) { return CKR[k]; }')
        o.append('DEV vt chic(int c) { return CCR[c]; }')
        isone = 'fabs(z - 1.0) > 1e-9'
    elif target == 'cuda':
        o.append('DEV vt chik(int k) { return vt(CKR[k], CKI[k]); }')
        o.append('DEV vt chic(int c) { return vt(CCR[c], CCI[c]); }')
        isone = 'fabs(z.real() - 1.0) + fabs(z.imag()) > 1e-9'
    else:
        o.append('DEV vt chik(int k) { return CKR[k] + CKI[k] * I; }')
        o.append('DEV vt chic(int c) { return CCR[c] + CCI[c] * I; }')
        isone = 'fabs(creal(z) - 1.0) + fabs(cimag(z)) > 1e-9'
    g = edc.Gen(model, nup, None, 1, 'static'); g.source(); g.U = 'ull'     # source() fills the family metadata
    for ci, c in enumerate(cosets):
        o.append(f'DEV st_t C{ci}(st_t s) {{ return {g.apply_expr(c, "s")}; }}')
    # representative: minimum over the cosets; stabiliser count and phase chi(g) of one minimising element
    body = ['DEV st_t rep_of(st_t t, int* stab, vt* ph) { st_t best = ~(st_t)0; int cnt = 0; vt p = 1.0;']
    for ci in range(len(cosets)):
        body.append(f' {{ st_t u = C{ci}(t); unsigned x = (unsigned)(u >> SH); unsigned mt = MINT[x];'
                    f' if ((st_t)mt <= (best >> SH)) {{ int k0 = KARG[x]; st_t v0 = appk(k0, u);'
                    f' for (int q = SPTR[mt]; q < SPTR[mt + 1]; q++) {{ int h = SIDX[q]; st_t v = h ? appk(h, v0) : v0;'
                    f' if (v < best) {{ best = v; cnt = 1; p = chik(h) * chik(k0) * chic({ci}); }} else if (v == best) cnt++; }} }} }}')
    body.append(' *stab = cnt; *ph = p; return best; }')
    o.append(' '.join(body))
    nc = len(cosets)
    if incremental:      # the same search, with the coset images supplied by the caller
        b2 = ['DEV st_t rep_inc(' + ', '.join(f'st_t u{ci}' for ci in range(nc)) + ', int* stab, vt* ph) {'
              ' st_t best = ~(st_t)0; int cnt = 0; vt p = 1.0;']
        for ci in range(nc):
            b2.append(f' {{ st_t u = u{ci}; unsigned x = (unsigned)(u >> SH); unsigned mt = MINT[x];'
                      f' if ((st_t)mt <= (best >> SH)) {{ int k0 = KARG[x]; st_t v0 = appk(k0, u);'
                      f' for (int q = SPTR[mt]; q < SPTR[mt + 1]; q++) {{ int h = SIDX[q]; st_t v = h ? appk(h, v0) : v0;'
                      f' if (v < best) {{ best = v; cnt = 1; p = chik(h) * chik(k0) * chic({ci}); }} else if (v == best) cnt++; }} }} }}')
        b2.append(' *stab = cnt; *ph = p; return best; }')
        o.append(' '.join(b2))
    # is s a representative of this sector?  0 if not, else the stabiliser size
    chk = ['DEV int stab_if_rep(st_t s) { int cnt = 0; unsigned st = (unsigned)(s >> SH);']
    for ci in range(len(cosets)):
        chk.append(f' {{ st_t u = C{ci}(s); unsigned x = (unsigned)(u >> SH); unsigned mt = MINT[x]; if (mt < st) return 0;'
                   f' if (mt == st) {{ int k0 = KARG[x]; st_t v0 = appk(k0, u);'
                   f' for (int q = SPTR[mt]; q < SPTR[mt + 1]; q++) {{ int h = SIDX[q]; st_t v = h ? appk(h, v0) : v0;'
                   f' if (v < s) return 0; if (v == s) {{ cnt++; vt z = chik(h) * chik(k0) * chic({ci}); if ({isone}) return 0; }} }} }} }}')
    chk.append(' return cnt; }')
    o.append(' '.join(chk))
    # index of a representative: offset of its top-part block + rank of the rest among strings of its popcount
    o.append('DEV long long index_of(st_t r) { unsigned mt = (unsigned)(r >> SH); st_t rho = r & RESTM; int p = NUPC - POPC64((u64)mt);'
             ' return OFFQ[QIDX[mt]] + JAP[(i64)p * NH + (i64)(rho >> LB)] + JB[rho & LOWB]; }')
    o.append('DEV st_t state_at(i64 f) { int lo = 0, hi = (int)NQ - 1;'
             ' while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (OFFQ[mid] <= f) lo = mid; else hi = mid - 1; }'
             ' i64 rr = f - OFFQ[lo]; unsigned mt = QMIN[lo]; int p = NUPC - POPC64((u64)mt); const long long* ja = JAP + (i64)p * NH;'
             ' int a = 0, b = NH - 1; while (a < b) { int mid = (a + b + 1) >> 1; if (ja[mid] <= rr) a = mid; else b = mid - 1; }'
             ' int pl = p - POPC64((u64)a); st_t rho = ((st_t)a << LB) | LOWS[OFF[pl] + (rr - ja[a])];'
             ' return ((st_t)mt << SH) | rho; }')
    o.append('DEV st_t state_of_row(i64 i) { i64 lo = 0, hi = NWD - 1;'
             ' while (lo < hi) { i64 mid = (lo + hi + 1) >> 1; if ((i64)RANK[mid] <= i) lo = mid; else hi = mid - 1; }'
             ' u64 word = BITS[lo]; int k = (int)(i - (i64)RANK[lo]); for (int t = 0; t < k; t++) word &= word - 1;'
             ' return state_at(lo * 64 + (FFS64(word) - 1)); }')
    # matrix elements: families of bonds as shift-and-mask (from edc.Gen)
    rows, rows_cnt, rows_fill, rows_diag, cxyg, dconst = [], [], [], [], [], 0.0
    def add(t):
        rows.append(t); rows_cnt.append(t); rows_fill.append(t); rows_diag.append(t)
    nf = len(g.fams)
    DC, HO = 2*nf*L, 2*nf*L+1                       # CPL layout: cxy[f*NS+a], wz[nf*NS+f*NS+a], dconst, h[i]
    for fi, f in enumerate(g.fams):
        ff = {a: v for a, v in f.items() if a != '_meta'}
        meta = f['_meta']
        flipm = [0]*L; cxy = [0.0]*L; wz = [0.0]*L
        for a, (b, jxy, jz) in ff.items():
            flipm[a] = (1 << a) | (1 << b); cxy[a] = jxy/2; wz[a] = jz/4; dconst += jz/4
        o.append(f'GLOB const st_t FLIP{fi}[NS] = {{{", ".join(_hex(v) for v in flipm)}}};')
        mall = sum(1 << a for a in ff)
        if incremental:
            for ci, c in enumerate(cosets):
                img = [sum(1 << c[b] for b in range(L) if (m >> b) & 1) for m in flipm]
                o.append(f'GLOB const st_t CF{fi}_{ci}[NS] = {{{", ".join(_hex(v) for v in img)}}};')
        cxyg += cxy
        add(f'{{ st_t u = {g.u_expr(ff)};')
        add(f'{{ st_t z = u & {_hex(mall)}; while (z) {{ int a = FFS64(z) - 1; z &= z - 1; d -= 2.0 * CPL[{nf + fi} * NS + a]; }} }}')
        if True:
            cexpr = f'CPL[{fi} * NS + a]'
            open_ = (f'{{ st_t fl = u & {_hex(mall)}; while (fl) {{ int a = FFS64(fl) - 1; fl &= fl - 1; '
                     + ((f'int sj; vt ph; st_t r = rep_inc(' + ', '.join(f'U{ci} ^ CF{fi}_{ci}[a]' for ci in range(nc)) + ', &sj, &ph); ')
                        if incremental else f'int sj; vt ph; st_t r = rep_of(s ^ FLIP{fi}[a], &sj, &ph); ') +
                     f'u64 f = (u64)index_of(r); u64 wd = BITS[f >> 6]; if (!((wd >> (f & 63)) & 1ull)) continue; '
                     f'i64 j = (i64)RANK[f >> 6] + POPC64(wd & ((1ull << (f & 63)) - 1)); ')
            rows.append(open_ + f'acc += ({cexpr} * sqrt((double)sj)) * ph * XVAL(j); }} }}')
            rows_cnt.append(open_ + 'cnt++; } }')
            rows_fill.append(open_ + f'EMIT({fi} * NS + a, sj, ph, j); }} }}')
        add('}')
    h = model.h
    if np.any(h != 0):
        for i in range(L):
            add(f'd += CPL[{HO + i}] * (double)((s >> {i}) & 1);')
    head = f'st_t s = STOREF ? ST[i] : state_of_row(i); double d = CPL[{DC}]; vt acc = 0.0;'
    tail = '  int st; vt ph; rep_of(s, &st, &ph); return d * XVAL(i) + acc / sqrt((double)st); }'
    if incremental:
        head += ' ' + ' '.join(f'st_t U{ci} = C{ci}(s);' for ci in range(nc))
        tail = ('  int st; vt ph; rep_inc(' + ', '.join(f'U{ci}' for ci in range(nc))
                + ', &st, &ph); return d * XVAL(i) + acc / sqrt((double)st); }')
    o.append('#define XVAL(j) x[j]')
    o.append('DEV vt row(i64 i, const vt* x) { ' + head)
    o += ['  '+r for r in rows] + [tail, '#undef XVAL']
    if batch:
        o.append(f'#define RB {batch}')
        if batch_fp16:
            assert target == 'cuda'
            o.append('#include <cuda_fp16.h>\ntypedef __half xb;\n#define BTOV(a) ((double)__half2float(a))\n#define BTOX(a) (__float2half((float)(a)))' if real else
                     '#include <cuda_fp16.h>\nstruct xb { __half re, im; };\nstatic __device__ __forceinline__ vt BTOV(xb a) { return vt((double)__half2float(a.re), (double)__half2float(a.im)); }\nstatic __device__ __forceinline__ xb BTOX(vt a) { xb r; r.re = __float2half((float)a.real()); r.im = __float2half((float)a.imag()); return r; }')
        elif not batch_fp32:
            o.append('typedef vt xb;\n#define BTOV(a) (a)\n#define BTOX(a) (a)')
        elif target == 'cuda':
            o.append('typedef float xb;\n#define BTOV(a) ((double)(a))\n#define BTOX(a) ((float)(a))' if real else
                     'typedef complex<float> xb;\n#define BTOV(a) (vt((double)(a).real(), (double)(a).imag()))\n'
                     '#define BTOX(a) (xb((float)(a).real(), (float)(a).imag()))')
        else:
            o.append('typedef float xb;\n#define BTOV(a) ((double)(a))\n#define BTOX(a) ((float)(a))' if real else
                     'typedef float _Complex xb;\n#define BTOV(a) ((vt)(a))\n#define BTOX(a) ((xb)(a))')
        pat = re.compile(r'acc \+= (\(.*\)) \* ph \* XVAL\(j\);')
        rows_b = [pat.sub(r'ACCB(\1 * ph, j);', r) for r in rows]
        assert sum('ACCB(' in r for r in rows_b) == sum('XVAL(j)' in r for r in rows)
        o.append('#define ACCB(c, j) { vt c_ = (c); const xb* xj_ = x + (i64)(j) * RB; for (int r_ = 0; r_ < RB; r_++) acc[r_] += c_ * BTOV(xj_[r_]); }')
        o.append('DEV void rowb(i64 i, const xb* x, vt* out) { st_t s = STOREF ? ST[i] : state_of_row(i); double d = '
                 + f'CPL[{DC}]' + '; vt acc[RB]; for (int r_ = 0; r_ < RB; r_++) acc[r_] = 0.0;'
                 + ((' ' + ' '.join(f'st_t U{ci} = C{ci}(s);' for ci in range(nc))) if incremental else ''))
        o += ['  '+r for r in rows_b]
        o.append('  int st; vt ph; rep_of(s, &st, &ph); double is_ = 1.0 / sqrt((double)st);'
                 ' for (int r_ = 0; r_ < RB; r_++) out[r_] = d * BTOV(x[i * RB + r_]) + acc[r_] * is_; }')
        o.append('#undef ACCB')
    if target == 'cuda':
        # several GPUs: rows split in blocks [BND0, BND1), ...; x entries of other blocks are read through peer access
        o.append('GLOB long long BND1, BND2, BND3;')
        o.append('DEV vt xget(const vt* x0, const vt* x1, const vt* x2, const vt* x3, i64 j) {'
                 ' if (j < BND1) return x0[j]; if (j < BND2) return x1[j - BND1]; if (j < BND3) return x2[j - BND2]; return x3[j - BND3]; }')
        o.append('#define XVAL(j) xget(x0, x1, x2, x3, (j))')
        o.append('DEV vt rowm(i64 i, const vt* x0, const vt* x1, const vt* x2, const vt* x3) { ' + head)
        o += ['  '+r for r in rows] + [tail, '#undef XVAL']
    if stored:
        compact = isinstance(stored, dict)
        M = int(stored['M']) if compact else int(stored)
        assert M < 4096 and nf*L <= 1024
        o.append(f'#define MROOT {M}')
        cty = 'unsigned char' if (compact and stored.get('code8')) else 'unsigned short'   # 5-byte entries when there are < 256 codes
        if compact:            # 6-byte entries: u16 code = ((family * NSJ) + stabiliser index) * M + phase index -> VT[code]
            divs = stored['divs']
            sjidx = [0]*(max(divs)+1)
            for t, dv in enumerate(divs):
                sjidx[dv] = t
            o.append(f'#define NSJ {len(divs)}')
            o.append(f'GLOB const unsigned short SJIDX[{len(sjidx)}] = {{{", ".join(str(v) for v in sjidx)}}};')
            o.append(f'GLOB const unsigned short BCLS[{nf*L}] = {{{", ".join(str(v) for v in stored["bcls"])}}};')
        o.append('GLOB const double SQT[1024] = {' + ', '.join(repr(math.sqrt(v)) for v in range(1024)) + '};')
        o.append(f'GLOB const double RRT[{M}] = {{{", ".join(repr(math.cos(2*math.pi*m/M)) for m in range(M))}}};')
        o.append(f'GLOB const double RIT[{M}] = {{{", ".join(repr(math.sin(2*math.pi*m/M)) for m in range(M))}}};')
        if real:
            o.append('DEV unsigned pcode(vt ph) { return ph < 0 ? (unsigned)(MROOT / 2) : 0u; }')
            o.append('#define ROOTV(m) RRT[m]')
        else:
            re_, im_ = ('ph.real()', 'ph.imag()') if target == 'cuda' else ('creal(ph)', 'cimag(ph)')
            o.append(f'DEV unsigned pcode(vt ph) {{ long long m = llround(atan2({im_}, {re_}) * (MROOT / 6.283185307179586));'
                     ' m %= MROOT; if (m < 0) m += MROOT; return (unsigned)m; }')
            o.append('#define ROOTV(m) vt(RRT[m], RIT[m])' if target == 'cuda' else '#define ROOTV(m) (RRT[m] + RIT[m] * I)')
        if compact:
            o.append('#define EMIT(B, SJ, PH, J) { JJ[p] = (unsigned)(J);'
                     f' MM[p] = ({cty})((((unsigned)BCLS[B] * NSJ) + SJIDX[SJ]) * MROOT + pcode(PH)); p++; }}')
        else:
            o.append('#define EMIT(B, SJ, PH, J) { JJ[p] = (unsigned)(J); MM[p] = pcode(PH) | ((unsigned)(B) << 12) | ((unsigned)(SJ) << 22); p++; }')
        mty = cty if compact else 'unsigned'
        hd = f'st_t s = STOREF ? ST[i] : state_of_row(i); double d = CPL[{DC}];'
        if incremental:
            hd += ' ' + ' '.join(f'st_t U{ci} = C{ci}(s);' for ci in range(nc))
        own = ('rep_inc(' + ', '.join(f'U{ci}' for ci in range(nc)) + ', &st, &ph)') if incremental else 'rep_of(s, &st, &ph)'
        o.append('DEV unsigned row_cnt(i64 i) { ' + hd + ' unsigned cnt = 0;')
        o += ['  '+r for r in rows_cnt] + ['  return cnt; }']
        o.append(f'DEV void row_fill(i64 i, unsigned* JJ, {mty}* MM, i64 p, double* DG, unsigned short* STR) {{ ' + hd)
        o += ['  '+r for r in rows_fill] + [f'  int st; vt ph; {own}; DG[i] = d; STR[i] = (unsigned short)st; }}']
        o.append('#undef EMIT')
        o.append('DEV double row_diag(i64 i) { ' + hd)
        o += ['  '+r for r in rows_diag] + ['  return d; }']
        spmv_body = ('{ acc += VT[MM[e]] * x[JJ[e]]; }' if compact else
                     '{ unsigned m = MM[e]; acc += (CPL[(m >> 12) & 1023u] * SQT[m >> 22]) * ROOTV(m & 4095u) * x[JJ[e]]; }')
        vtarg = ', const vt* __restrict__ VT' if (compact and target == 'cuda') else (', const vt* VT' if compact else '')
        # single-precision vectors (xt), double accumulation: spmv32
        if target == 'cuda':
            o.append('typedef float xt;\n#define TOV(a) ((double)(a))\n#define TOX(a) ((float)(a))' if real else
                     'typedef complex<float> xt;\n#define TOV(a) (vt((double)(a).real(), (double)(a).imag()))\n'
                     '#define TOX(a) (xt((float)(a).real(), (float)(a).imag()))')
        else:
            o.append('typedef float xt;\n#define TOV(a) ((double)(a))\n#define TOX(a) ((float)(a))' if real else
                     'typedef float _Complex xt;\n#define TOV(a) ((vt)(a))\n#define TOX(a) ((xt)(a))')
        body32 = spmv_body.replace('x[JJ[e]]', 'TOV(x[JJ[e]])')
        if target == 'cuda':
            o.append('extern "C" __global__ void scnt(unsigned* c, i64 n) {'
                     ' i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = row_cnt(i); }')
            o.append('extern "C" __global__ void sdiag(double* DG, i64 n) {'
                     ' i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) DG[i] = row_diag(i); }')
            o.append(f'extern "C" __global__ void sfill(const u64* RP, unsigned* JJ, {mty}* MM, double* DG, unsigned short* STR, i64 n) {{'
                     ' i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) row_fill(i, JJ, MM, (i64)RP[i], DG, STR); }')
            red = ('acc += __shfl_down_sync(0xffffffffu, acc, off, 8);' if real else
                   '{ double re = acc.real(), im = acc.imag(); re += __shfl_down_sync(0xffffffffu, re, off, 8);'
                   ' im += __shfl_down_sync(0xffffffffu, im, off, 8); acc = vt(re, im); }')
            o.append('extern "C" __global__ void spmv(const vt* __restrict__ x, vt* __restrict__ y, i64 n, int accumulate,'
                     f' const u64* __restrict__ RP, const unsigned* __restrict__ JJ, const {mty}* __restrict__ MM,'
                     ' const double* __restrict__ DG, const unsigned short* __restrict__ STR' + vtarg + ') {'
                     ' i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x; i64 i = t >> 3; int lane = (int)(t & 7); vt acc = 0.0;'
                     ' if (i < n) { for (u64 e = RP[i] + lane; e < RP[i + 1]; e += 8) ' + spmv_body + ' }'
                     ' for (int off = 4; off > 0; off >>= 1) ' + red +
                     ' if (lane == 0 && i < n) { vt r = DG[i] * x[i] + acc / sqrt((double)STR[i]); y[i] = accumulate ? y[i] + r : r; } }')
            o.append('extern "C" __global__ void spmv32(const xt* __restrict__ x, xt* __restrict__ y, i64 n, int accumulate,'
                     f' const u64* __restrict__ RP, const unsigned* __restrict__ JJ, const {mty}* __restrict__ MM,'
                     ' const double* __restrict__ DG, const unsigned short* __restrict__ STR' + vtarg + ') {'
                     ' i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x; i64 i = t >> 3; int lane = (int)(t & 7); vt acc = 0.0;'
                     ' if (i < n) { for (u64 e = RP[i] + lane; e < RP[i + 1]; e += 8) ' + body32 + ' }'
                     ' for (int off = 4; off > 0; off >>= 1) ' + red +
                     ' if (lane == 0 && i < n) { vt r = DG[i] * TOV(x[i]) + acc / sqrt((double)STR[i]);'
                     ' y[i] = TOX(accumulate ? TOV(y[i]) + r : r); } }')
        else:
            o.append('void scnt(unsigned* c, i64 n) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                     ' for (i64 i = 0; i < n; i++) c[i] = row_cnt(i); }')
            o.append('void sdiag(double* DG, i64 n) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                     ' for (i64 i = 0; i < n; i++) DG[i] = row_diag(i); }')
            o.append(f'void sfill(const u64* RP, unsigned* JJ, {mty}* MM, double* DG, unsigned short* STR, i64 n) {{\n'
                     '#pragma omp parallel for schedule(dynamic, 1024)\n for (i64 i = 0; i < n; i++) row_fill(i, JJ, MM, (i64)RP[i], DG, STR); }')
            o.append(f'void spmv(const vt* x, vt* y, i64 n, int accumulate, const u64* RP, const unsigned* JJ, const {mty}* MM,'
                     ' const double* DG, const unsigned short* STR' + vtarg + ') {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                     ' for (i64 i = 0; i < n; i++) { vt acc = 0.0; for (u64 e = RP[i]; e < RP[i + 1]; e++) ' + spmv_body +
                     ' vt r = DG[i] * x[i] + acc / sqrt((double)STR[i]); y[i] = accumulate ? y[i] + r : r; } }')
            o.append(f'void spmv32(const xt* x, xt* y, i64 n, int accumulate, const u64* RP, const unsigned* JJ, const {mty}* MM,'
                     ' const double* DG, const unsigned short* STR' + vtarg + ') {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                     ' for (i64 i = 0; i < n; i++) { vt acc = 0.0; for (u64 e = RP[i]; e < RP[i + 1]; e++) ' + body32 +
                     ' vt r = DG[i] * TOV(x[i]) + acc / sqrt((double)STR[i]); y[i] = TOX(accumulate ? TOV(y[i]) + r : r); } }')
    o.append('DEV u64 bits_word(i64 w, i64 n) { u64 word = 0; for (int b = 0; b < 64 && w * 64 + b < n; b++)'
             ' if (stab_if_rep(state_at(w * 64 + b))) word |= 1ull << b; return word; }')
    if target == 'cuda':
        o.append('extern "C" __global__ void mv(const vt* __restrict__ x, vt* __restrict__ y, i64 n, int accumulate) {'
                 ' i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; vt r = row(i, x); y[i] = accumulate ? y[i] + r : r; }')
        o.append('extern "C" __global__ void mvr(const vt* __restrict__ x, vt* __restrict__ y, i64 r0, i64 n, int accumulate) {'
                 ' i64 i = r0 + (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; vt r = row(i, x); y[i] = accumulate ? y[i] + r : r; }')
        o.append('extern "C" __global__ void mvm(const vt* x0, const vt* x1, const vt* x2, const vt* x3, vt* __restrict__ y, i64 r0, i64 n, int accumulate) {'
                 ' i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (t >= n) return; vt r = rowm(r0 + t, x0, x1, x2, x3); y[t] = accumulate ? y[t] + r : r; }')
        if batch and batch_lanes:
            assert not batch_fp32
            o.append('#define XVAL(j) x[(i64)(j) * RB + q]')
            o.append('DEV vt rowq(i64 i, const vt* x, int q) { ' + head)
            o += ['  '+r for r in rows] + [tail, '#undef XVAL']
            o.append('extern "C" __global__ void mvb(const vt* __restrict__ x, vt* __restrict__ y, i64 n, int accumulate) {'
                     ' i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x; i64 i = t / RB; int q = (int)(t - i * RB); if (i >= n) return;'
                     ' vt r = rowq(i, x, q); y[t] = accumulate ? y[t] + r : r; }')
        elif batch:
            o.append('extern "C" __global__ void mvb(const xb* __restrict__ x, xb* __restrict__ y, i64 n, int accumulate) {'
                     ' i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; vt r[RB]; rowb(i, x, r);'
                     ' for (int q = 0; q < RB; q++) { i64 t = i * RB + q; y[t] = accumulate ? BTOX(BTOV(y[t]) + r[q]) : BTOX(r[q]); } }')
            o.append('extern "C" __global__ void mvbr(const xb* __restrict__ x, xb* __restrict__ y, i64 r0, i64 n, int accumulate) {'
                     ' i64 i = r0 + (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; vt r[RB]; rowb(i, x, r);'
                     ' for (int q = 0; q < RB; q++) { i64 t = i * RB + q; y[t] = accumulate ? BTOX(BTOV(y[t]) + r[q]) : BTOX(r[q]); } }')
        o.append('extern "C" __global__ void rep_bits(u64* out, i64 nw, i64 n) {'
                 ' i64 w = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (w < nw) out[w] = bits_word(w, n); }')
        o.append('extern "C" __global__ void make_reps(st_t* st, i64 nw) {'
                 ' i64 w = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (w >= nw) return; u64 word = BITS[w]; i64 k = (i64)RANK[w];'
                 ' while (word) { int b = __ffsll(word) - 1; word &= word - 1; st[k++] = state_at(w * 64 + b); } }')
    else:
        o.append('void mv(const vt* x, vt* y, i64 n, int accumulate) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                 ' for (i64 i = 0; i < n; i++) { vt r = row(i, x); y[i] = accumulate ? y[i] + r : r; } }')
        o.append('void mvr(const vt* x, vt* y, i64 r0, i64 n, int accumulate) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                 ' for (i64 i = r0; i < n; i++) { vt r = row(i, x); y[i] = accumulate ? y[i] + r : r; } }')
        if batch:
            o.append('void mvb(const xb* x, xb* y, i64 n, int accumulate) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                     ' for (i64 i = 0; i < n; i++) { vt r[RB]; rowb(i, x, r);'
                     ' for (int q = 0; q < RB; q++) y[i * RB + q] = accumulate ? BTOX(BTOV(y[i * RB + q]) + r[q]) : BTOX(r[q]); } }')
            o.append('void mvbr(const xb* x, xb* y, i64 r0, i64 n, int accumulate) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                     ' for (i64 i = r0; i < n; i++) { vt r[RB]; rowb(i, x, r);'
                     ' for (int q = 0; q < RB; q++) y[i * RB + q] = accumulate ? BTOX(BTOV(y[i * RB + q]) + r[q]) : BTOX(r[q]); } }')
        o.append('void rep_bits(u64* out, i64 nw, i64 n) {\n#pragma omp parallel for schedule(dynamic, 256)\n'
                 ' for (i64 w = 0; w < nw; w++) out[w] = bits_word(w, n); }')
        o.append('void make_reps(st_t* st, i64 nw) {\n#pragma omp parallel for schedule(dynamic, 256)\n'
                 ' for (i64 w = 0; w < nw; w++) { u64 word = BITS[w]; i64 k = (i64)RANK[w];'
                 ' while (word) { int b = __builtin_ctzll(word); word &= word - 1; st[k++] = state_at(w * 64 + b); } } }')
        o.append('double vdot_re(const double* a, const double* b, i64 m) { double s = 0.0;\n#pragma omp parallel for reduction(+:s) schedule(static)\n'
                 ' for (i64 i = 0; i < m; i++) s += a[i] * b[i]; return s; }')
        o.append('void axpy(double c, const double* x, double* y, i64 m) {\n#pragma omp parallel for schedule(static)\n'
                 ' for (i64 i = 0; i < m; i++) y[i] += c * x[i]; }')
        o.append('void scale2(double* u, double cu, double* v, double cv, i64 m) {\n#pragma omp parallel for schedule(static)\n'
                 ' for (i64 i = 0; i < m; i++) { u[i] *= cu; v[i] *= cv; } }')
    return '\n'.join(o)+'\n'


def spmm_source(real, target, K, fp32=False, fp16=False, mm='unsigned short'):
    """Stored connectivity (compact format: 4-byte column, 2-byte code into the value table VT, per row the diagonal DG and
    the stabiliser order STR) applied to K interleaved vectors:  y[i][k] (+)= DG[i] x[i][k] + sum_e VT[MM[e]] x[JJ[e]][k] / sqrt(STR[i]).
    fp32: single-precision vectors (xt), sums and matrix values in double."""
    if target == 'cuda':
        vt = 'double' if real else 'complex<double>'
        red = ('acc[k] += __shfl_down_sync(0xffffffffu, acc[k], off, 8);' if real else
               '{ double re = acc[k].real(), im = acc[k].imag(); re += __shfl_down_sync(0xffffffffu, re, off, 8);'
               ' im += __shfl_down_sync(0xffffffffu, im, off, 8); acc[k] = vt(re, im); }')
        xt = (('#include <cuda_fp16.h>\ntypedef __half xt;\n#define TOV(a) ((double)__half2float(a))\n#define TOX(a) (__float2half((float)(a)))' if real else
               '#include <cuda_fp16.h>\nstruct xt { __half re, im; };\nstatic __device__ __forceinline__ vt TOV(xt a) { return vt((double)__half2float(a.re), (double)__half2float(a.im)); }\nstatic __device__ __forceinline__ xt TOX(vt a) { xt r; r.re = __float2half((float)a.real()); r.im = __float2half((float)a.imag()); return r; }') if fp16 else
              'typedef vt xt;\n#define TOV(a) (a)\n#define TOX(a) (a)' if not fp32 else
              'typedef float xt;\n#define TOV(a) ((double)(a))\n#define TOX(a) ((float)(a))' if real else
              'typedef complex<float> xt;\n#define TOV(a) (vt((double)(a).real(), (double)(a).imag()))\n'
              '#define TOX(a) (xt((float)(a).real(), (float)(a).imag()))')
        return ('' if real else '#include <cupy/complex.cuh>\n')+f"""
typedef long long i64; typedef unsigned long long u64; typedef {vt} vt;
{xt}
#define KV {K}
extern "C" __global__ void spmm(const xt* __restrict__ x, xt* __restrict__ y, i64 n, int accumulate, const u64* __restrict__ RP,
                                const unsigned* __restrict__ JJ, const {mm}* __restrict__ MM, const double* __restrict__ DG,
                                const unsigned short* __restrict__ STR, const vt* __restrict__ VT) {{
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x; i64 i = t >> 3; int lane = (int)(t & 7);
    vt acc[KV];
    for (int k = 0; k < KV; k++) acc[k] = 0.0;
    if (i < n) for (u64 e = RP[i] + lane; e < RP[i + 1]; e += 8) {{
        vt v = VT[MM[e]]; const xt* xc = x + (i64)JJ[e] * KV;
        for (int k = 0; k < KV; k++) acc[k] += v * TOV(xc[k]);
    }}
    for (int k = 0; k < KV; k++) for (int off = 4; off > 0; off >>= 1) {red}
    if (lane == 0 && i < n) {{
        double s = 1.0 / sqrt((double)STR[i]);
        for (int k = 0; k < KV; k++) {{ vt r = DG[i] * TOV(x[i * KV + k]) + acc[k] * s; y[i * KV + k] = accumulate ? TOX(TOV(y[i * KV + k]) + r) : TOX(r); }}
    }}
}}
"""
    vt = 'double' if real else 'double _Complex'
    return f"""
#include <math.h>
#include <complex.h>
typedef long long i64; typedef unsigned long long u64; typedef {vt} vt;
#define KV {K}
void spmm(const vt* x, vt* y, i64 n, int accumulate, const u64* RP, const unsigned* JJ, const {mm}* MM, const double* DG,
          const unsigned short* STR, const vt* VT) {{
#pragma omp parallel for schedule(dynamic, 1024)
    for (i64 i = 0; i < n; i++) {{
        vt acc[KV];
        for (int k = 0; k < KV; k++) acc[k] = 0.0;
        for (u64 e = RP[i]; e < RP[i + 1]; e++) {{
            vt v = VT[MM[e]]; const vt* xc = x + (i64)JJ[e] * KV;
            for (int k = 0; k < KV; k++) acc[k] += v * xc[k];
        }}
        double s = 1.0 / sqrt((double)STR[i]);
        for (int k = 0; k < KV; k++) {{ vt r = DG[i] * x[i * KV + k] + acc[k] * s; y[i * KV + k] = accumulate ? y[i * KV + k] + r : r; }}
    }}
}}
"""


class SubEngine:
    """Lowest levels of one symmetry sector (momentum k, 1D irrep of the little group, optional spin flip) of a
    spin-1/2 model on a cluster (edc_lattice.Cluster or edc_lattices.LatticeCluster)."""

    def __init__(self, model, nup, cl, k, irrep, flip=0, device='cpu', top=None, wmax=22, store_states=None, block=256,
                 devices=None, rank64=None, point_ops=None, incremental=None, stored=None, precision='fp64', reserve_vectors=3):
        t0 = time.perf_counter()
        L = model.n
        assert L <= 64 and flip in (0, 1, -1) and (flip == 0 or 2*nup == L)
        if point_ops is None:
            perms, chars, ops = cl.group(k, irrep)
        elif hasattr(cl, 'irrep_table'):                 # irreps of the little group restricted to point_ops
            perms, chars, ops = cl.group(k, irrep, allowed=point_ops)
        else:
            import edc_lattice as el
            perms, chars, ops = cl.group(k, irrep, [o for o in cl.little_group(k) if o in el.IRREPS[irrep] and o in point_ops])
        chars = np.array(chars, complex)
        T = list(top) if top is not None else choose_top(cl, perms, wmax)
        w = len(T)
        # relabel: non-T sites first (in order), T sites on the top w bits
        order = [t for t in range(L) if t not in set(T)]+sorted(T)
        new = {old: i for i, old in enumerate(order)}
        perms = [[new[p[order[i]]] for i in range(L)] for p in perms]
        self._order, self._new, self._perms, self.flip = order, new, perms, flip
        m2 = self._relabel(model)
        self._m2 = m2
        Tn = set(range(L-w, L))
        Kidx = [gi for gi, p in enumerate(perms) if {p[a] for a in Tn} == Tn]
        assert Kidx[0] == 0
        # right cosets K g, labelled by the preimage of T
        cos = {}
        for gi, p in enumerate(perms):
            pre = frozenset(a for a in range(L) if p[a] in Tn)
            cos.setdefault(pre, gi)
        creps = list(cos.values())
        nf = 2 if flip else 1
        fs = float(flip) if flip else 1.0
        # K acting on the w top bits; elements kk = 2 * (index in K) + flip bit
        x = np.arange(1 << w, dtype=np.int64)
        wmask = (1 << w)-1
        mint = np.full(1 << w, np.iinfo(np.int64).max); karg = np.zeros(1 << w, np.int64)
        kperm_w = []
        for kk, gi in enumerate(Kidx):
            p = perms[gi]
            img = np.zeros_like(x)
            for b in range(w):
                img |= ((x >> b) & 1) << (p[L-w+b]-(L-w))
            kperm_w.append(img)
            for f in range(nf):
                im = img ^ wmask if f else img
                better = im < mint
                mint = np.where(better, im, mint); karg = np.where(better, 2*kk+f, karg)
        mins = np.flatnonzero(mint == x)                     # strings that are their own K-orbit minimum
        stab = [[] for _ in range(len(mins))]
        for kk in range(len(Kidx)):
            img = kperm_w[kk]
            for f in range(nf):
                im = (img ^ wmask) if f else img
                for t in np.flatnonzero(im[mins] == mins):
                    stab[t].append(2*kk+f)
        sptr = np.zeros((1 << w)+1, np.int32)
        cnt = np.zeros(1 << w, np.int32); cnt[mins] = [len(s) for s in stab]
        sptr[1:] = np.cumsum(cnt)
        sidx = np.zeros(max(int(sptr[-1]), 1), np.uint16)
        for t, mv in enumerate(mins):
            sidx[sptr[mv]:sptr[mv+1]] = sorted(stab[t])          # identity (0) first
        assert len(Kidx)*nf < 65536
        # byte tables for the full action of K elements
        nbytes = (L+7)//8
        byte = np.arange(256, dtype=np.uint64)
        ptk = np.zeros((len(Kidx), nbytes, 256), np.uint64)
        for kk, gi in enumerate(Kidx):
            p = perms[gi]
            for c in range(nbytes):
                v = np.zeros(256, np.uint64)
                for bi in range(8):
                    a = 8*c+bi
                    if a < L:
                        v |= ((byte >> np.uint64(bi)) & np.uint64(1)) << np.uint64(p[a])
                ptk[kk, c] = v
        ck = np.array([chars[gi]*(fs if f else 1.0) for gi in Kidx for f in range(nf)])
        ck_full = np.zeros(2*len(Kidx), complex); ck_full[:len(ck)] = ck
        if nf == 1:          # index = 2*kk: spread
            ck_full = np.zeros(2*len(Kidx), complex); ck_full[0::2] = ck
        cc = chars[creps]
        self._chars_all = np.concatenate([chars, chars*fs]) if flip else chars
        self.real = bool(np.all(np.abs(chars.imag) < 1e-12))
        # index of representatives: blocks of rest strings (L - w bits, complementary popcount) per top-part minimum
        NR = L-w
        pq = nup-np.bitwise_count(mins)
        sizes = np.array([math.comb(NR, int(p)) if 0 <= p <= NR else 0 for p in pq], np.int64)
        offq = np.zeros(len(mins)+1, np.int64); offq[1:] = np.cumsum(sizes)
        qidx = np.full(1 << w, -1, np.int32); qidx[mins] = np.arange(len(mins), dtype=np.int32)
        lows, off, _, jb, _ = edc.lin_tables(NR, 0)
        jap = np.stack([edc.lin_tables(NR, p)[2] for p in range(NR+1)])
        self.host = dict(MINT=mint.astype(np.uint32), KARG=karg.astype(np.uint16), SPTR=sptr, SIDX=sidx, PTK=ptk.ravel(),
                         CKR=ck_full.real.copy(), CKI=ck_full.imag.copy(), CCR=cc.real.copy(), CCI=cc.imag.copy(),
                         JAP=jap.ravel(), JB=jb, LOWS=lows.astype(np.uint64), OFF=off, QIDX=qidx, QMIN=mins.astype(np.uint32), OFFQ=offq,
                         CPL=cpl_array(m2))
        self.top, self.w, self.nK, self.ncos, self.group_order = T, w, len(Kidx)*nf, len(creps), len(perms)*nf
        self.nq, self.full_dim, self.device, self.store_states = len(mins), math.comb(L, nup), device, store_states
        nbits = int(offq[-1]); self.index_bits = nbits
        target = 'cuda' if device == 'gpu' else 'c'
        self.rank64 = rank64 if rank64 is not None else nbits >= 2**32 - 1     # prefix counts may exceed 32 bits
        self.rtype = np.uint64 if self.rank64 else np.uint32
        self.incremental = INCREMENTAL if incremental is None else bool(incremental)
        self._src_args = (L, w, nup, [perms[c] for c in creps], len(Kidx), nbytes, self.real, flip, target, self.rank64)
        self.src = self._source(m2)
        self.tables_seconds = time.perf_counter()-t0
        nw = (nbits+63)//64
        self.nwd = nw
        self.devices = list(devices) if devices else None
        if self.devices and len(self.devices) > 1:
            self._setup_multi(nbits, nw, block)
        elif device == 'gpu':
            self._setup_gpu(nbits, nw, block)
        else:
            self._setup_cpu(nbits, nw)
        self.reserve_vectors = reserve_vectors                      # double-precision vectors the GPU store leaves room for
        policy = STORED if stored is None else stored
        if policy and not (self.devices and len(self.devices) > 1) and self.dim >= (STORED_MIN_DIM if policy == 'auto' else 1):
            budget = None
            if device != 'gpu':
                import os
                ram = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
                budget = 0.5*ram - 3*self.dim*(8 if self.real else 16)
            import os as _os
            frac = _os.environ.get('EDC_STORED_FRACTION')            # test hook: budget = this fraction of the full need
            ok = self.build_stored(budget, partial=(policy == 'partial'))
            if frac and ok and policy == 'partial':
                full = self.stored_bytes; del self._st
                ok = self.build_stored(float(frac)*full, partial=True)
            assert ok or policy in ('auto', 'partial'), 'stored connectivity does not fit'
        # single-precision vectors (double accumulation): the stored product (one vector or several) and the on-the-fly
        # multi-vector product (batched) have them; the on-the-fly product of one vector does not
        assert precision in ('fp64', 'fp32', 'fp16')               # fp16: storage of the vectors of batched() only (GPU)
        self.fp32 = precision == 'fp32'; self.fp16 = precision == 'fp16'
        self.setup_seconds = time.perf_counter()-t0

    ORDER = ['MINT', 'KARG', 'SPTR', 'SIDX', 'PTK', 'CKR', 'CKI', 'CCR', 'CCI', 'JAP', 'JB', 'LOWS', 'OFF', 'QIDX', 'QMIN', 'OFFQ',
             'BITS', 'RANK', 'ST', 'CPL']

    def _setup_cpu(self, full, nw):
        self.xp = np
        self.lib, self.compile_seconds, _ = edc.compile_source(self.src, 'sub')
        self.arrs = dict(self.host); self.arrs.update(BITS=np.zeros(1, np.uint64), RANK=np.zeros(1, self.rtype), ST=np.zeros(1, np.uint64))
        bits = np.zeros(nw, np.uint64); self._bind_cpu()
        self.lib.rep_bits(bits.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(nw), ctypes.c_int64(full))
        cnt = _popcount64(bits).astype(np.uint64); incl = np.cumsum(cnt)
        self.dim = n = int(incl[-1]) if nw else 0
        self.arrs.update(BITS=bits, RANK=(incl-cnt).astype(self.rtype)); self._bind_cpu()
        self.store = bool(self.store_states if self.store_states is not None else 8*n <= (4 << 30))
        if self.store:
            st = np.zeros(max(n, 1), np.uint64)
            self.lib.make_reps(st.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(nw))
            self.arrs.update(ST=st)
        self._bind_cpu()
        self.device_gib = sum(a.nbytes for a in self.arrs.values())/2**30

    def _relabel(self, model):
        """The model on the engine's site labels (T sites on the top bits)."""
        L, new, order = model.n, self._new, self._order
        m2 = edc.SpinHalfModel(L)
        for (i, j), (jxy, jz) in model.bonds.items():
            oi, oj = model.orient.get((i, j), (i, j))
            m2.exchange(new[oi], new[oj], jxy); m2.zz(new[oi], new[oj], jz)
        m2.h = np.array([model.h[order[i]] for i in range(L)])
        m2.keep_zero = getattr(model, 'keep_zero', False)
        return m2

    def _source(self, m2, batch=0, stored=0, batch_fp32=False, batch_lanes=False, batch_fp16=False):
        L, w, nup, cosets, nK, nbytes, real, flip, target, rank64 = self._src_args
        return source(L, w, m2, nup, cosets, nK, nbytes, real, flip, target, rank64, batch, self.incremental, stored,
                      batch_fp32, batch_lanes, batch_fp16)

    def _mroot(self):
        """Smallest M with every character an M-th root of unity (phases are stored as m, exp(2 pi i m / M))."""
        c = self._chars_all
        for M in range(1, 4096):
            if np.max(np.abs(c**M-1)) < 1e-8:
                return 2 if (M == 1 and self.real) else M
        raise ValueError('characters are not roots of unity of order < 4096')

    def _coupling_classes(self, cpl=None, bcls=None):
        """Bonds grouped by their Jxy/2 value: (class of each bond slot f*L+a, value of each class).  With bcls given,
        checks that the couplings are still equal inside each class (returns None otherwise)."""
        c = self.host['CPL'] if cpl is None else cpl
        L = self._m2.n
        fams = edc.families(self._m2)
        slots = [fi*L+a for fi, f in enumerate(fams) for a in f if a != '_meta']
        if bcls is None:
            vals = sorted({float(c[t]) for t in slots})
            bcls = [0]*(len(fams)*L)
            for t in slots:
                bcls[t] = vals.index(float(c[t]))
            return bcls, vals
        vals = {}
        for t in slots:
            if vals.setdefault(bcls[t], float(c[t])) != float(c[t]):
                return None
        return bcls, [vals[q] for q in range(len(vals))]

    def _vt_table(self, cpl=None):
        """Values of the compact codes: VT[((class * NSJ) + stabiliser index) * M + m] = Jxy/2 sqrt(s) exp(2 pi i m / M)."""
        st = self._st
        _, cls = self._coupling_classes(cpl, st['bcls'])
        M, divs = st['M'], st['divs']
        ph = np.exp(2j*np.pi*np.arange(M)/M)
        vt = (np.array(cls)[:, None, None]*np.sqrt(np.array(divs, float))[None, :, None]*ph[None, None, :]).ravel()
        return vt.real.copy() if self.real else vt

    def build_stored(self, budget_bytes=None, fmt='auto', partial=False):
        """Write every matrix element once (count pass, fill pass) and switch matvec to the stored product.
        partial: when the entries of all rows do not fit, store those of the first rows that do (at least 5 % of them) and
        compute the other rows on the fly in every product: time between the two products, in proportion to the rows
        stored (single vectors: double precision; batched(): every precision of the stored multi-vector product).
        The default budget on a GPU leaves room for self.reserve_vectors (3) double-precision vectors of the block.
        fmt 5: u32 column + u8 code into a value table (fewer than 256 codes: low-symmetry blocks, the large ones); fmt 6:
        the same with a u16 code (every bond family has one coupling; few codes); fmt 8: u32 column
        + u32 code (phase, bond, stabiliser); 'auto' takes the smallest that is possible.  Returns False (and keeps the on-the-fly
        product) when the entries exceed budget_bytes or the format limits (dimension < 2^32, |G| < 1024, M < 4096)."""
        assert not (self.devices and len(self.devices) > 1), 'stored: one device only'
        if self.dim == 0 or self.dim >= 2**32 or self.group_order >= 1024:
            return False
        M = self._mroot()
        divs = [d for d in range(1, self.group_order+1) if self.group_order % d == 0]
        bcls, cls = self._coupling_classes()
        compact = fmt in ('auto', 5, 6) and len(cls)*len(divs)*M < 65536
        code8 = compact and fmt in ('auto', 5) and len(cls)*len(divs)*M < 256    # one-byte codes: 5 bytes per entry
        if (fmt == 6 and not compact) or (fmt == 5 and not code8):
            return False
        src = self._source(self._m2, stored=dict(M=M, divs=divs, bcls=bcls, code8=code8) if compact else M)
        esize = 5 if code8 else 6 if compact else 8
        mdt = np.uint8 if code8 else np.uint16 if compact else np.uint32
        n = self.dim
        vsize = 8 if self.real else 16
        t0 = time.perf_counter()
        if self.device == 'gpu':
            import cupy as cp
            mod = cp.RawModule(code=src, options=('-std=c++17',))
            self._bind_gpu(mod)
            cnt = cp.empty(n, cp.uint32)
            blk = self.block
            mod.get_function('scnt')(((n+blk-1)//blk,), (blk,), (cnt, np.int64(n)))
            RP = cp.zeros(n+1, cp.uint64); cp.cumsum(cnt, dtype=cp.uint64, out=RP[1:]); del cnt
            nnz = int(RP[-1])
            need = nnz*esize + n*18
            free, _ = cp.cuda.runtime.memGetInfo()
            budget = budget_bytes if budget_bytes is not None else 0.85*free - float(getattr(self, 'reserve_vectors', 3))*n*vsize
            ns = n
            if need > budget:
                if partial:                                      # rows 0..ns-1: entries, diagonal, stabiliser and row pointer fit
                    ns = int(cp.searchsorted(RP[1:]*esize+cp.arange(1, n+1, dtype=cp.uint64)*18, cp.asarray(max(budget, 0.0)), side='right'))
                if ns < 0.05*n or ns == n:
                    del RP; cp.get_default_memory_pool().free_all_blocks()
                    return False
                RP = RP[:ns+1].copy(); cp.get_default_memory_pool().free_all_blocks()
                nnz = int(RP[-1]); need = nnz*esize + ns*18
            JJ = cp.empty(nnz, cp.uint32); MM = cp.empty(nnz, mdt)
            DG = cp.empty(ns, cp.float64); STR = cp.empty(ns, cp.uint16)
            mod.get_function('sfill')(((ns+blk-1)//blk,), (blk,), (RP, JJ, MM, DG, STR, np.int64(ns)))
            cp.cuda.Device().synchronize()
            self._st = dict(M=M, divs=divs, bcls=bcls, compact=compact, mod=mod, k=mod.get_function('spmv'), k32=mod.get_function('spmv32'), RP=RP, JJ=JJ, MM=MM, DG=DG, STR=STR,
                            rows=ns, kr=mod.get_function('mvr'))
            if compact:
                self._st['VT'] = cp.asarray(self._vt_table())
        else:
            lib, _, _ = edc.compile_source(src, 'subst')
            self._bind_cpu(lib)
            P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            cnt = np.empty(n, np.uint32)
            lib.scnt(P(cnt), ctypes.c_int64(n))
            RP = np.zeros(n+1, np.uint64); np.cumsum(cnt, dtype=np.uint64, out=RP[1:]); del cnt
            nnz = int(RP[-1])
            need = nnz*esize + n*18
            ns = n
            if budget_bytes is not None and need > budget_bytes:
                if partial:
                    ns = int(np.searchsorted(RP[1:]*esize+np.arange(1, n+1, dtype=np.uint64)*18, max(budget_bytes, 0.0), side='right'))
                if ns < 0.05*n or ns == n:
                    return False
                RP = RP[:ns+1].copy(); nnz = int(RP[-1]); need = nnz*esize + ns*18
            JJ = np.empty(nnz, np.uint32); MM = np.empty(nnz, mdt); DG = np.empty(ns); STR = np.empty(ns, np.uint16)
            lib.sfill(P(RP), P(JJ), P(MM), P(DG), P(STR), ctypes.c_int64(ns))
            self._st = dict(M=M, divs=divs, bcls=bcls, compact=compact, lib=lib, RP=RP, JJ=JJ, MM=MM, DG=DG, STR=STR, rows=ns)
            if compact:
                self._st['VT'] = np.ascontiguousarray(self._vt_table())
        self._st['mm'] = 'unsigned char' if code8 else 'unsigned short'
        self.stored_nnz, self.stored_bytes, self.stored_format, self.stored_rows = nnz, need, esize, ns
        self.stored_build_seconds = time.perf_counter()-t0
        return True

    def _batched_stored(self, R, rows=None):
        """Y (+)= H X for R interleaved vectors with the stored connectivity in the compact format: column and value code of
        every entry are read once, and one gather fetches the R components of the neighbour."""
        st = self._st; n = self.dim; nr = n if rows is None else int(rows)   # nr < n: only the stored rows (partial store)
        if self.device == 'gpu':
            import cupy as cp
            kern = cp.RawKernel(spmm_source(self.real, 'cuda', R, fp32=getattr(self, 'fp32', False), fp16=getattr(self, 'fp16', False), mm=st['mm']), 'spmm'); blk = self.block
            def apply(X, Y, accumulate=False):
                assert X.shape[:2] == (n, R) and X.flags.c_contiguous and Y.flags.c_contiguous
                kern(((8*nr+blk-1)//blk,), (blk,), (X, Y, np.int64(nr), np.int32(1 if accumulate else 0), st['RP'], st['JJ'], st['MM'], st['DG'],
                                                 st['STR'], st['VT']))
                return Y
        else:
            assert not getattr(self, 'fp32', False), 'multi-vector product with single-precision vectors: GPU only'
            lib, _, _ = edc.compile_source(spmm_source(self.real, 'c', R, mm=st['mm']), 'subm')
            P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            def apply(X, Y, accumulate=False):
                assert X.shape == (n, R) and X.flags.c_contiguous and Y.flags.c_contiguous
                lib.spmm(P(X), P(Y), ctypes.c_int64(nr), ctypes.c_int(1 if accumulate else 0), P(st['RP']), P(st['JJ']), P(st['MM']), P(st['DG']),
                         P(st['STR']), P(st['VT']))
                return Y
        return apply

    def batched(self, R):
        """Y (+)= H X for R vectors stored interleaved (X[j, r] = component j of vector r, C order, shape (dim, R)),
        each matrix element computed once for all R vectors.  One device."""
        assert not (self.devices and len(self.devices) > 1), 'batched: one device only'
        st = getattr(self, '_st', None)
        ns = st.get('rows', self.dim) if st is not None else 0
        if st is not None and st['compact'] and ns == self.dim:      # stored connectivity: entries read once for all R vectors
            return self._batched_stored(R)
        part = self._batched_stored(R, rows=ns) if (st is not None and st['compact'] and 0 < ns < self.dim) else None
        import os
        f32 = bool(getattr(self, 'fp32', False)); lanes = bool(os.environ.get('EDC_BATCH_LANES')) and self.device == 'gpu' and not f32
        f16 = bool(getattr(self, 'fp16', False)); assert not (f16 and self.device != 'gpu'), 'half-precision vectors: GPU only'
        src = self._source(self._m2, batch=R, batch_fp32=f32, batch_lanes=lanes and not f16, batch_fp16=f16)
        if self.device == 'gpu':
            import cupy as cp
            mod = cp.RawModule(code=src, options=('-std=c++17',))
            kern = mod.get_function('mvb')
            grid = (self.dim*(R if (lanes and not f16) else 1)+self.block-1)//self.block
            if part is not None:                                 # rows below ns from the store, the others on the fly
                assert not lanes, 'partly stored connectivity: the one-thread-per-row multi-vector kernel'
                kr = mod.get_function('mvbr'); gr = (self.dim-ns+self.block-1)//self.block
                def apply(X, Y, accumulate=False):
                    part(X, Y, accumulate)
                    self._bind_gpu(mod)
                    kr((gr,), (self.block,), (X, Y, np.int64(ns), np.int64(self.dim), np.int32(1 if accumulate else 0)))
                    return Y
                return apply
            def apply(X, Y, accumulate=False):
                assert X.shape[:2] == (self.dim, R) and X.flags.c_contiguous and Y.flags.c_contiguous
                self._bind_gpu(mod)
                kern((grid,), (self.block,), (X, Y, np.int64(self.dim), np.int32(1 if accumulate else 0)))
                return Y
        else:
            lib, _, _ = edc.compile_source(src, 'subb')
            if part is not None:
                def apply(X, Y, accumulate=False):
                    part(X, Y, accumulate)
                    self._bind_cpu(lib)
                    lib.mvbr(X.ctypes.data_as(ctypes.c_void_p), Y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(ns), ctypes.c_int64(self.dim),
                             ctypes.c_int(1 if accumulate else 0))
                    return Y
                return apply
            def apply(X, Y, accumulate=False):
                assert X.shape == (self.dim, R) and X.flags.c_contiguous and Y.flags.c_contiguous
                self._bind_cpu(lib)
                lib.mvb(X.ctypes.data_as(ctypes.c_void_p), Y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(self.dim),
                        ctypes.c_int(1 if accumulate else 0))
                return Y
        return apply

    def _bind_cpu(self, lib=None, cpl=None):
        lib = lib or self.lib
        if lib is self.lib:
            self.arrs = {k: np.ascontiguousarray(v) for k, v in self.arrs.items()}
            self._ptrs = (ctypes.c_void_p*len(self.ORDER))(*[self.arrs[k].ctypes.data for k in self.ORDER])
        ptrs = self._ptrs
        if cpl is not None:                       # another operator's couplings on the same tables
            ptrs = (ctypes.c_void_p*len(self.ORDER))(*[cpl.ctypes.data if k == 'CPL' else self.arrs[k].ctypes.data for k in self.ORDER])
        lib.bind(ptrs, ctypes.c_longlong(self.nq), ctypes.c_longlong(self.nwd), ctypes.c_int(int(getattr(self, 'store', False))))
        _BOUND[('cpu', lib._name)] = id(self) if lib is self.lib else None

    def _bind_gpu(self, mod=None, cpl=None):
        mod = mod or self.mod
        for nm, a in self.dev.items():
            if nm == 'CPL' and cpl is not None:
                a = cpl
            mod.get_global(nm).copy_from_host(np.array([a.data.ptr], np.uint64).ctypes.data, 8)
        for nm, v, dt in (('NQ', self.nq, np.int64), ('NWD', self.nwd, np.int64), ('STOREF', int(getattr(self, 'store', False)), np.int32)):
            mod.get_global(nm).copy_from_host(np.array([v], dt).ctypes.data, np.dtype(dt).itemsize)
        _BOUND[('gpu', int(mod.get_global('ST').ptr))] = id(self) if mod is self.mod else None

    def retune(self, model):
        """Change the couplings (same bonds, new values).  The generated code depends only on the bond structure, so
        nothing is compiled: the run-time coupling table is overwritten in place and, with stored connectivity, the
        stored diagonal is recomputed (one pass).  Raises ValueError if the bonds or their families change."""
        m2 = self._relabel(model)
        if structure_key(m2) != structure_key(self._m2):
            raise ValueError('bond structure changed; build a new engine')
        c = cpl_array(m2)
        st0 = getattr(self, '_st', None)
        if st0 is not None and st0['compact'] and self._coupling_classes(c, st0['bcls']) is None:
            raise ValueError('compact stored entries: bonds that had equal couplings must stay equal; build with fmt=8')
        self.host['CPL'][:] = c
        if self.device == 'gpu':
            import cupy as cp
            for dev in ([self.dev] if not (self.devices and len(self.devices) > 1) else self.devt):
                dev['CPL'].set(c)
            cp.cuda.Device().synchronize()
        else:
            self.arrs['CPL'][:] = c
        st = getattr(self, '_st', None)
        if st is not None:
            if st['compact']:
                vt = self._vt_table()
                if self.device == 'gpu':
                    st['VT'].set(vt)
                else:
                    st['VT'][:] = vt
            if self.device == 'gpu':
                ns = st.get('rows', self.dim)
                self._bind_gpu(st['mod'])                        # engines with identical source share the module and its table pointers
                st['mod'].get_function('sdiag')(((ns+self.block-1)//self.block,), (self.block,), (st['DG'], np.int64(ns)))
            else:
                self._bind_cpu(st['lib'])
                st['lib'].sdiag(st['DG'].ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(st.get('rows', self.dim)))
        self._m2 = m2
        return self

    def operator(self, op_model):
        """y (+)= O x for another spin-1/2 model O on the same sites, in this sector, reusing this engine's tables
        (one device).  O must commute with every element of the sector's group; this is checked on its couplings."""
        assert not (self.devices and len(self.devices) > 1), 'operator: one device only'
        m2 = self._relabel(op_model)
        C = {k: tuple(v) for k, v in m2.bonds.items() if v[0] or v[1]}
        for p in self._perms:
            for (i, j), v in C.items():
                img = C.get((min(p[i], p[j]), max(p[i], p[j])), (0.0, 0.0))
                assert abs(img[0]-v[0]) < 1e-10 and abs(img[1]-v[1]) < 1e-10, 'operator not invariant under the group'
            assert np.allclose(m2.h[np.array(p)], m2.h), 'field not invariant under the group'
        assert not (self.flip and np.any(m2.h)), 'a field breaks the spin-flip symmetry'
        src = self._source(m2)
        cpl = cpl_array(m2)
        if self.device == 'gpu':
            import cupy as cp
            mod = cp.RawModule(code=src, options=('-std=c++17',))
            kern = mod.get_function('mv')
            cpl_d = cp.asarray(cpl)
            def apply(x, y, accumulate=False):
                self._bind_gpu(mod, cpl_d)
                kern((self.grid,), (self.block,), (x, y, np.int64(self.dim), np.int32(1 if accumulate else 0)))
                return y
        else:
            lib, _, _ = edc.compile_source(src, 'subop')
            def apply(x, y, accumulate=False):
                self._bind_cpu(lib, cpl)
                lib.mv(x.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(self.dim),
                       ctypes.c_int(1 if accumulate else 0))
                return y
        return apply

    def _setup_gpu(self, full, nw, block):
        import cupy as cp
        global _POPC
        if _POPC is None:
            _POPC = cp.ElementwiseKernel('uint64 x', 'uint32 c', 'c = __popcll(x)', 'edc_sub_popc')
        self.xp = cp
        self.mod = cp.RawModule(code=self.src, options=('-std=c++17',))
        self.dev = {nm: cp.asarray(a) for nm, a in self.host.items()}
        self.dev.update(BITS=cp.zeros(1, cp.uint64), RANK=cp.zeros(1, self.rtype), ST=cp.zeros(1, cp.uint64))
        bits = cp.empty(nw, cp.uint64); self._bind_gpu()
        self.mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (bits, np.int64(nw), np.int64(full)))
        cnt = _POPC(bits)                                  # uint32 per word, no temporaries
        incl = cp.cumsum(cnt, dtype=cp.uint64)
        self.dim = n = int(incl[-1]) if nw else 0
        incl -= cnt; rank = incl if self.rank64 else incl.astype(cp.uint32); del incl, cnt
        self.dev.update(BITS=bits, RANK=rank); self._bind_gpu()
        free, total = cp.cuda.runtime.memGetInfo()
        self.store = bool(self.store_states if self.store_states is not None else 8*n <= 0.15*free)
        if self.store:
            st = cp.empty(max(n, 1), cp.uint64)
            self.mod.get_function('make_reps')(((nw+block-1)//block,), (block,), (st, np.int64(nw)))
            self.dev.update(ST=st)
        self._bind_gpu()
        self.kern = self.mod.get_function('mv')
        self.block, self.grid = block, (max(n, 1)+block-1)//block
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        self.device_gib = sum(a.nbytes for a in self.dev.values())/2**30

    # ------------------------------------------------------------ several GPUs of one node
    def _setup_multi(self, nbits, nw, block):
        """Rows split in contiguous blocks over the devices; tables replicated; peer access between all devices."""
        import cupy as cp
        global _POPC
        devs = self.devices
        assert 2 <= len(devs) <= 4
        self.xp, self.store, self.block = cp, False, block
        for a in devs:
            with cp.cuda.Device(a):
                for b in devs:
                    if a != b:
                        try:
                            cp.cuda.runtime.deviceEnablePeerAccess(b)
                        except cp.cuda.runtime.CUDARuntimeError:
                            pass                                  # already enabled
        self.mods, self.devt = [], []
        for di, a in enumerate(devs):
            with cp.cuda.Device(a):
                if _POPC is None:
                    _POPC = cp.ElementwiseKernel('uint64 x', 'uint32 c', 'c = __popcll(x)', 'edc_sub_popc')
                mod = cp.RawModule(code=self.src, options=('-std=c++17',))
                dev = {nm: cp.asarray(v) for nm, v in self.host.items()}
                dev.update(BITS=cp.zeros(1, cp.uint64), RANK=cp.zeros(1, self.rtype), ST=cp.zeros(1, cp.uint64))
                self.mods.append(mod); self.devt.append(dev)
        # representatives found once, on the first device
        with cp.cuda.Device(devs[0]):
            self.mod, self.dev = self.mods[0], self.devt[0]
            self.bnd = [0, 1 << 62, 1 << 62, 1 << 62]
            self._bind_dev(0)
            bits = cp.empty(nw, cp.uint64)
            self.mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (bits, np.int64(nw), np.int64(nbits)))
            cnt = _POPC(bits); incl = cp.cumsum(cnt, dtype=cp.uint64)
            self.dim = n = int(incl[-1]) if nw else 0
            incl -= cnt; rank = incl if self.rank64 else incl.astype(cp.uint32); del incl, cnt
            self.devt[0].update(BITS=bits, RANK=rank)
        P = len(devs)
        cuts = [n*d//P for d in range(P+1)]
        self.rows = [(cuts[d], cuts[d+1]-cuts[d]) for d in range(P)]
        self.bnd = cuts[1:P]+[1 << 62]*(4-P)
        self.bnd = [0]+self.bnd[:3]
        for di in range(1, P):
            with cp.cuda.Device(devs[di]):
                self.devt[di].update(BITS=cp.asarray(self.devt[0]['BITS']), RANK=cp.asarray(self.devt[0]['RANK']))
        self.kerns = []
        for di in range(P):
            with cp.cuda.Device(devs[di]):
                self._bind_dev(di)
                self.kerns.append(self.mods[di].get_function('mvm'))
                cp.cuda.Stream.null.synchronize()
        self.device_gib = sum(a.nbytes for a in self.devt[0].values())/2**30

    def _bind_dev(self, di):
        mod, dev = self.mods[di], self.devt[di]
        for nm, a in dev.items():
            mod.get_global(nm).copy_from_host(np.array([a.data.ptr], np.uint64).ctypes.data, 8)
        for nm, v, dt in (('NQ', self.nq, np.int64), ('NWD', self.nwd, np.int64), ('STOREF', 0, np.int32),
                          ('BND1', self.bnd[1], np.int64), ('BND2', self.bnd[2], np.int64), ('BND3', self.bnd[3], np.int64)):
            mod.get_global(nm).copy_from_host(np.array([v], dt).ctypes.data, np.dtype(dt).itemsize)

    def mv_multi(self, xs, ys, accumulate=False):
        """ys[d] (+)= rows of H x on device d; xs: list of the per-device parts of x."""
        import cupy as cp
        P = len(self.devices)
        for a in self.devices:          # every device reads the others' parts of x: all pending updates must be finished
            cp.cuda.Device(a).synchronize()
        xp4 = list(xs)+[xs[-1]]*(4-P)
        for di, a in enumerate(self.devices):
            with cp.cuda.Device(a):
                r0, nl = self.rows[di]
                self.kerns[di](((max(nl, 1)+self.block-1)//self.block,), (self.block,),
                               (xp4[0], xp4[1], xp4[2], xp4[3], ys[di], np.int64(r0), np.int64(nl), np.int32(1 if accumulate else 0)))
        for a in self.devices:
            cp.cuda.Device(a).synchronize()

    def ground_energy_multi(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3):
        """Two-vector Lanczos with vectors split over the devices (same recurrence as edc_hubbard_sym.lanczos)."""
        import cupy as cp
        from scipy.linalg import eigvalsh_tridiagonal
        from edc_gpu import _dotc, _ddot
        from edc_hubbard_sym import _distinct
        devs, real = self.devices, self.real
        dt = cp.float64 if real else cp.complex128
        t0 = time.perf_counter()
        ax = []
        for a in devs:
            with cp.cuda.Device(a):
                ax.append(cp.ElementwiseKernel('T x, float64 a', 'T y', 'y = y + a * x', 'sub_axpy'))
        def dot(A, B):
            tot = 0.0
            for di, a in enumerate(devs):
                with cp.cuda.Device(a):
                    tot += _ddot(A[di], B[di]) if real else float(_dotc(A[di], B[di]).real)
            return tot
        def axpy(c, X, Y):
            for di, a in enumerate(devs):
                with cp.cuda.Device(a):
                    ax[di](X[di], c, Y[di])
        v, u = [], []
        for di, a in enumerate(devs):
            with cp.cuda.Device(a):
                r0, nl = self.rows[di]; rng = cp.random.default_rng(seed+di)
                x = cp.empty(nl, dt); ch = 1 << 26
                for i in range(0, nl, ch):
                    c = min(ch, nl-i)
                    x[i:i+c] = rng.standard_normal(c) if real else rng.standard_normal(c)+1j*rng.standard_normal(c)
                v.append(x); u.append(cp.zeros(nl, dt))
        nrm = math.sqrt(dot(v, v))
        for di, a in enumerate(devs):
            with cp.cuda.Device(a):
                v[di] *= 1.0/nrm
        A, B = [], []; e_old = None; hits = 0
        for it in range(1, maxit+1):
            self.mv_multi(v, u, accumulate=True)
            al = dot(v, u); axpy(-al, v, u); be = math.sqrt(dot(u, u))
            A.append(al); B.append(be)
            if nev == 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0] if it > 1 else al
                ok = e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e))
            else:
                ev = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, min(it-1, 3*nev+2))) if it > 1 else np.array([al])
                e = _distinct(ev, nev)
                ok = (e_old is not None and len(e) == nev and len(e_old) == nev and
                      all(abs(a-b) < tol*max(1.0, abs(a)) for a, b in zip(e, e_old)))
            hits = hits+1 if ok else 0
            if hits >= stable or be < 1e-12:
                return e, it, time.perf_counter()-t0
            e_old = e
            for di, a in enumerate(devs):
                with cp.cuda.Device(a):
                    u[di] *= 1.0/be; v[di] *= -be
            v, u = u, v
        return e_old, it, time.perf_counter()-t0

    def time_multi(self, reps=2):
        import cupy as cp
        dt = cp.float64 if self.real else cp.complex128
        xs, ys = [], []
        for di, a in enumerate(self.devices):
            with cp.cuda.Device(a):
                xs.append(cp.ones(self.rows[di][1], dt)); ys.append(cp.empty(self.rows[di][1], dt))
        self.mv_multi(xs, ys); best = np.inf
        for _ in range(reps):
            t = time.perf_counter(); self.mv_multi(xs, ys); best = min(best, time.perf_counter()-t)
        del xs, ys
        return best

    def _ensure_bound(self):
        key = self._bind_key()
        if _BOUND.get(key) != id(self):
            self._bind_gpu() if self.device == 'gpu' else self._bind_cpu()

    def _bind_key(self):
        return ('gpu', int(self.mod.get_global('ST').ptr)) if self.device == 'gpu' else ('cpu', self.lib._name)

    def matvec(self, x, y, accumulate=False):
        st = getattr(self, '_st', None)
        if st is not None:
            n = st.get('rows', self.dim); part = n < self.dim         # part: rows n.. are computed on the fly
            assert not (part and x.dtype in (np.float32, np.complex64)), 'partly stored connectivity: double-precision vectors'
            if self.device == 'gpu':
                blk = self.block
                args = (x, y, np.int64(n), np.int32(1 if accumulate else 0), st['RP'], st['JJ'], st['MM'], st['DG'], st['STR'])
                k = st['k32'] if x.dtype in (np.float32, np.complex64) else st['k']
                k(((8*n+blk-1)//blk,), (blk,), args+((st['VT'],) if st['compact'] else ()))
                if part:
                    key = ('gpu', int(st['mod'].get_global('ST').ptr))
                    if _BOUND.get(key) != ('st', id(self)):
                        self._bind_gpu(st['mod']); _BOUND[key] = ('st', id(self))
                    st['kr'](((self.dim-n+blk-1)//blk,), (blk,), (x, y, np.int64(n), np.int64(self.dim), np.int32(1 if accumulate else 0)))
            else:
                P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
                args = [P(x), P(y), ctypes.c_int64(n), ctypes.c_int(1 if accumulate else 0),
                        P(st['RP']), P(st['JJ']), P(st['MM']), P(st['DG']), P(st['STR'])]
                if st['compact']:
                    args.append(P(st['VT']))
                (st['lib'].spmv32 if x.dtype in (np.float32, np.complex64) else st['lib'].spmv)(*args)
                if part:
                    self._bind_cpu(st['lib'])
                    st['lib'].mvr(P(x), P(y), ctypes.c_int64(n), ctypes.c_int64(self.dim), ctypes.c_int(1 if accumulate else 0))
            return y
        assert x.dtype not in (np.float32, np.complex64, np.float16), 'single-precision vectors without stored connectivity: batched() only'
        self._ensure_bound()
        if self.device == 'gpu':
            self.kern((self.grid,), (self.block,), (x, y, np.int64(self.dim), np.int32(1 if accumulate else 0)))
        else:
            self.lib.mv(x.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(self.dim),
                        ctypes.c_int(1 if accumulate else 0))
        return y

    def time_matvec(self, reps=3):
        if self.devices and len(self.devices) > 1:
            return self.time_multi(reps)
        xp = self.xp
        dt = np.float64 if self.real else np.complex128
        x = xp.asarray(np.random.default_rng(0).standard_normal(self.dim).astype(dt)); y = xp.empty_like(x)
        best = np.inf
        for _ in range(reps+1):
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            t = time.perf_counter(); self.matvec(x, y)
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            best = min(best, time.perf_counter()-t)
        return best

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3, v0=None, keep=False, rtol=None, etol=None):
        if self.devices and len(self.devices) > 1:
            return self.ground_energy_multi(tol, maxit, seed, nev, stable)
        return lanczos(self, tol, maxit, seed, nev, stable, v0=v0, keep=keep, rtol=rtol, etol=etol)

    def scan(self, models, tol=1e-12, warm=False, nev=1):
        """Ground energies along a list of models with the same bonds (retune between points).
        warm: False (default) = every point from a random vector; 'mixed' (also True) = from the previous ground state
        plus half its norm of a random vector; 'guess' = from the previous ground state alone (fastest, but it follows
        the previous level through a crossing with a level of another hidden quantum number, e.g. total spin, and then
        returns wrong energies: 0.49 off in one test).
        Returns a list of dicts."""
        from edc_hubbard_sym import ground_state_vector
        from edc_scan import start_vector
        mode = {True: 'mixed', False: 'random'}.get(warm, warm)
        warm = mode != 'random'
        out, psi = [], None
        vsize = 8 if self.real else 16
        if self.device == 'gpu':
            import cupy as cp
            budget = 0.5*cp.cuda.runtime.memGetInfo()[0]
        else:
            import os
            budget = 0.25*os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        keep = warm and max(0, int(budget/(self.dim*vsize)))  # keep up to this many Krylov vectors instead of replaying
        if keep and keep < 8:
            keep = False
        for i, m in enumerate(models):
            if i:
                self.retune(m)
            t0 = time.perf_counter()
            E, it, _ = self.ground_energy(tol=tol, nev=nev, v0=start_vector(self, psi, mode, 500+i) if warm else None, keep=keep)
            rec = dict(E=E, iterations=it)
            if warm and self.dim > 400:
                psi, _, res = ground_state_vector(self); rec['residual'] = res
                rec['replay_iterations'] = 0 if 'psi' in self._lanczos else it
            rec['seconds'] = time.perf_counter()-t0
            out.append(rec)
        return out
