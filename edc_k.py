#!/usr/bin/env python3
"""edc_k — translation symmetry at any momentum (complex vectors), CPU / OpenMP.

Momentum state |r~> = N_r^{-1/2} sum_m chi(g_m)^* |m>, m = g_m(r) over the orbit of the representative r.
For a flip t of r':  (H x)_{r'} += c * sqrt(N_{r'} / N_{r(t)}) * conj(chi(g_t)) * x_{r(t)},  with g_t(r(t)) = t.
The table stores, for every state t of the Sz sector, the orbit index of t (4 bytes) and g_t (1 byte).
Representatives whose stabiliser carries a character != 1 have no state at this momentum and are dropped.
"""
import time, ctypes, cmath
import numpy as np
import edc
from edc import ptr, PD, PI64, PU32


def ring_characters(L, q):
    return [cmath.exp(-2j*cmath.pi*q*k/L) for k in range(L)]


def torus_characters(Lx, Ly, qx, qy):
    return [cmath.exp(-2j*cmath.pi*(qx*dx/Lx+qy*dy/Ly)) for dy in range(Ly) for dx in range(Lx)]


class KGen(edc.Gen):
    def __init__(self, model, nup, group, chars, schedule='dynamic, 4096'):
        super().__init__(model, nup, group, 1, schedule)
        self.chars = [complex(c) for c in chars]

    def source(self):
        edc.Gen.source(self)                       # fills family metadata; output not used
        L, LA = self.L, self.LA
        G = self.group
        low = (1 << LA)-1
        one = [abs(c-1) < 1e-9 for c in self.chars]
        o = ['#include <stdint.h>', '#include <math.h>', '#include <complex.h>', '#include <omp.h>',
             f'typedef {self.T} st_t; typedef {self.JAT} ja_t; typedef double _Complex cplx;',
             'static const ja_t *JA; static const uint32_t *JB, *TAB; static const uint8_t *GIX; static const st_t *ST;',
             'static const double *SQ, *ISQ; static cplx *XS; static int64_t NROW;',
             'void edc_bind(const ja_t* ja, const uint32_t* jb, const uint32_t* tab, const uint8_t* gix, const st_t* st,',
             '  const double* sq, const double* isq, cplx* xs, int64_t nrow){ JA=ja; JB=jb; TAB=tab; GIX=gix; ST=st; SQ=sq; ISQ=isq; XS=xs; NROW=nrow; }',
             'void edc_threads(int n){ omp_set_num_threads(n); }',
             f'#define FULLIDX(t) ((int64_t)JA[(t) >> {LA}] + JB[(t) & {self.c(low)}])']
        o.append('static const cplx CHIC[%d] = {%s};' % (len(G), ', '.join(f'{c.real!r} - ({c.imag!r}) * I' for c in self.chars)))
        for gi, perm in enumerate(G):
            o.append(f'static inline st_t G{gi}(st_t s){{ return {self.apply_expr(perm, "s")}; }}')
        tests = ' '.join(f'{{ st_t r = G{gi}(s); if (r < s) return 0; if (r == s) {{ st++; {"" if one[gi] else "bad = 1;"} }} }}'
                         for gi in range(1, len(G)))
        o.append(f'static inline int is_rep(st_t s, int* stab){{ int st = 1, bad = 0; {tests} *stab = st; return !bad; }}')
        # kernel
        body = []
        dconst = 0.0
        for fi, f in enumerate(self.fams):
            meta = f['_meta']
            ff = {a: v for a, v in f.items() if a != '_meta'}
            for a, (b, jxy, jz) in ff.items():
                dconst += jz/4
            body.append(f'{{ st_t u = {self.u_expr(ff)};')
            if meta['mz']:
                if meta['wz'] is not None:
                    body.append(f'd -= {2*meta["wz"]!r} * (double){self.POP}(u & {self.c(meta["mz"])});')
                else:
                    body.append(f'{{ st_t z = u & {self.c(meta["mz"])}; while (z) {{ int a = {self.CTZ}(z); z &= z - 1; d -= 2.0 * WZ{fi}[a]; }} }}')
            if meta['mxy']:
                cexpr = repr(meta['cxy']) if meta['cxy'] is not None else f'CXY{fi}[a]'
                body.append(f'{{ st_t fl = u & {self.c(meta["mxy"])}; while (fl) {{ int a = {self.CTZ}(fl); fl &= fl - 1; '
                            f'st_t t = s ^ FLIP{fi}[a]; int64_t f = FULLIDX(t); acc += {cexpr} * CHIC[GIX[f]] * XS[TAB[f]]; }} }}')
            body.append('}')
        h = self.m.h
        dconst = float(dconst)-float(h.sum())/2
        body += [f'd += {float(h[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if h[i] != 0]
        # tables of the families (same names as in edc.Gen)
        for fi, f in enumerate(self.fams):
            ff = {a: v for a, v in f.items() if a != '_meta'}
            flip = [0]*L; cxy = [0.0]*L; wz = [0.0]*L
            for a, (b, jxy, jz) in ff.items():
                flip[a] = (1 << a) | (1 << b); cxy[a] = jxy/2; wz[a] = jz/4
            o.append(f'static const st_t FLIP{fi}[{L}] = {{{", ".join(self.c(v) for v in flip)}}};')
            o.append(f'static const double CXY{fi}[{L}] = {{{", ".join(repr(float(v)) for v in cxy)}}};')
            o.append(f'static const double WZ{fi}[{L}] = {{{", ".join(repr(float(v)) for v in wz)}}};')
        o.append('void mv_free(const cplx* restrict x, cplx* restrict y){')
        o.append('  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW; i++) XS[i] = x[i] * ISQ[i];')
        o.append(f'  #pragma omp parallel for schedule({self.schedule})\n  for (int64_t i = 0; i < NROW; i++) {{')
        o.append(f'    st_t s = ST[i]; double d = {dconst!r}; cplx acc = 0.0;')
        o += ['    '+b for b in body]
        o.append('    y[i] = d * x[i] + SQ[i] * acc; }\n}')
        # setup
        o.append('void count_reps(const st_t* lows, const int64_t* off, int64_t* cnt){')
        o.append(f'  #pragma omp parallel for schedule(dynamic, 16)\n  for (int64_t h = 0; h < {1 << (L-LA)}; h++) {{ int64_t c = 0;')
        o.append(f'    int p = {self.nup} - {self.POP}((st_t)h); if (p >= 0 && p <= {LA})')
        o.append(f'      for (int64_t q = off[p]; q < off[p+1]; q++) {{ int stb; c += is_rep(((st_t)h << {LA}) | lows[q], &stb); }}')
        o.append('    cnt[h] = c; }\n}')
        o.append('void fill_reps(const st_t* lows, const int64_t* off, const int64_t* start, st_t* reps, double* sq, double* isq){')
        o.append(f'  #pragma omp parallel for schedule(dynamic, 16)\n  for (int64_t h = 0; h < {1 << (L-LA)}; h++) {{ int64_t o = start[h];')
        o.append(f'    int p = {self.nup} - {self.POP}((st_t)h); if (p < 0 || p > {LA}) continue;')
        o.append(f'    for (int64_t q = off[p]; q < off[p+1]; q++) {{ st_t s = ((st_t)h << {LA}) | lows[q]; int stb;')
        o.append(f'      if (is_rep(s, &stb)) {{ reps[o] = s; double n = {len(G)}.0 / stb; sq[o] = sqrt(n); isq[o] = 1.0 / sqrt(n); o++; }} }} }}\n}}')
        o.append('void fill_tab(uint32_t* tab, uint8_t* gix){')
        o.append('  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW; i++) { st_t s = ST[i];')
        o += [f'    {{ st_t t = G{gi}(s); int64_t f = FULLIDX(t); tab[f] = (uint32_t)i; gix[f] = {gi}; }}' for gi in range(len(G))]
        o.append('  }\n}')
        o.append('void lanczos_step(const cplx* restrict v, const cplx* restrict vp, cplx* restrict w, double bprev, double* ab){')
        o.append('  mv_free(v, w); double al = 0.0;')
        o.append('  #pragma omp parallel for schedule(static) reduction(+:al)\n  for (int64_t i = 0; i < NROW; i++) al += creal(conj(v[i]) * w[i]);')
        o.append('  double nr = 0.0;\n  #pragma omp parallel for schedule(static) reduction(+:nr)')
        o.append('  for (int64_t i = 0; i < NROW; i++) { cplx t = w[i] - al * v[i] - bprev * vp[i]; w[i] = t; nr += creal(t) * creal(t) + cimag(t) * cimag(t); }')
        o.append('  double be = sqrt(nr), ib = 1.0 / be;\n  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW; i++) w[i] *= ib;')
        o.append('  ab[0] = al; ab[1] = be; }')
        return '\n'.join(o)+'\n'


class KEngine:
    def __init__(self, model, nup, group, chars, threads=None):
        t0 = time.perf_counter()
        self.gen = KGen(model, nup, group, chars)
        self.lib, self.compile_seconds, _ = edc.compile_source(self.gen.source(), tag='kmom')
        if threads:
            self.lib.edc_threads(threads)
        L, LA = model.n, model.n//2
        st_np = np.uint32 if L <= 32 else np.uint64
        st_c = ctypes.c_uint32 if L <= 32 else ctypes.c_uint64
        lows, off, ja, jb, full = edc.lin_tables(L, nup)
        ja_np = np.uint32 if self.gen.JAT == 'uint32_t' else np.uint64
        ja_c = ctypes.c_uint32 if ja_np is np.uint32 else ctypes.c_uint64
        self.ja, self.jb, self.lows, self.off = ja.astype(ja_np), jb, lows.astype(st_np), off
        self.lib.edc_bind.argtypes = [ctypes.POINTER(ja_c), PU32, PU32, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(st_c),
                                      PD, PD, ctypes.c_void_p, ctypes.c_int64]
        self.lib.edc_bind(ptr(self.ja, ja_c), ptr(self.jb, ctypes.c_uint32), None, None, None, None, None, None, 0)
        ch = np.zeros(1 << (L-LA), np.int64)
        self.lib.count_reps.argtypes = [ctypes.POINTER(st_c), PI64, PI64]
        self.lib.count_reps(ptr(self.lows, st_c), ptr(off, ctypes.c_int64), ptr(ch, ctypes.c_int64))
        start = np.zeros_like(ch); start[1:] = np.cumsum(ch)[:-1]
        self.dim = int(ch.sum())
        self.st = np.empty(self.dim, st_np); self.sq = np.empty(self.dim); self.isq = np.empty(self.dim)
        self.lib.fill_reps.argtypes = [ctypes.POINTER(st_c), PI64, PI64, ctypes.POINTER(st_c), PD, PD]
        self.lib.fill_reps(ptr(self.lows, st_c), ptr(off, ctypes.c_int64), ptr(start, ctypes.c_int64),
                           ptr(self.st, st_c), ptr(self.sq, ctypes.c_double), ptr(self.isq, ctypes.c_double))
        # states of orbits that do not exist at this momentum point to a slot that always holds zero
        self.tab = np.full(full, self.dim, np.uint32); self.gix = np.zeros(full, np.uint8)
        self.xs = np.zeros(self.dim+1, np.complex128)
        self.lib.edc_bind(ptr(self.ja, ja_c), ptr(self.jb, ctypes.c_uint32), ptr(self.tab, ctypes.c_uint32),
                          ptr(self.gix, ctypes.c_uint8), ptr(self.st, st_c), ptr(self.sq, ctypes.c_double),
                          ptr(self.isq, ctypes.c_double), self.xs.ctypes.data, self.dim)
        self.lib.fill_tab.argtypes = [PU32, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.fill_tab(ptr(self.tab, ctypes.c_uint32), ptr(self.gix, ctypes.c_uint8))
        self.lib.mv_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.lanczos_step.argtypes = [ctypes.c_void_p]*3+[ctypes.c_double, PD]
        self.setup_seconds = time.perf_counter()-t0

    def matvec(self, x, y=None):
        y = np.empty_like(x) if y is None else y
        self.lib.mv_free(x.ctypes.data, y.ctypes.data)
        return y

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42):
        from scipy.linalg import eigvalsh_tridiagonal
        n = self.dim
        if n == 0:
            return None, 0, 0.0
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(n)+1j*rng.standard_normal(n); v /= np.linalg.norm(v)
        vp = np.zeros(n, np.complex128); w = np.empty(n, np.complex128); ab = np.zeros(2)
        A, B = [], []; bprev = 0.0; e_old = None
        t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.lib.lanczos_step(v.ctypes.data, vp.ctypes.data, w.ctypes.data, bprev, ptr(ab, ctypes.c_double))
            A.append(ab[0]); B.append(ab[1])
            if n == 1:
                return A[0], 1, time.perf_counter()-t0
            if it > 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0]
                if (e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e))) or it >= n or ab[1] < 1e-12:
                    return e, it, time.perf_counter()-t0
                e_old = e
            bprev = ab[1]
            vp, v, w = v, w, vp
        return e_old, it, time.perf_counter()-t0
