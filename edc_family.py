#!/usr/bin/env python3
"""edc_family — solve a one-parameter family H(lambda) = H_A + lambda * H_B at K values of lambda at once.

The K vectors are stored interleaved (element i of all K vectors next to each other), so every random read of
the matrix-vector product serves all K problems.  K Lanczos iterations run in lock step; each column keeps its own
alpha, beta and convergence test.  Matrix-free kernel only; Sz sector, optional translations at momentum zero.
"""
import time, ctypes
import numpy as np
import edc
from edc import ptr, PD, PI64, PU32


class FamilyGen(edc.Gen):
    def __init__(self, comps, nup, group, K, schedule='dynamic, 4096'):
        super().__init__(comps[0], nup, group, K, schedule)
        self.comps = comps

    def source(self):
        L, LA, K = self.L, self.LA, self.K
        sym = self.group is not None
        low = (1 << LA)-1
        nc = len(self.comps)
        o = ['#include <stdint.h>', '#include <math.h>', '#include <omp.h>', f'typedef {self.T} st_t;',
             'static const uint32_t *JA, *JB, *TAB; static const st_t *ST; static const double *SQ, *ISQ;',
             'static double *XS; static int64_t NROW; static const double *LAM;',
             'void edc_bind(const uint32_t* ja, const uint32_t* jb, const uint32_t* tab, const st_t* st, const double* sq,',
             '  const double* isq, double* xs, int64_t nrow, const double* lam){ JA=ja; JB=jb; TAB=tab; ST=st; SQ=sq; ISQ=isq; XS=xs; NROW=nrow; LAM=lam; }',
             'void edc_threads(int n){ omp_set_num_threads(n); }',
             f'#define FULLIDX(t) ((int64_t)JA[(t) >> {LA}] + JB[(t) & {self.c(low)}])',
             f'#define IDX(t) ({"(int64_t)TAB[FULLIDX(t)]" if sym else "FULLIDX(t)"})']
        body = []
        for ci, comp in enumerate(self.comps):
            fams = edc.families(comp)
            dconst = 0.0
            body.append(f'double d{ci} = DC{ci}; double a{ci}[{K}]; for (int q = 0; q < {K}; q++) a{ci}[q] = 0.0;')
            for fi, f in enumerate(fams):
                flip = [0]*L; cxy = [0.0]*L; wz = [0.0]*L
                for a, (b, jxy, jz) in f.items():
                    flip[a] = (1 << a) | (1 << b); cxy[a] = jxy/2; wz[a] = jz/4; dconst += jz/4
                tag = f'{ci}_{fi}'
                o.append(f'static const st_t FLIP{tag}[{L}] = {{{", ".join(self.c(v) for v in flip)}}};')
                o.append(f'static const double CXY{tag}[{L}] = {{{", ".join(repr(float(v)) for v in cxy)}}};')
                o.append(f'static const double WZ{tag}[{L}] = {{{", ".join(repr(float(v)) for v in wz)}}};')
                mxy = sum(1 << a for a, (b, jxy, jz) in f.items() if jxy != 0)
                mz = sum(1 << a for a, (b, jxy, jz) in f.items() if jz != 0)
                uz = {wz[a] for a in f if wz[a] != 0}
                uxy = {cxy[a] for a in f if cxy[a] != 0}
                body.append(f'{{ st_t u = {self.u_expr(f)};')
                if mz:
                    if len(uz) == 1:
                        body.append(f'd{ci} -= {2*next(iter(uz))!r} * (double){self.POP}(u & {self.c(mz)});')
                    else:
                        body.append(f'{{ st_t z = u & {self.c(mz)}; while (z) {{ int a = {self.CTZ}(z); z &= z - 1; d{ci} -= 2.0 * WZ{tag}[a]; }} }}')
                if mxy:
                    cexpr = repr(next(iter(uxy))) if len(uxy) == 1 else f'CXY{tag}[a]'
                    body.append(f'{{ st_t fl = u & {self.c(mxy)}; while (fl) {{ int a = {self.CTZ}(fl); fl &= fl - 1; '
                                f'st_t t = s ^ FLIP{tag}[a]; int64_t j = IDX(t); double c = {cexpr}; '
                                f'for (int q = 0; q < {K}; q++) a{ci}[q] += c * X[j*{K}+q]; }} }}')
                body.append('}')
            hfield = comp.h
            dconst = float(dconst) - float(hfield.sum())/2
            body += [f'd{ci} += {float(hfield[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if hfield[i] != 0]
            o.append(f'#define DC{ci} {dconst!r}')
        terms_d = ' + '.join(['d0']+[f'LAM[{ci}*{K}+q] * d{ci}' for ci in range(1, nc)])
        terms_a = ' + '.join(['a0[q]']+[f'LAM[{ci}*{K}+q] * a{ci}[q]' for ci in range(1, nc)])
        scale = 'SQ[i] * ' if sym else ''
        o.append('void mv_free(const double* restrict x, double* restrict y){')
        if sym:
            o.append(f'  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW*{K}; i++) XS[i] = x[i] * ISQ[i/{K}];')
            o.append('  const double* restrict X = XS;')
        else:
            o.append('  const double* restrict X = x;')
        o.append(f'  #pragma omp parallel for schedule({self.schedule})\n  for (int64_t i = 0; i < NROW; i++) {{ st_t s = ST[i];')
        o += ['    '+b for b in body]
        o.append(f'    for (int q = 0; q < {K}; q++) y[i*{K}+q] = ({terms_d}) * x[i*{K}+q] + {scale}({terms_a}); }}\n}}')
        # batched Lanczos step: every column q has its own alpha_q, beta_q
        o.append(f'void lanczos_step(const double* restrict v, const double* restrict vp, double* restrict w, const double* bprev, double* al, double* be){{')
        o.append('  mv_free(v, w);')
        o.append(f'  double A[{K}] = {{0}}, B[{K}] = {{0}};')
        o.append(f'  #pragma omp parallel for schedule(static) reduction(+:A[:{K}])\n  for (int64_t i = 0; i < NROW; i++) for (int q = 0; q < {K}; q++) A[q] += v[i*{K}+q] * w[i*{K}+q];')
        o.append(f'  #pragma omp parallel for schedule(static) reduction(+:B[:{K}])\n  for (int64_t i = 0; i < NROW; i++) for (int q = 0; q < {K}; q++) {{ double t = w[i*{K}+q] - A[q] * v[i*{K}+q] - bprev[q] * vp[i*{K}+q]; w[i*{K}+q] = t; B[q] += t * t; }}')
        o.append(f'  double ib[{K}]; for (int q = 0; q < {K}; q++) {{ al[q] = A[q]; be[q] = sqrt(B[q]); ib[q] = be[q] > 0 ? 1.0 / be[q] : 0.0; }}')
        o.append(f'  #pragma omp parallel for schedule(static)\n  for (int64_t i = 0; i < NROW; i++) for (int q = 0; q < {K}; q++) w[i*{K}+q] *= ib[q];\n}}')
        return '\n'.join(o)+'\n'


class FamilyEngine:
    """K problems H_A + lam_q H_B (q = 0..K-1) solved together."""

    def __init__(self, comp_a, comp_b, lams, nup, group=None, threads=None):
        t0 = time.perf_counter()
        self.K = K = len(lams)
        self.base = edc.Engine(comp_a, nup, group=group, K=1, threads=threads)   # basis, tables, reps
        b = self.base
        self.gen = FamilyGen([comp_a, comp_b], nup, group, K)
        self.lib, self.compile_seconds, _ = edc.compile_source(self.gen.source(), tag='fam')
        if threads:
            self.lib.edc_threads(threads)
        self.lam = np.zeros(2*K); self.lam[K:] = lams
        self.dim = b.dim
        self.xs = np.empty(b.dim*K) if b.sym else None
        self.lib.edc_bind.argtypes = [PU32, PU32, PU32, ctypes.POINTER(b.st_c), PD, PD, PD, ctypes.c_int64, PD]
        self.lib.edc_bind(ptr(b.ja, ctypes.c_uint32), ptr(b.jb, ctypes.c_uint32), ptr(b.tab, ctypes.c_uint32),
                          ptr(b.st, b.st_c), ptr(b.sq, ctypes.c_double) if b.sym else None,
                          ptr(b.isq, ctypes.c_double) if b.sym else None, ptr(self.xs, ctypes.c_double) if b.sym else None,
                          b.dim, ptr(self.lam, ctypes.c_double))
        self.lib.mv_free.argtypes = [PD, PD]
        self.lib.lanczos_step.argtypes = [PD, PD, PD, PD, PD, PD]
        self.setup_seconds = time.perf_counter()-t0

    def threads(self, n):
        self.lib.edc_threads(n)

    def ground_energies(self, tol=1e-12, maxit=1000, seed=42):
        from scipy.linalg import eigvalsh_tridiagonal
        K, n = self.K, self.dim
        v = np.random.default_rng(seed).standard_normal((n, K)); v /= np.linalg.norm(v, axis=0)
        v = np.ascontiguousarray(v).ravel(); vp = np.zeros(n*K); w = np.empty(n*K)
        bprev = np.zeros(K); al = np.zeros(K); be = np.zeros(K)
        A, B = [], []
        E = np.full(K, np.nan); done = np.zeros(K, bool); its = np.zeros(K, int)
        t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.lib.lanczos_step(*(ptr(a, ctypes.c_double) for a in (v, vp, w, bprev, al, be)))
            A.append(al.copy()); B.append(be.copy())
            if it > 1:
                Aa, Bb = np.array(A), np.array(B)
                for q in np.flatnonzero(~done):
                    e = eigvalsh_tridiagonal(Aa[:, q], Bb[:-1, q], select='i', select_range=(0, 0))[0]
                    if abs(e-E[q]) < tol*max(1.0, abs(e)):
                        done[q] = True; its[q] = it
                    E[q] = e
            if done.all():
                break
            bprev = be.copy()
            vp, v, w = v, w, vp
        return E, its, time.perf_counter()-t0
