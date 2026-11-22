#!/usr/bin/env python3
"""edc_hubbard — Fermi-Hubbard model, fixed N_up and N_down, no spatial symmetry.

H = -sum_<ij>,sigma t_ij (c+_i,sigma c_j,sigma + h.c.) + U sum_i n_i,up n_i,down

Structure used: with all up operators ordered before all down operators (Jordan-Wigner), the up hopping does not
see the down electrons and vice versa, so H = T_up (x) 1 + 1 (x) T_dn + U D.  Store the vector as a matrix
X[up config][down config]:
  * T_dn acts inside each row (short rows, cache resident),
  * T_up adds whole rows to whole rows (streaming, vectorised),
  * U D is elementwise.
Two generated kernels: 'kron' (uses the structure) and 'flat' (one loop over all basis states, same data).
"""
import time, ctypes, math
import numpy as np
import edc
from edc import ptr, PD, PI64, PU32


class HubbardModel:
    def __init__(self, n):
        self.n = n; self.hops = []; self.U = 0.0

    def hop(self, i, j, t=1.0):
        self.hops.append((min(i, j), max(i, j), t))


def chain(L, t=1.0, U=8.0):
    m = HubbardModel(L)
    for i in range(L):
        m.hop(i, (i+1) % L, t)
    m.U = U
    return m


def square(Lx, Ly, t=1.0, U=8.0):
    m = HubbardModel(Lx*Ly)
    for y in range(Ly):
        for x in range(Lx):
            i = x+Lx*y
            m.hop(i, (x+1) % Lx+Lx*y, t); m.hop(i, x+Lx*((y+1) % Ly), t)
    m.U = U
    return m


def basis(L, n):
    s = np.array([sum(1 << k for k in c) for c in __import__('itertools').combinations(range(L), n)], np.int64)
    s.sort()
    return s


def hopping_csr(states, hops, L):
    """T[r, c] = -t * sign for every hop that connects configuration c to configuration r (one species)."""
    rows, cols, vals = [], [], []
    for (i, j, t) in hops:
        mask = (1 << i) | (1 << j)
        between = ((1 << j)-1) & ~((1 << (i+1))-1)
        bi, bj = (states >> i) & 1, (states >> j) & 1
        ok = np.flatnonzero(bi != bj)
        tgt = states[ok] ^ mask
        par = np.zeros(len(ok), np.int64)
        tmp = states[ok] & between
        while np.any(tmp):
            par ^= tmp & 1; tmp = tmp >> 1
        sign = 1.0-2.0*par
        c = np.searchsorted(states, tgt)
        rows.append(ok); cols.append(c); vals.append(-t*sign)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    ptr_ = np.zeros(len(states)+1, np.int64)
    np.add.at(ptr_, rows+1, 1); np.cumsum(ptr_, out=ptr_)
    return ptr_, cols.astype(np.uint32), vals


SRC = r'''
#include <stdint.h>
#include <math.h>
#include <omp.h>
static int64_t DU, DD; static double UU;
static const int64_t *UP, *DN, *TUP, *TDP; static const uint32_t *TUI, *TDI; static const double *TUV, *TDV;
void edc_threads(int n){ omp_set_num_threads(n); }
void bind(int64_t du, int64_t dd, double u, const int64_t* up, const int64_t* dn, const int64_t* tup, const uint32_t* tui,
          const double* tuv, const int64_t* tdp, const uint32_t* tdi, const double* tdv){
  DU=du; DD=dd; UU=u; UP=up; DN=dn; TUP=tup; TUI=tui; TUV=tuv; TDP=tdp; TDI=tdi; TDV=tdv; }
void mv_kron(const double* restrict x, double* restrict y){
  #pragma omp parallel for schedule(dynamic, 1)
  for (int64_t iu = 0; iu < DU; iu++) {
    const int64_t su = UP[iu]; double* restrict yr = y + iu*DD; const double* restrict xr = x + iu*DD;
    for (int64_t id = 0; id < DD; id++) {
      double acc = UU * (double)__builtin_popcountll(su & DN[id]) * xr[id];
      for (int64_t p = TDP[id]; p < TDP[id+1]; p++) acc += TDV[p] * xr[TDI[p]];
      yr[id] = acc; }
    for (int64_t p = TUP[iu]; p < TUP[iu+1]; p++) {
      const double c = TUV[p]; const double* restrict x2 = x + (int64_t)TUI[p]*DD;
      for (int64_t id = 0; id < DD; id++) yr[id] += c * x2[id]; }
  }
}
void mv_flat(const double* restrict x, double* restrict y){
  #pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t i = 0; i < DU*DD; i++) {
    const int64_t iu = i / DD, id = i - iu*DD;
    double acc = UU * (double)__builtin_popcountll(UP[iu] & DN[id]) * x[i];
    for (int64_t p = TUP[iu]; p < TUP[iu+1]; p++) acc += TUV[p] * x[(int64_t)TUI[p]*DD + id];
    for (int64_t p = TDP[id]; p < TDP[id+1]; p++) acc += TDV[p] * x[iu*DD + TDI[p]];
    y[i] = acc; }
}
void lanczos_step(int kind, const double* restrict v, const double* restrict vp, double* restrict w, double bprev, double* ab){
  const int64_t n = DU*DD;
  if (kind == 1) mv_kron(v, w); else mv_flat(v, w);
  double al = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:al)
  for (int64_t i = 0; i < n; i++) al += v[i] * w[i];
  double nr = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:nr)
  for (int64_t i = 0; i < n; i++) { double t = w[i] - al * v[i] - bprev * vp[i]; w[i] = t; nr += t * t; }
  double be = sqrt(nr), ib = 1.0 / be;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) w[i] *= ib;
  ab[0] = al; ab[1] = be; }
'''


class HubbardEngine:
    def __init__(self, model, nup, ndn, threads=None):
        t0 = time.perf_counter()
        L = model.n
        self.lib, self.compile_seconds, _ = edc.compile_source(SRC, tag='hub')
        if threads:
            self.lib.edc_threads(threads)
        self.up, self.dn = basis(L, nup), basis(L, ndn)
        self.tu = hopping_csr(self.up, model.hops, L)
        self.td = hopping_csr(self.dn, model.hops, L)
        self.du, self.dd = len(self.up), len(self.dn)
        self.dim = self.du*self.dd
        self.lib.bind.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_double, PI64, PI64, PI64, PU32, PD, PI64, PU32, PD]
        self.lib.bind(self.du, self.dd, model.U, ptr(self.up, ctypes.c_int64), ptr(self.dn, ctypes.c_int64),
                      ptr(self.tu[0], ctypes.c_int64), ptr(self.tu[1], ctypes.c_uint32), ptr(self.tu[2], ctypes.c_double),
                      ptr(self.td[0], ctypes.c_int64), ptr(self.td[1], ctypes.c_uint32), ptr(self.td[2], ctypes.c_double))
        for f in ('mv_kron', 'mv_flat'):
            getattr(self.lib, f).argtypes = [PD, PD]
        self.lib.lanczos_step.argtypes = [ctypes.c_int, PD, PD, PD, ctypes.c_double, PD]
        self.setup_seconds = time.perf_counter()-t0

    def threads(self, n):
        self.lib.edc_threads(n)

    def matvec(self, x, y, kernel='kron'):
        getattr(self.lib, 'mv_'+kernel)(ptr(x, ctypes.c_double), ptr(y, ctypes.c_double))
        return y

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, kernel='kron'):
        from scipy.linalg import eigvalsh_tridiagonal
        n = self.dim
        v = np.random.default_rng(seed).standard_normal(n); v /= np.linalg.norm(v)
        vp = np.zeros(n); w = np.empty(n); ab = np.zeros(2)
        A, B = [], []; bprev = 0.0; e_old = None
        t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.lib.lanczos_step(1 if kernel == 'kron' else 0, ptr(v, ctypes.c_double), ptr(vp, ctypes.c_double),
                                  ptr(w, ctypes.c_double), bprev, ptr(ab, ctypes.c_double))
            A.append(ab[0]); B.append(ab[1])
            if it > 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0]
                if e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e)):
                    return e, it, time.perf_counter()-t0
                e_old = e
            bprev = ab[1]
            vp, v, w = v, w, vp
        return e_old, it, time.perf_counter()-t0
