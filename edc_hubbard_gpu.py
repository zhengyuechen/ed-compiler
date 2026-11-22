#!/usr/bin/env python3
"""Hubbard model on the GPU: one thread per (up, down) pair; T_up reads whole rows (coalesced), T_dn reads inside
the row (cache resident), U elementwise.  Same CSR hopping tables as edc_hubbard."""
import time
import numpy as np
import cupy as cp
import edc_hubbard as eh
from edc_gpu import _ddot

SRC = r'''
extern "C" __global__ void mv(const double* __restrict__ x, double* __restrict__ y, const long long* __restrict__ UP,
    const long long* __restrict__ DN, const long long* __restrict__ TUP, const unsigned int* __restrict__ TUI,
    const double* __restrict__ TUV, const long long* __restrict__ TDP, const unsigned int* __restrict__ TDI,
    const double* __restrict__ TDV, double U, long long DU, long long DD) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= DU*DD) return;
  long long iu = i / DD, id = i - iu*DD;
  double acc = U * (double)__popcll(UP[iu] & DN[id]) * x[i];
  for (long long p = TUP[iu]; p < TUP[iu+1]; p++) acc += TUV[p] * x[(long long)TUI[p]*DD + id];
  const double* xr = x + iu*DD;
  for (long long p = TDP[id]; p < TDP[id+1]; p++) acc += TDV[p] * xr[TDI[p]];
  y[i] = acc; }
'''
_update = cp.ElementwiseKernel('float64 v, float64 vp, float64 a, float64 b', 'float64 w', 'w = w - a * v - b * vp', 'hub_update')


class HubbardGPU:
    def __init__(self, model, nup, ndn, block=256):
        t0 = time.perf_counter()
        L = model.n
        up, dn = eh.basis(L, nup), eh.basis(L, ndn)
        tu, td = eh.hopping_csr(up, model.hops, L), eh.hopping_csr(dn, model.hops, L)
        self.cpu_seconds = time.perf_counter()-t0
        self.kern = cp.RawKernel(SRC, 'mv')
        self.args = [cp.asarray(a) for a in (up, dn, tu[0], tu[1], tu[2], td[0], td[1], td[2])]
        self.U, self.du, self.dd = float(model.U), len(up), len(dn)
        self.dim = self.du*self.dd
        self.block, self.grid = block, (self.dim+block-1)//block
        cp.cuda.Stream.null.synchronize()
        self.setup_seconds = time.perf_counter()-t0

    def matvec(self, x, y):
        self.kern((self.grid,), (self.block,), (x, y, *self.args, self.U, np.int64(self.du), np.int64(self.dd)))
        return y

    def time_matvec(self, reps=5):
        x = cp.random.default_rng(0).standard_normal(self.dim); y = cp.empty_like(x)
        self.matvec(x, y); cp.cuda.Stream.null.synchronize(); best = np.inf
        for _ in range(reps):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event(); e0.record(); self.matvec(x, y); e1.record(); e1.synchronize()
            best = min(best, cp.cuda.get_elapsed_time(e0, e1)/1e3)
        return best

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42):
        from scipy.linalg import eigvalsh_tridiagonal
        n = self.dim
        v = cp.asarray(np.random.default_rng(seed).standard_normal(n)); v /= np.sqrt(_ddot(v, v))
        vp = cp.zeros(n); w = cp.empty(n); A, B = [], []; bprev = 0.0; e_old = None
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.matvec(v, w); al = _ddot(v, w); _update(v, vp, al, bprev, w)
            be = float(np.sqrt(_ddot(w, w))); w *= 1.0/be; A.append(al); B.append(be)
            if it > 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0]
                if e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e)):
                    return e, it, time.perf_counter()-t0
                e_old = e
            bprev = be; vp, v, w = v, w, vp
        return e_old, it, time.perf_counter()-t0
