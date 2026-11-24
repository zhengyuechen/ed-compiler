#!/usr/bin/env python3
"""GPU back end of the layer DMRG of edc_dmrg.py (one GPU, double precision).

Version 0 (the baseline): the lowest state of the effective Hamiltonian of a layer is found on the GPU -- the Lanczos
vectors live there and every product with the effective Hamiltonian runs there -- while the moves between layers
(singular value decompositions, sketched expansion, growth of the environments) stay on the host in the compiled
library of edc_dmrg.py.  For every layer solve the environments of that layer and the offset tables of its plan are
copied to the device.

The product is the stacked form of edc_dmrg.py (theta in two layouts, M per left label and U per right label):
    left half    out_i = LH_i M_i + GL_i S^L_i       S^L_i: column slabs of M moved by the operators Y_t of the left bond
    right half   oU_j  = U_j RH_j + S^R_j GR_j       S^R_j: row slabs of U moved by the operators X_t of the right bond
    layer        oU   += (Hamiltonian inside the layer) U
The slab moves are CUDA kernels with one thread per output number: the thread finds its label, term, layer state and
position by binary search in the offset tables of the plan and gathers its few sources.  The matrix products are one
cuBLAS call per label.

Couplings two layers apart (J2 on kagome) are not in this back end yet."""
import time, math, ctypes
import numpy as np
import cupy as cp
import edc_dmrg as ed
from edc_dmrg_factor import tall_qr

_SRC = r'''
typedef long long i64;
__device__ inline int upper(const i64* a, int n, i64 x) { int lo = 0, hi = n; while (hi - lo > 1) { int m = (lo + hi) >> 1; if (a[m] <= x) lo = m; else hi = m; } return lo; }
__device__ inline int upper32(const int* a, int n, int x) { int lo = 0, hi = n; while (hi - lo > 1) { int m = (lo + hi) >> 1; if (a[m] <= x) lo = m; else hi = m; } return lo; }

/* S^L of every left label: block i is (WL_i x C_i); row = (term t, state a of the source label), column = (layer state, b) */
extern "C" __global__ void sl_build(const double* M, double* SL, i64 total, int nL, int S, int NT, const i64* SLoff, const int* C, const int* DL,
    const int* srcL, const int* wstL, const int* cst, const int* cum, const i64* Moff, const int* ptr, const int* ein, const double* ev) {
  i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;
  int i = upper(SLoff, nL, x); i64 loc = x - SLoff[i]; int Ci = C[i]; int r = (int)(loc / Ci), c = (int)(loc % Ci);
  int t = -1, a = 0, is = -1;
  for (int tt = 0; tt < NT; tt++) { int s_ = srcL[i*NT+tt]; if (s_ < 0) continue; int w0 = wstL[i*NT+tt];
    if (r >= w0 && r < w0 + DL[s_]) { t = tt; a = r - w0; is = s_; break; } }
  if (t < 0) { SL[x] = 0.0; return; }
  const int* cu = cum + (i64)i*S; int sp = upper32(cu, S, c); int b = c - cu[sp];
  const double* Ms = M + Moff[is] + (i64)a*C[is]; double acc = 0.0;
  for (int e = ptr[t*S+sp]; e < ptr[t*S+sp+1]; e++) { int cs = cst[is*S + ein[e]]; if (cs >= 0) acc += ev[e]*Ms[cs + b]; }
  SL[x] = acc;
}
/* S^R of every right label: block j is (R_j x WR_j); row = (layer state, a), column = (term t, state b of the source label) */
extern "C" __global__ void sr_build(const double* U, double* SR, i64 total, int nR, int S, int NT, const i64* SRoff, const int* WR, const int* DR,
    const int* srcR, const int* wstR, const int* rst, const int* cum, const i64* Uoff, const int* ptr, const int* ein, const double* ev) {
  i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;
  int j = upper(SRoff, nR, x); i64 loc = x - SRoff[j]; int W = WR[j]; int r = (int)(loc / W), w = (int)(loc % W);
  int t = -1, b = 0, js = -1;
  for (int tt = 0; tt < NT; tt++) { int s_ = srcR[j*NT+tt]; if (s_ < 0) continue; int w0 = wstR[j*NT+tt];
    if (w >= w0 && w < w0 + DR[s_]) { t = tt; b = w - w0; js = s_; break; } }
  if (t < 0) { SR[x] = 0.0; return; }
  const int* cu = cum + (i64)j*S; int sp = upper32(cu, S, r); int a = r - cu[sp];
  const double* Us = U + Uoff[js]; int Djs = DR[js]; double acc = 0.0;
  for (int e = ptr[t*S+sp]; e < ptr[t*S+sp+1]; e++) { int rs = rst[js*S + ein[e]]; if (rs >= 0) acc += ev[e]*Us[(i64)(rs + a)*Djs + b]; }
  SR[x] = acc;
}
/* oU += (Hamiltonian inside the layer) U, U layout: block j is (R_j x D_j) */
extern "C" __global__ void hr_add(const double* U, double* oU, i64 total, int nR, int S, const i64* Uoff, const int* DR, const int* rst, const int* cum,
    const int* ptr, const int* ein, const double* ev) {
  i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;
  int j = upper(Uoff, nR, x); i64 loc = x - Uoff[j]; int Dj = DR[j]; int r = (int)(loc / Dj), b = (int)(loc % Dj);
  const int* cu = cum + (i64)j*S; int sp = upper32(cu, S, r); int a = r - cu[sp];
  const double* Us = U + Uoff[j]; double acc = 0.0;
  for (int e = ptr[sp]; e < ptr[sp+1]; e++) { int rs = rst[j*S + ein[e]]; if (rs >= 0) acc += ev[e]*Us[(i64)(rs + a)*Dj + b]; }
  oU[x] += acc;
}
/* U layout from M layout */
extern "C" __global__ void m2u(const double* M, double* U, i64 total, int nR, int S, const i64* Uoff, const int* DR, const int* cumr, const int* ul,
    const i64* Moff, const int* C, const int* cst) {
  i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;
  int j = upper(Uoff, nR, x); i64 loc = x - Uoff[j]; int Dj = DR[j]; int r = (int)(loc / Dj), b = (int)(loc % Dj);
  const int* cu = cumr + (i64)j*S; int s = upper32(cu, S, r); int a = r - cu[s]; int i = ul[j*S+s];
  U[x] = M[Moff[i] + (i64)a*C[i] + cst[i*S+s] + b];
}
/* out (M layout) += oU (U layout) */
extern "C" __global__ void u2m_add(const double* oU, double* out, i64 total, int nL, int S, const i64* Moff, const int* C, const int* cumc, const int* mr,
    const i64* Uoff, const int* DR, const int* rst) {
  i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;
  int i = upper(Moff, nL, x); i64 loc = x - Moff[i]; int Ci = C[i]; int a = (int)(loc / Ci), c = (int)(loc % Ci);
  const int* cu = cumc + (i64)i*S; int s = upper32(cu, S, c); int b = c - cu[s]; int j = mr[i*S+s];
  out[x] += oU[Uoff[j] + (i64)(rst[j*S+s] + a)*DR[j] + b];
}
'''
_mod = None
BLOCK = 256


def _kernel(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=('-std=c++14',))
    return _mod.get_function(name)


def _launch(name, total, args):
    if total > 0:
        _kernel(name)(((int(total) + BLOCK - 1)//BLOCK,), (BLOCK,), args)


def _csr(ops, S):
    """Terms given as (out, in, value) lists -> one table sorted by (term, out state): ptr[t*S + s], in state, value."""
    ptr = np.zeros(len(ops)*S + 1, dtype=np.int32); ins, vals = [], []
    for t, (o, i, v) in enumerate(ops):
        o = np.asarray(o, dtype=np.int64); order = np.argsort(o, kind='stable'); cnt = np.bincount(o, minlength=S)
        ptr[t*S+1:(t+1)*S+1] = ptr[t*S] + np.cumsum(cnt); ins.append(np.asarray(i, dtype=np.int32)[order]); vals.append(np.asarray(v, dtype=np.float64)[order])
    return cp.asarray(ptr), cp.asarray(np.concatenate(ins) if ins else np.zeros(1, np.int32)), cp.asarray(np.concatenate(vals) if vals else np.zeros(1))


class Tables:
    """What does not depend on the bond dimensions: operators of the layers and of the stacks, on the device."""
    def __init__(self, model):
        self.H = [_csr([kd.H], kd.S) for kd in model.kinds]; lkeys = {}; rkeys = {}
        for r in range(model.n):
            if model.site_ls[r] >= 0:
                lkeys[model.site_ls[r]] = model.kinds[model.site_kind[r]].S
            if model.site_rs[r] >= 0:
                rkeys[model.site_rs[r]] = model.kinds[model.site_kind[r]].S
        self.L = {k: _csr(model.lstacks[k].ops, S) for k, S in lkeys.items()}
        self.R = {k: _csr(model.rstacks[k].ops, S) for k, S in rkeys.items()}


class Product:
    """The effective Hamiltonian of one layer on the device, for the bond dimensions of its plan."""
    def __init__(self, eng, r, prof):
        p = self.p = eng.plan[r]; md = eng.model; tb = eng.tables; self.prof = prof; assert p.c.NTH == 0, 'couplings two layers apart: not in the GPU back end yet'
        t0 = time.perf_counter()
        self.n = p.size; self.nL, self.nR, self.S = p.nL, p.nR, p.c.S; self.NTL, self.NTR = max(1, p.c.NTL), max(1, p.c.NTR)
        g = lambda a, dt: cp.asarray(np.ascontiguousarray(a, dtype=dt))
        self.DL, self.DR, self.C, self.R = (g(a, np.int32) for a in (p.DL, p.DR, p.C, p.R))
        self.cst, self.rst, self.ul, self.mr = (g(a, np.int32) for a in (p.cst, p.rst, p.ul, p.mr))
        self.cumc = g(np.cumsum(p.cw, axis=1) - p.cw, np.int32); self.cumr = g(np.cumsum(p.rh, axis=1) - p.rh, np.int32)
        self.Moff, self.Uoff = g(p.Moff, np.int64), g(p.Uoff, np.int64)
        self.srcL, self.wstL, self.srcR, self.wstR = (g(a, np.int32) for a in (p.srcL, p.wstL, p.srcR, p.wstR)); self.WRg = g(p.WR, np.int32)
        self.SLoff_h = ed._off(p.WL.astype(np.int64)*p.C); self.SRoff_h = ed._off(p.R.astype(np.int64)*p.WR)
        self.SLoff, self.SRoff = g(self.SLoff_h, np.int64), g(self.SRoff_h, np.int64)
        self.hasL = md.site_ls[r] >= 0 and r > 0; self.hasR = md.site_rs[r] >= 0 and r < md.n-1
        self.tL = tb.L[md.site_ls[r]] if self.hasL else None; self.tR = tb.R[md.site_rs[r]] if self.hasR else None; self.tH = tb.H[md.site_kind[r]]
        self.LH, self.GL, self.RH, self.GR = (cp.asarray(a) for a in (eng.LH[r], eng.GL[r], eng.RH[r+1], eng.GR[r+1]))
        n = self.n; self.Mb = cp.empty(n); self.Ob = cp.empty(n); self.U = cp.empty(n); self.oU = cp.empty(n)
        self.SL = cp.empty(max(1, int(self.SLoff_h[-1]))); self.SR = cp.empty(max(1, int(self.SRoff_h[-1])))
        # views, made once: (LH_i, M_i, out_i, GL_i, SL_i) per left label, (U_j, RH_j, oU_j, SR_j, GR_j) per right label
        self.left = []; self.right = []
        for i in range(p.nL):
            Di, Ci, W = int(p.DL[i]), int(p.C[i]), int(p.WL[i])
            if Di and Ci:
                a, b = int(p.Moff[i]), int(p.Moff[i+1]); st = W and self.hasL
                self.left.append((self.LH[int(p.LHoff[i]):int(p.LHoff[i+1])].reshape(Di, Di) if r > 0 else None, self.Mb[a:b].reshape(Di, Ci), self.Ob[a:b].reshape(Di, Ci),
                                  self.GL[int(p.GLoff[i]):int(p.GLoff[i+1])].reshape(Di, W) if st else None,
                                  self.SL[int(self.SLoff_h[i]):int(self.SLoff_h[i+1])].reshape(W, Ci) if st else None))
        for j in range(p.nR):
            Dj, Rj, W = int(p.DR[j]), int(p.R[j]), int(p.WR[j])
            if Dj and Rj:
                a, b = int(p.Uoff[j]), int(p.Uoff[j+1]); st = W and self.hasR
                self.right.append((self.U[a:b].reshape(Rj, Dj), self.RH[int(p.RHoff[j]):int(p.RHoff[j+1])].reshape(Dj, Dj) if r < md.n-1 else None, self.oU[a:b].reshape(Rj, Dj),
                                   self.SR[int(self.SRoff_h[j]):int(self.SRoff_h[j+1])].reshape(Rj, W) if st else None,
                                   self.GR[int(p.GRoff[j]):int(p.GRoff[j+1])].reshape(W, Dj) if st else None))
        self._t('upload', t0)

    def _t(self, key, t0):
        cp.cuda.Device().synchronize(); t1 = time.perf_counter(); self.prof[key] = self.prof.get(key, 0.0) + t1 - t0
        return t1

    def to_u(self, x):
        """U layout (device) of a tensor given in the M layout (device)."""
        self.Mb[...] = x; n = self.n
        _launch('m2u', n, (self.Mb, self.U, np.int64(n), np.int32(self.nR), np.int32(self.S), self.Uoff, self.DR, self.cumr, self.ul, self.Moff, self.C, self.cst))
        return self.U

    def apply(self, v):
        i32, i64 = np.int32, np.int64; n = self.n; t = time.perf_counter(); self.Mb[...] = v; self.Ob.fill(0.0); self.oU.fill(0.0); t = self._t('copy', t)
        if self.hasL:
            ptr, ein, ev = self.tL
            _launch('sl_build', self.SL.size, (self.Mb, self.SL, i64(self.SL.size), i32(self.nL), i32(self.S), i32(self.NTL), self.SLoff, self.C, self.DL,
                                               self.srcL, self.wstL, self.cst, self.cumc, self.Moff, ptr, ein, ev))
        t = self._t('slab', t)
        for LH, M, O, GL, SL in self.left:
            if LH is not None:
                cp.matmul(LH, M, out=O)
            if GL is not None:
                O += GL @ SL
        t = self._t('gemm', t)
        _launch('m2u', n, (self.Mb, self.U, i64(n), i32(self.nR), i32(self.S), self.Uoff, self.DR, self.cumr, self.ul, self.Moff, self.C, self.cst))
        t = self._t('layout', t)
        if self.hasR:
            ptr, ein, ev = self.tR
            _launch('sr_build', self.SR.size, (self.U, self.SR, i64(self.SR.size), i32(self.nR), i32(self.S), i32(self.NTR), self.SRoff, self.WRg, self.DR,
                                               self.srcR, self.wstR, self.rst, self.cumr, self.Uoff, ptr, ein, ev))
        t = self._t('slab', t)
        for U, RH, oU, SR, GR in self.right:
            if RH is not None:
                cp.matmul(U, RH, out=oU)
            if GR is not None:
                oU += SR @ GR
        t = self._t('gemm', t)
        ptr, ein, ev = self.tH
        _launch('hr_add', n, (self.U, self.oU, i64(n), i32(self.nR), i32(self.S), self.Uoff, self.DR, self.rst, self.cumr, ptr, ein, ev))
        t = self._t('slab', t)
        _launch('u2m_add', n, (self.oU, self.Ob, i64(n), i32(self.nL), i32(self.S), self.Moff, self.C, self.cumc, self.mr, self.Uoff, self.DR, self.rst))
        t = self._t('layout', t)
        out = self.Ob.copy(); self._t('copy', t)
        return out


class DMRGGPU(ed.DMRG):
    """edc_dmrg.DMRG with the layer solves on the GPU (version 0: everything else on the host)."""
    version = 'gpu-v0'

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.tables = Tables(self.model); self.prof = {}; self.peak = 0; self.host = False; self.keep_dev = False; self._dev = None

    def solve(self, r, tol=1e-6, maxit=40):
        if self.host:
            return super().solve(r, tol, maxit)
        t00 = time.perf_counter(); pr = self.prof; H = Product(self, r, pr); n = H.n; t0 = time.perf_counter()
        v = cp.asarray(self.M[r]); v /= cp.linalg.norm(v)
        if self.noise > 0:
            xi = cp.asarray(self.rng.standard_normal(n)); v = v + (self.noise/float(cp.linalg.norm(xi)))*xi; v /= cp.linalg.norm(v)
        m = min(maxit, n); V = cp.empty((m, n)); V[0] = v; al = []; be = []; E = None; y = np.array([1.0]); k = 1
        t0 = H._t('lanczos', t0)
        for it in range(m):
            w = H.apply(V[it]); self.stats['products'] += 1; t0 = time.perf_counter()
            a = float(V[it] @ w); al.append(a); w -= a*V[it]
            if be:
                w -= be[-1]*V[it-1]
            w -= V[:k].T @ (V[:k] @ w); b = float(cp.linalg.norm(w))
            ev, Sv = np.linalg.eigh(np.diag(al) + np.diag(be, 1) + np.diag(be, -1)); E = float(ev[0]); y = Sv[:, 0]
            stop = b*abs(y[-1]) < tol*max(1.0, abs(E)) or b < 1e-13 or k == n or k == m
            if not stop:
                be.append(b); V[k] = w/b; k += 1
            t0 = H._t('lanczos', t0)
            if stop:
                break
        x = cp.asarray(y[:k]) @ V[:k]; x /= cp.linalg.norm(x); self.M[r] = cp.asnumpy(x); H._t('download', t0)
        self.peak = max(self.peak, cp.get_default_memory_pool().total_bytes())
        self._dev = (r, H, x) if self.keep_dev else None
        del V, v
        self.stats['solves'] += 1; self.stats['t_products'] += time.perf_counter()-t00
        return E


class DMRGGPU1(DMRGGPU):
    """Version 1 = version 0 with ONE change: the factorisation of the tall blocks in the moves.  Version 0 hands every
    block (R_j x D_j, R_j up to 2^(layer spins) times D_j) to LAPACK's SVD on the host.  Here the solved tensor stays on
    the device, each block is factored there with matrix products only (edc_dmrg_factor.tall_qr: Gram matrix and two
    Cholesky passes) and the host decomposes only the small square factor.  Blocks that are not tall or not of full
    rank to working precision take the old path.  The sketch of the expansion, the plans and the growth of the
    environments are unchanged (host)."""
    version = 'gpu-v1a'                                                # first attempt: every block through the device (slower than version 0)
    min_cols = 0; min_ratio = 1                                       # a block goes to the device if it has >= min_cols columns and >= min_ratio times as many rows

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.keep_dev = True; self.blocks_dev = 0; self.blocks_host = 0

    def _host_svd(self, A):
        t = time.perf_counter(); out = ('host',) + tuple(np.linalg.svd(A, full_matrices=False)); self.blocks_host += 1
        self.prof['mv_host_svd'] = self.prof.get('mv_host_svd', 0.0) + time.perf_counter() - t
        return out

    def _acc(self, key, t0):
        cp.cuda.Device().synchronize(); t1 = time.perf_counter(); self.prof[key] = self.prof.get(key, 0.0) + t1 - t0
        return t1

    def _factor(self, A):
        """A (device, tall): ('dev', Q, Uc, s, Vh) with A = (Q Uc) s Vh, Q on the device and the rest small on the host;
        or ('host', Us, s, Vh) from an ordinary SVD."""
        t = time.perf_counter(); out = tall_qr(A, cp, cp.asnumpy, cp.asarray) if A.shape[1] else None
        if out is None:
            self.blocks_host += 1; Us, s, Vh = np.linalg.svd(cp.asnumpy(A), full_matrices=False); self._acc('mv_fallback', t)
            return ('host', Us, s, Vh)
        t = self._acc('mv_gram_cholesky', t); Uc, s, Vh = np.linalg.svd(out[1]); self.blocks_dev += 1; self._acc('mv_small_svd', t)
        return ('dev', out[0], Uc, s, Vh)

    def move_right(self, r, Dmax, alpha, cutoff=1e-12):
        dev = self._dev
        if self.host or dev is None or dev[0] != r:
            return super().move_right(r, Dmax, alpha, cutoff)
        _, H, x = dev; self._dev = None; _p = ed._p
        p = self.plan[r]; lib = self.lib; md = self.model; t0 = time.perf_counter()
        kv = self._sketch(p.DR, p.WR, p.R-p.DR, alpha, int(p.DR.sum()) < Dmax)
        eoff = ed._off(p.R.astype(np.int64)*kv); ooff = ed._off(p.WR.astype(np.int64)*kv)
        Om = self.rng.standard_normal(max(1, int(ooff[-1]))); E = np.zeros(max(1, int(eoff[-1])))
        if kv.any() or self.min_cols:
            U = np.empty(p.size); lib.m2u(p.ref, _p(self.M[r]), _p(U))
        if kv.any():
            lib.sr_apply(p.ref, _p(Om), _p(ooff), _p(kv), _p(U), _p(E), _p(eoff))
        t = self._acc('mv_sketch', t0); Ud = H.to_u(x); res = {}
        for j in np.nonzero((p.R > 0) & (p.DR+kv > 0))[0]:
            Rj, Dj = int(p.R[j]), int(p.DR[j]); n = Dj + int(kv[j]); B = None
            if kv[j]:
                B = E[eoff[j]:eoff[j+1]].reshape(p.R[j], kv[j])/math.sqrt(kv[j]) + (self.random/math.sqrt(p.R[j]))*self.rng.standard_normal((p.R[j], kv[j]))
            if n >= self.min_cols and Rj >= self.min_ratio*n:
                A = Ud[int(p.Uoff[j]):int(p.Uoff[j+1])].reshape(Rj, Dj)
                res[j] = self._factor(A if B is None else cp.concatenate([A, cp.asarray(alpha*B)], axis=1))
            else:
                A = U[p.Uoff[j]:p.Uoff[j+1]].reshape(Rj, Dj); res[j] = self._host_svd(A if B is None else np.concatenate([A, alpha*B], axis=1))
        keep = self._keep({j: v[-2] for j, v in res.items()}, Dmax, cutoff)
        pn = self.plan[r+1]; Mn = self.M[r+1]; old = p.DR.copy(); new = np.zeros_like(old)
        for j, k in keep.items():
            new[j] = k
        self.bond[r+1].dims[:] = new; p2 = ed.Plan(md, r, self.bond[r], self.bond[r+1]); pn2 = ed.Plan(md, r+1, self.bond[r+1], self.bond[r+2])
        t = time.perf_counter(); U2d = cp.zeros(p2.size); Mn2 = np.zeros(pn2.size); kept = 0.0
        for j, v in res.items():
            k = int(new[j]); s, Vh = v[-2], v[-1]
            if not k:
                continue
            if v[0] == 'dev':
                U2d[int(p2.Uoff[j]):int(p2.Uoff[j+1])] = (v[1] @ cp.asarray(np.ascontiguousarray(v[2][:, :k]))).ravel()
            else:
                U2d[int(p2.Uoff[j]):int(p2.Uoff[j+1])] = cp.asarray(np.ascontiguousarray(v[1][:, :k]).ravel())
            if old[j] and pn.C[j]:
                Cm = s[:k, None]*Vh[:k, :old[j]]; kept += float(np.sum(Cm*Cm))
                Mn2[pn2.Moff[j]:pn2.Moff[j+1]] = (Cm @ Mn[pn.Moff[j]:pn.Moff[j+1]].reshape(old[j], pn.C[j])).reshape(-1)
        U2 = cp.asnumpy(U2d); self._acc('mv_assemble', t); del res, U2d, H, x, Ud
        self.stats['truncation'] += max(0.0, 1.0-kept); self.stats['t_svd'] += time.perf_counter()-t0
        M2 = np.zeros(p2.size); lib.u2m(p2.ref, _p(U2), _p(M2), 0)
        self.plan[r] = p2; self.M[r] = M2; self.plan[r+1] = pn2; self.M[r+1] = Mn2/np.linalg.norm(Mn2)
        t0 = time.perf_counter(); self._grow_left(r); self.stats['t_env'] += time.perf_counter()-t0

    def move_left(self, r, Dmax, alpha, cutoff=1e-12):
        dev = self._dev
        if self.host or dev is None or dev[0] != r:
            return super().move_left(r, Dmax, alpha, cutoff)
        _, H, x = dev; self._dev = None; _p = ed._p
        p = self.plan[r]; lib = self.lib; md = self.model; t0 = time.perf_counter(); M = self.M[r]
        kv = self._sketch(p.DL, p.WL, p.C-p.DL, alpha, int(p.DL.sum()) < Dmax)
        eoff = ed._off(kv.astype(np.int64)*p.C); ooff = ed._off(kv.astype(np.int64)*p.WL)
        Om = self.rng.standard_normal(max(1, int(ooff[-1]))); E = np.zeros(max(1, int(eoff[-1])))
        if kv.any():
            lib.sl_apply(p.ref, _p(Om), _p(ooff), _p(kv), _p(M), _p(E), _p(eoff))
        self._acc('mv_sketch', t0); res = {}
        for i in np.nonzero((p.C > 0) & (p.DL+kv > 0))[0]:
            Di, Ci = int(p.DL[i]), int(p.C[i]); n = Di + int(kv[i]); B = None
            if kv[i]:
                B = E[eoff[i]:eoff[i+1]].reshape(kv[i], p.C[i])/math.sqrt(kv[i]) + (self.random/math.sqrt(p.C[i]))*self.rng.standard_normal((kv[i], p.C[i]))
            if not (n >= self.min_cols and Ci >= self.min_ratio*n):
                A = M[p.Moff[i]:p.Moff[i+1]].reshape(Di, Ci); res[i] = self._host_svd(A if B is None else np.concatenate([A, alpha*B], axis=0)); continue
            A = x[int(p.Moff[i]):int(p.Moff[i+1])].reshape(Di, Ci)
            if B is not None:
                A = cp.concatenate([A, cp.asarray(alpha*B)], axis=0)
            v = self._factor(cp.ascontiguousarray(A.T))                # A^T = (Q Uc) s Vh  ->  A = Vh^T s (Q Uc)^T
            res[i] = ('dev', v[1], v[2], v[3], v[4].T) if v[0] == 'dev' else ('host', v[3].T, v[2], v[1].T)   # host: (Us, s, Vh) of A
        keep = self._keep({i: (v[3] if v[0] == 'dev' else v[2]) for i, v in res.items()}, Dmax, cutoff)
        pp = self.plan[r-1]; Up = np.empty(pp.size); lib.m2u(pp.ref, _p(self.M[r-1]), _p(Up)); old = p.DL.copy(); new = np.zeros_like(old)
        for i, k in keep.items():
            new[i] = k
        self.bond[r].dims[:] = new; p2 = ed.Plan(md, r, self.bond[r], self.bond[r+1]); pp2 = ed.Plan(md, r-1, self.bond[r-1], self.bond[r])
        t = time.perf_counter(); M2d = cp.zeros(p2.size); Up2 = np.zeros(pp2.size); kept = 0.0
        for i, v in res.items():
            k = int(new[i])
            if not k:
                continue
            if v[0] == 'dev':                                         # rows of the new tensor: (Q Uc[:, :k])^T ; Us of A = v[4]
                _, Q, Uc, s, Us = v; M2d[int(p2.Moff[i]):int(p2.Moff[i+1])] = cp.ascontiguousarray((Q @ cp.asarray(np.ascontiguousarray(Uc[:, :k]))).T).ravel()
            else:
                _, Us, s, Vh = v; M2d[int(p2.Moff[i]):int(p2.Moff[i+1])] = cp.asarray(np.ascontiguousarray(Vh[:k]).ravel())
            if old[i] and pp.R[i]:
                Cm = Us[:old[i], :k]*s[:k]; kept += float(np.sum(Cm*Cm))
                Up2[pp2.Uoff[i]:pp2.Uoff[i+1]] = (Up[pp.Uoff[i]:pp.Uoff[i+1]].reshape(pp.R[i], old[i]) @ Cm).reshape(-1)
        M2 = cp.asnumpy(M2d); self._acc('mv_assemble', t); del res, M2d, H, x
        self.stats['truncation'] += max(0.0, 1.0-kept); self.stats['t_svd'] += time.perf_counter()-t0
        Mp2 = np.zeros(pp2.size); lib.u2m(pp2.ref, _p(Up2), _p(Mp2), 0)
        self.plan[r] = p2; self.M[r] = M2; self.plan[r-1] = pp2; self.M[r-1] = Mp2/np.linalg.norm(Mp2)
        t0 = time.perf_counter(); self._grow_right(r); self.stats['t_env'] += time.perf_counter()-t0


class DMRGGPU1b(DMRGGPU1):
    """Version 1: as the first attempt, but only the heavy blocks go to the device (at least 48 columns and at least four
    times as many rows); the many small blocks keep the host SVD, which costs them 0.02 to 0.2 ms each."""
    version = 'gpu-v1'
    min_cols = 48; min_ratio = 4


def run(model, D, nup=None, K=0, cls=DMRGGPU, sweeps=None, tol=3e-5, seed=0, log=print):
    """The schedule of edc_dmrg.ground_state with the time, energy and profile of every sweep recorded."""
    nup = model.lat.N//2 if nup is None else nup; eng = cls(model, nup, K, seed=seed); alphas = list(ed.ALPHAS)
    alphas = alphas[:sweeps] if sweeps and sweeps <= len(alphas) else alphas + [0.0]*((sweeps or 0)-len(alphas))
    gpu = hasattr(eng, 'prof'); rows = []; t0 = time.perf_counter()
    sched = [(max(min(D, 64), D//4), 1e-2, tol, 1e-2), (max(min(D, 64), D//2), 1e-2, tol, 1e-2)] + [(D, al, tol if al > 0 else tol/30, 0.0) for al in alphas]
    for sw, (Dw, al, tl, noise) in enumerate(sched):
        ts = time.perf_counter(); p0 = eng.stats['products']; before = dict(eng.prof) if gpu else {}
        s0 = {k: eng.stats[k] for k in ('t_products', 't_svd', 't_env')}
        E = eng.sweep(Dw, al, tl, noise=noise); te = time.perf_counter()
        if sw < 2:
            eng.stats['energies'].pop()
        row = dict(sweep=sw+1, D=Dw, expansion=al, E=E, seconds=te-ts, elapsed=te-t0, products=eng.stats['products']-p0, largest_bond=max(eng.bond_sizes()),
                   solve=eng.stats['t_products']-s0['t_products'], svd=eng.stats['t_svd']-s0['t_svd'], env=eng.stats['t_env']-s0['t_env'])
        if gpu:
            row['gpu'] = {k: eng.prof[k]-before.get(k, 0.0) for k in eng.prof}; row['peak_pool_GiB'] = eng.peak/2**30
        rows.append(row)
        if log:
            extra = (' | device: ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(row['gpu'].items())) + f' | pool peak {row["peak_pool_GiB"]:.2f} GiB') if gpu else ''
            log(f'  sweep {sw+1:2d} D={Dw:5d} exp={al:.0e}: E = {E:.10f} | {row["seconds"]:7.2f} s (elapsed {row["elapsed"]:8.2f}) | products {row["products"]:5d} | '
                f'solve {row["solve"]:.2f}, svd {row["svd"]:.2f}, env {row["env"]:.2f}{extra}')
    return E, eng, rows
