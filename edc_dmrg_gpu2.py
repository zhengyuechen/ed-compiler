#!/usr/bin/env python3
"""Layer DMRG GPU backends derived from edc_dmrg_gpu.DMRGGPU1b.

  DMRGGPU1t  version 1 with one host setting changed: BLAS on one thread everywhere except while an environment
             grows on the host, which runs with four (the cheap control: how much does the thread setting alone buy).
  DMRGGPU2   version 1 with one item moved: the environments grow on the device and stay there.  After a move the
             new left (right) environment of the next layer is
                 LH' = U^T [ (LH M + GL S^L) + H_layer U ]     GL' = U^T S^R_scaled            (grow to the right)
                 RH' = [ (U RH + S^R GR) + H_layer U ] M^T     GR' = S^L_scaled M^T             (grow to the left)
             label by label, with the slab kernels of edc_dmrg_gpu and one cuBLAS product per label; the layer solves
             then find their environments on the device (no upload per solve).  Moves, sketch and plans are as in
             version 1 (host, one BLAS thread).
Both: nearest-neighbour couplings only, like the rest of the GPU back end."""
import time
import numpy as np
import cupy as cp
import edc_dmrg as ed
import edc_dmrg_gpu as eg
from threadpoolctl import ThreadpoolController

_CTL = None


def _controller():
    global _CTL
    if _CTL is None:
        _CTL = ThreadpoolController()
    return _CTL


def blas_threads():
    return {f"{d['internal_api']}:{d['filepath'].split('/')[-1]}": d['num_threads'] for d in _controller().info() if d['user_api'] == 'blas'}


class DMRGGPU1t(eg.DMRGGPU1b):
    version = 'gpu-v1t'; env_threads = 4; base_threads = 1

    def __init__(self, *a, **k):
        _controller().limit(limits=self.base_threads, user_api='blas'); self._base_seen = blas_threads(); self.threads_seen = {}
        super().__init__(*a, **k)

    def _grow(self, f, r):
        with _controller().limit(limits=self.env_threads, user_api='blas'):
            if 'environments' not in self.threads_seen:
                self.threads_seen = dict(base=self._base_seen, environments=blas_threads())
            return f(r)

    def _grow_left(self, r):
        return self._grow(super()._grow_left, r)

    def _grow_right(self, r):
        return self._grow(super()._grow_right, r)


# ------------------------------------------------------------------ device growth of the environments
_SCALED = {}


def _scaled(model):
    """Stacks with their coefficients folded in (the host library's scale = 1 builds): {stack index: csr} for the
    left stacks and the right stacks."""
    if id(model) not in _SCALED:
        L, R = {}, {}
        for r in range(model.n):
            S = model.kinds[model.site_kind[r]].S
            for idx, stacks, out in ((model.site_ls[r], model.lstacks, L), (model.site_rs[r], model.rstacks, R)):
                if idx >= 0 and idx not in out:
                    st = stacks[idx]; out[idx] = eg._csr([(o, i, np.asarray(v)*f) for (o, i, v), f in zip(st.ops, st.scale)], S)
        _SCALED[id(model)] = (L, R)
    return _SCALED[id(model)]


class PlanDev:
    """The offset tables of one plan on the device and the slab kernels of edc_dmrg_gpu on them."""
    def __init__(self, p):
        g = lambda a, dt: cp.asarray(np.ascontiguousarray(a, dtype=dt))
        self.p = p; self.n = p.size; self.nL, self.nR, self.S = p.nL, p.nR, p.c.S; self.NTL, self.NTR = max(1, p.c.NTL), max(1, p.c.NTR)
        self.DL, self.DR, self.C = g(p.DL, np.int32), g(p.DR, np.int32), g(p.C, np.int32)
        self.cst, self.rst, self.ul, self.mr = (g(a, np.int32) for a in (p.cst, p.rst, p.ul, p.mr))
        self.cumc = g(np.cumsum(p.cw, axis=1) - p.cw, np.int32); self.cumr = g(np.cumsum(p.rh, axis=1) - p.rh, np.int32)
        self.Moff, self.Uoff = g(p.Moff, np.int64), g(p.Uoff, np.int64)
        self.srcL, self.wstL, self.srcR, self.wstR = (g(a, np.int32) for a in (p.srcL, p.wstL, p.srcR, p.wstR)); self.WRg = g(p.WR, np.int32)
        self.SLoff_h = ed._off(p.WL.astype(np.int64)*p.C); self.SRoff_h = ed._off(p.R.astype(np.int64)*p.WR)
        self.SLoff, self.SRoff = g(self.SLoff_h, np.int64), g(self.SRoff_h, np.int64)

    def m2u(self, M):
        U = cp.empty(self.n); i32, i64 = np.int32, np.int64
        eg._launch('m2u', self.n, (M, U, i64(self.n), i32(self.nR), i32(self.S), self.Uoff, self.DR, self.cumr, self.ul, self.Moff, self.C, self.cst))
        return U

    def u2m_add(self, oU, out):
        i32, i64 = np.int32, np.int64
        eg._launch('u2m_add', self.n, (oU, out, i64(self.n), i32(self.nL), i32(self.S), self.Moff, self.C, self.cumc, self.mr, self.Uoff, self.DR, self.rst))

    def sl(self, M, table):
        SL = cp.empty(max(1, int(self.SLoff_h[-1]))); ptr, ein, ev = table; i32, i64 = np.int32, np.int64
        eg._launch('sl_build', int(self.SLoff_h[-1]), (M, SL, i64(int(self.SLoff_h[-1])), i32(self.nL), i32(self.S), i32(self.NTL), self.SLoff, self.C, self.DL,
                                                         self.srcL, self.wstL, self.cst, self.cumc, self.Moff, ptr, ein, ev))
        return SL

    def sr(self, U, table):
        SR = cp.empty(max(1, int(self.SRoff_h[-1]))); ptr, ein, ev = table; i32, i64 = np.int32, np.int64
        eg._launch('sr_build', int(self.SRoff_h[-1]), (U, SR, i64(int(self.SRoff_h[-1])), i32(self.nR), i32(self.S), i32(self.NTR), self.SRoff, self.WRg, self.DR,
                                                         self.srcR, self.wstR, self.rst, self.cumr, self.Uoff, ptr, ein, ev))
        return SR

    def hr_add(self, U, oU, table):
        ptr, ein, ev = table; i32, i64 = np.int32, np.int64
        eg._launch('hr_add', self.n, (U, oU, i64(self.n), i32(self.nR), i32(self.S), self.Uoff, self.DR, self.rst, self.cumr, ptr, ein, ev))


def grow_left_dev(model, tables, r, p, Md, LH, GL):
    """Left environment of layer r + 1 from the left-orthonormal tensor of layer r (device, M layout) and the left
    environment of layer r (device).  Returns LH[r+1] (layout p.RHoff) and GL[r+1] (layout p.GLoffR)."""
    assert p.c.NTH == 0 and np.array_equal(p.WRp, p.WR), 'couplings two layers apart: not in the GPU back end'
    T = PlanDev(p); n = p.size; phi = cp.zeros(n); hasL = model.site_ls[r] >= 0 and r > 0; hasR = model.site_rs[r] >= 0 and r < model.n-1
    SL = T.sl(Md, tables.L[model.site_ls[r]]) if hasL else None
    for i in range(p.nL):
        Di, Ci, W = int(p.DL[i]), int(p.C[i]), int(p.WL[i])
        if not (Di and Ci):
            continue
        a, b = int(p.Moff[i]), int(p.Moff[i+1]); Mi = Md[a:b].reshape(Di, Ci); out = phi[a:b].reshape(Di, Ci)
        if r > 0:
            cp.matmul(LH[int(p.LHoff[i]):int(p.LHoff[i+1])].reshape(Di, Di), Mi, out=out)
        if W and hasL:
            out += GL[int(p.GLoff[i]):int(p.GLoff[i+1])].reshape(Di, W) @ SL[int(T.SLoff_h[i]):int(T.SLoff_h[i+1])].reshape(W, Ci)
    U = T.m2u(Md); pU = T.m2u(phi); T.hr_add(U, pU, tables.H[model.site_kind[r]])
    LHn = cp.zeros(max(1, int(p.RHoff[-1]))); GLn = cp.zeros(max(1, int(p.GLoffR[-1])))
    SR = T.sr(U, _scaled(model)[1][model.site_rs[r]]) if hasR else None
    for j in range(p.nR):
        Dj, Rj, W = int(p.DR[j]), int(p.R[j]), int(p.WR[j])
        if not (Dj and Rj):
            continue
        a, b = int(p.Uoff[j]), int(p.Uoff[j+1]); Uj = U[a:b].reshape(Rj, Dj); H = Uj.T @ pU[a:b].reshape(Rj, Dj)
        LHn[int(p.RHoff[j]):int(p.RHoff[j+1])] = (0.5*(H + H.T)).ravel()
        if W and hasR:
            GLn[int(p.GLoffR[j]):int(p.GLoffR[j+1])] = (Uj.T @ SR[int(T.SRoff_h[j]):int(T.SRoff_h[j+1])].reshape(Rj, W)).ravel()
    return LHn, GLn


def grow_right_dev(model, tables, r, p, Md, RH, GR):
    """Right environment of layer r from the right-orthonormal tensor of layer r (device, M layout) and the right
    environment of layer r + 1 (device).  Returns RH[r] (layout p.LHoff) and GR[r] (layout p.GRoffL)."""
    assert p.c.NTH == 0 and np.array_equal(p.WLq, p.WL), 'couplings two layers apart: not in the GPU back end'
    T = PlanDev(p); n = p.size; U = T.m2u(Md); oU = cp.zeros(n); hasL = model.site_ls[r] >= 0 and r > 0; hasR = model.site_rs[r] >= 0 and r < model.n-1
    SR = T.sr(U, tables.R[model.site_rs[r]]) if hasR else None
    for j in range(p.nR):
        Dj, Rj, W = int(p.DR[j]), int(p.R[j]), int(p.WR[j])
        if not (Dj and Rj):
            continue
        a, b = int(p.Uoff[j]), int(p.Uoff[j+1]); Uj = U[a:b].reshape(Rj, Dj); out = oU[a:b].reshape(Rj, Dj)
        if r < model.n-1:
            cp.matmul(Uj, RH[int(p.RHoff[j]):int(p.RHoff[j+1])].reshape(Dj, Dj), out=out)
        if W and hasR:
            out += SR[int(T.SRoff_h[j]):int(T.SRoff_h[j+1])].reshape(Rj, W) @ GR[int(p.GRoff[j]):int(p.GRoff[j+1])].reshape(W, Dj)
    T.hr_add(U, oU, tables.H[model.site_kind[r]]); phi = cp.zeros(n); T.u2m_add(oU, phi)
    RHn = cp.zeros(max(1, int(p.LHoff[-1]))); GRn = cp.zeros(max(1, int(p.GRoffL[-1])))
    SLs = T.sl(Md, _scaled(model)[0][model.site_ls[r]]) if hasL else None
    for i in range(p.nL):
        Di, Ci, W = int(p.DL[i]), int(p.C[i]), int(p.WL[i])
        if not (Di and Ci):
            continue
        a, b = int(p.Moff[i]), int(p.Moff[i+1]); Mi = Md[a:b].reshape(Di, Ci); H = phi[a:b].reshape(Di, Ci) @ Mi.T
        RHn[int(p.LHoff[i]):int(p.LHoff[i+1])] = (0.5*(H + H.T)).ravel()
        if W and hasL:
            GRn[int(p.GRoffL[i]):int(p.GRoffL[i+1])] = (SLs[int(T.SLoff_h[i]):int(T.SLoff_h[i+1])].reshape(W, Ci) @ Mi.T).ravel()
    return RHn, GRn


class DMRGGPU2(eg.DMRGGPU1b):
    version = 'gpu-v2'

    def __init__(self, *a, **k):
        super().__init__(*a, **k)                                      # the start (bond dimension 2) is grown on the host
        assert all(f < 0 for f in self.model.far_kind), 'couplings two layers apart: not in the GPU back end'
        for lst in (self.LH, self.GL, self.RH, self.GR, self.Q, self.P):
            for r in range(len(lst)):
                if lst[r] is not None:                                  # environments not grown yet stay None
                    lst[r] = cp.asarray(lst[r])
        self._env_dev = True

    def _grow_left(self, r):
        if not getattr(self, '_env_dev', False):
            return super()._grow_left(r)
        t = time.perf_counter(); Md = cp.asarray(self.M[r]); t = self._acc('env_upload', t)
        LHn, GLn = grow_left_dev(self.model, self.tables, r, self.plan[r], Md, self.LH[r], self.GL[r])
        self.LH[r+1] = LHn; self.GL[r+1] = GLn; self.Q[r+1] = GLn; self._acc('env_device', t)

    def _grow_right(self, r):
        if not getattr(self, '_env_dev', False):
            return super()._grow_right(r)
        t = time.perf_counter(); Md = cp.asarray(self.M[r]); t = self._acc('env_upload', t)
        RHn, GRn = grow_right_dev(self.model, self.tables, r, self.plan[r], Md, self.RH[r+1], self.GR[r+1])
        self.RH[r] = RHn; self.GR[r] = GRn; self.P[r] = GRn; self._acc('env_device', t)
