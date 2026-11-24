#!/usr/bin/env python3
"""Layer DMRG with GPU-resident environments and moves between layers.

  1. The layer tensors stay on the device for the whole run: a solve no longer downloads its result, a move no longer
     uploads a tensor.
  2. Sketch of the expansion: the slabs S^R (S^L) of the solved tensor with the slab kernel of the product, then per
     label one product with a random matrix drawn on the device; the random directions too are drawn on the device.
  3. Assembling: kept columns, carried part of the neighbour tensor (C M_next) and both layout changes on the device.
  4. The many small blocks (fewer than 48 columns, or not four times taller than wide) go to the host together in one
     transfer, LAPACK decomposes them as before, the kept vectors come back in one transfer.
  5. The heavy blocks:  DMRGGPU3   (factor = 'host')   version 1's Cholesky QR 2, with the Cholesky factorisations and the
                                                         SVD of the small square factor on the host;
                        DMRGGPU3d  (factor = 'device') everything on the device: Cholesky QR 2 with cuSOLVER, then the left
                                                         singular vectors of the small factor C from the symmetric
                                                         eigenproblem of C C^T.  A = Q1 T2 Uc s V^T with Q1 T2 Uc orthonormal to
                                                         working precision; singular values below ~1e-8 of the largest are
                                                         not resolved (blocks that ill-conditioned fail the Cholesky QR and
                                                         take the host SVD, as in version 1).  The carried part
                                                         s_k V^T[:k] = Uc_k^T C needs no division by s.  No host round trip
                                                         per block: the checks and singular values of all blocks of a move
                                                         come back in one transfer.
Random numbers come from cupy's generator (seeded with the run's seed), so runs are not digit-identical to version 2.
Nearest-neighbour couplings only, like the rest of the GPU back end."""
import time, math
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
import edc_dmrg as ed
import edc_dmrg_gpu as eg
import edc_dmrg_gpu2 as e2
from edc_dmrg_factor import tall_qr


def _pdev(p):
    """The device offset tables of a plan, made once per plan."""
    T = getattr(p, '_pdev', None)
    if T is None:
        T = p._pdev = e2.PlanDev(p)
    return T


class DMRGGPU3(e2.DMRGGPU2):
    version = 'gpu-v3'; factor = 'host'
    product_cls = eg.Product

    def __init__(self, model, nup, K=0, D0=2, seed=0):
        super().__init__(model, nup, K, D0, seed); cp.random.seed(seed)
        self.M = [cp.asarray(m) if m is not None else None for m in self.M]

    # ---- layer solve: as version 0 (edc_dmrg_gpu.DMRGGPU.solve), but the start comes from and the result stays on the device
    def solve(self, r, tol=1e-6, maxit=40):
        assert not self.host
        t00 = time.perf_counter(); H = self.product_cls(self, r, self.prof); n = H.n; t0 = time.perf_counter()
        v = self.M[r]/cp.linalg.norm(self.M[r])
        if self.noise > 0:
            xi = cp.random.standard_normal(n); v = v + (self.noise/float(cp.linalg.norm(xi)))*xi; v /= cp.linalg.norm(v)
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
        x = cp.asarray(y[:k]) @ V[:k]; x /= cp.linalg.norm(x); self.M[r] = x; H._t('lanczos', t0)
        self.peak = max(self.peak, cp.get_default_memory_pool().total_bytes()); self._dev = (r, H, x)
        del V, v
        self.stats['solves'] += 1; self.stats['t_products'] += time.perf_counter()-t00
        return E

    # ---- factorisation of the blocks of one move: every block A (m x n, device) as A = U s V^T
    #      results: ('host', s, Us, Vh) | ('devh', s, Q, Uc, Vh) with U = Q Uc | ('devd', s, Q1, T2, Uc, C) with U = Q1 T2 Uc, s V^T = Uc^T C
    def _heavy(self, A):
        return A.shape[1] >= self.min_cols and A.shape[0] >= self.min_ratio*A.shape[1]

    def _host_svds(self, blocks):
        if not blocks:
            return {}
        t = time.perf_counter(); keys = list(blocks); flat = cp.asnumpy(cp.concatenate([blocks[q].ravel() for q in keys])); out = {}; o = 0
        for q in keys:
            m, n = blocks[q].shape; Us, s, Vh = ed.svd(flat[o:o+m*n].reshape(m, n), full_matrices=False); o += m*n; out[q] = ('host', s, Us, Vh)
        self.blocks_host += len(keys); self._acc('mv_host_svd', t)
        return out

    def _factor_host(self, blocks):                                   # version 1's route, block by block
        out = {}
        for q, A in blocks.items():
            t = time.perf_counter(); f = tall_qr(A, cp, cp.asnumpy, cp.asarray)
            if f is None:
                self._acc('mv_fallback', t); out.update(self._host_svds({q: A})); continue
            t = self._acc('mv_gram_cholesky', t); Uc, s, Vh = ed.svd(f[1], full_matrices=True); self.blocks_dev += 1; self._acc('mv_small_svd', t)
            out[q] = ('devh', s, f[0], Uc, Vh)
        return out

    def _factor_device(self, blocks):
        if not blocks:
            return {}
        t = time.perf_counter(); pend = {}
        for q, A in blocks.items():
            n = A.shape[1]; I = cp.eye(n)
            G = A.T @ A; d = cp.sqrt(cp.diag(G)); L1 = cp.linalg.cholesky(G/cp.outer(d, d))
            T1 = solve_triangular(L1, cp.diag(1.0/d), lower=True).T                     # D^-1 L1^-T
            Q1 = A @ T1; G2 = Q1.T @ Q1; L2 = cp.linalg.cholesky(G2); T2 = solve_triangular(L2, I, lower=True).T   # L2^-T
            C = (L2.T @ L1.T)*d[None, :]                                               # A = Q1 T2 C
            w, Uc = cp.linalg.eigh(C @ C.T)
            ok = cp.isfinite(C).all() & cp.isfinite(w).all() & (cp.abs(G2 - I).max() < 0.05) & (d > 0).all()
            pend[q] = (Q1, T2, Uc[:, ::-1], cp.sqrt(cp.maximum(w[::-1], 0.0)), C, ok)
        t = self._acc('mv_factor_dev', t)
        flags = cp.asnumpy(cp.stack([v[5] for v in pend.values()])); sv = cp.asnumpy(cp.concatenate([v[3] for v in pend.values()]))
        self._acc('mv_factor_dev', t); out = {}; bad = {}; o = 0
        for (q, v), f in zip(pend.items(), flags):
            n = v[3].size; s = sv[o:o+n]; o += n
            if f:
                out[q] = ('devd', s, v[0], v[1], v[2], v[4]); self.blocks_dev += 1
            else:
                bad[q] = blocks[q]
        out.update(self._host_svds(bad))
        return out

    def _factor_all(self, blocks):
        heavy = {q: A for q, A in blocks.items() if self._heavy(A)}; small = {q: A for q, A in blocks.items() if q not in heavy}
        res = self._host_svds(small)
        res.update(self._factor_host(heavy) if self.factor == 'host' else self._factor_device(heavy))
        return res

    def _assemble(self, res, new, carry):
        """For every label q with new[q] > 0: U_k (device, m x k) and, where carry[q] > 0, Cm = s_k V^T[:k, :carry[q]]
        (device, k x carry[q]); and the kept weight sum |Cm|^2.  The small host pieces travel in one upload."""
        parts = []
        for q, v in res.items():
            k, c = int(new[q]), int(carry[q])
            if not k or v[0] == 'devd':
                continue
            parts.append((q, 'U', v[2][:, :k] if v[0] == 'host' else v[3][:, :k]))
            if c:
                parts.append((q, 'C', v[1][:k, None]*v[-1][:k, :c]))
        kept = sum(float(np.sum(a*a)) for _, w, a in parts if w == 'C'); up = {}
        if parts:
            flat = cp.asarray(np.concatenate([np.ascontiguousarray(a).ravel() for _, _, a in parts])); o = 0
            for q, w, a in parts:
                up[(q, w)] = flat[o:o+a.size].reshape(a.shape); o += a.size
        out = {}; kd = []
        for q, v in res.items():
            k, c = int(new[q]), int(carry[q])
            if not k:
                continue
            if v[0] == 'host':
                out[q] = (up[(q, 'U')], up.get((q, 'C')))
            elif v[0] == 'devh':
                out[q] = (v[2] @ up[(q, 'U')], up.get((q, 'C')))
            else:
                _, s, Q1, T2, Uc, C = v; Y = Uc[:, :k]; Cm = (Y.T @ C[:, :c]) if c else None
                if Cm is not None:
                    kd.append(cp.sum(Cm*Cm))
                out[q] = (Q1 @ (T2 @ Y), Cm)
        if kd:
            kept += float(cp.sum(cp.stack(kd)))
        return out, kept

    # ---- moves
    def move_right(self, r, Dmax, alpha, cutoff=1e-12):
        rr, H, x = self._dev; self._dev = None; assert rr == r and not self.host
        p = self.plan[r]; md = self.model; t0 = time.perf_counter(); i32, i64 = np.int32, np.int64
        kv = self._sketch(p.DR, p.WR, p.R-p.DR, alpha, int(p.DR.sum()) < Dmax); Ud = H.to_u(x); sk = bool(kv.any()) and H.hasR
        if sk:
            ptr, ein, ev = H.tR
            eg._launch('sr_build', H.SR.size, (Ud, H.SR, i64(H.SR.size), i32(H.nR), i32(H.S), i32(H.NTR), H.SRoff, H.WRg, H.DR,
                                               H.srcR, H.wstR, H.rst, H.cumr, H.Uoff, ptr, ein, ev))
        blocks = {}
        for j in np.nonzero((p.R > 0) & (p.DR+kv > 0))[0]:
            Rj, Dj, w, W = int(p.R[j]), int(p.DR[j]), int(kv[j]), int(p.WR[j]); A = Ud[int(p.Uoff[j]):int(p.Uoff[j+1])].reshape(Rj, Dj)
            if w:
                B = cp.random.standard_normal((Rj, w))*(self.random/math.sqrt(Rj))
                if sk and W:
                    B += (H.SR[int(H.SRoff_h[j]):int(H.SRoff_h[j+1])].reshape(Rj, W) @ cp.random.standard_normal((W, w)))*(1.0/math.sqrt(w))
                A = cp.concatenate([A, alpha*B], axis=1)
            blocks[j] = A
        self._acc('mv_sketch', t0); res = self._factor_all(blocks)
        keep = self._keep({j: v[1] for j, v in res.items()}, Dmax, cutoff)
        pn = self.plan[r+1]; Mn = self.M[r+1]; old = p.DR.copy(); new = np.zeros_like(old)
        for j, k in keep.items():
            new[j] = k
        t = time.perf_counter(); self.bond[r+1].dims[:] = new; p2 = ed.Plan(md, r, self.bond[r], self.bond[r+1]); pn2 = ed.Plan(md, r+1, self.bond[r+1], self.bond[r+2])
        t = self._acc('mv_plans', t); out, kept = self._assemble(res, new, np.where(pn.C > 0, old, 0))
        U2 = cp.zeros(p2.size); Mn2 = cp.zeros(pn2.size)
        for j, (Uk, Cm) in out.items():
            U2[int(p2.Uoff[j]):int(p2.Uoff[j+1])] = Uk.ravel()
            if Cm is not None:
                Mn2[int(pn2.Moff[j]):int(pn2.Moff[j+1])] = (Cm @ Mn[int(pn.Moff[j]):int(pn.Moff[j+1])].reshape(int(old[j]), int(pn.C[j]))).ravel()
        M2 = cp.zeros(p2.size); _pdev(p2).u2m_add(U2, M2); Mn2 /= cp.linalg.norm(Mn2)
        self._acc('mv_assemble', t); del res, out, blocks, H, x, Ud, U2
        self.stats['truncation'] += max(0.0, 1.0-kept); self.stats['t_svd'] += time.perf_counter()-t0
        self.plan[r] = p2; self.M[r] = M2; self.plan[r+1] = pn2; self.M[r+1] = Mn2
        t0 = time.perf_counter(); self._grow_left(r); self.stats['t_env'] += time.perf_counter()-t0

    def move_left(self, r, Dmax, alpha, cutoff=1e-12):
        rr, H, x = self._dev; self._dev = None; assert rr == r and not self.host
        p = self.plan[r]; md = self.model; t0 = time.perf_counter(); i32, i64 = np.int32, np.int64
        kv = self._sketch(p.DL, p.WL, p.C-p.DL, alpha, int(p.DL.sum()) < Dmax); sk = bool(kv.any()) and H.hasL
        if sk:
            ptr, ein, ev = H.tL
            eg._launch('sl_build', H.SL.size, (x, H.SL, i64(H.SL.size), i32(H.nL), i32(H.S), i32(H.NTL), H.SLoff, H.C, H.DL,
                                               H.srcL, H.wstL, H.cst, H.cumc, H.Moff, ptr, ein, ev))
        blocks = {}
        for i in np.nonzero((p.C > 0) & (p.DL+kv > 0))[0]:
            Ci, Di, w, W = int(p.C[i]), int(p.DL[i]), int(kv[i]), int(p.WL[i]); A = x[int(p.Moff[i]):int(p.Moff[i+1])].reshape(Di, Ci)
            if w:
                B = cp.random.standard_normal((w, Ci))*(self.random/math.sqrt(Ci))
                if sk and W:
                    B += (cp.random.standard_normal((w, W)) @ H.SL[int(H.SLoff_h[i]):int(H.SLoff_h[i+1])].reshape(W, Ci))*(1.0/math.sqrt(w))
                A = cp.concatenate([A, alpha*B], axis=0)
            blocks[i] = cp.ascontiguousarray(A.T)                         # A^T = U s V^T  ->  new rows U_k^T, carried part V s
        self._acc('mv_sketch', t0); res = self._factor_all(blocks)
        keep = self._keep({i: v[1] for i, v in res.items()}, Dmax, cutoff)
        pp = self.plan[r-1]; old = p.DL.copy(); new = np.zeros_like(old)
        for i, k in keep.items():
            new[i] = k
        t = time.perf_counter(); self.bond[r].dims[:] = new; p2 = ed.Plan(md, r, self.bond[r], self.bond[r+1]); pp2 = ed.Plan(md, r-1, self.bond[r-1], self.bond[r])
        t = self._acc('mv_plans', t); Up = _pdev(pp).m2u(self.M[r-1]); out, kept = self._assemble(res, new, np.where(pp.R > 0, old, 0))
        M2 = cp.zeros(p2.size); Up2 = cp.zeros(pp2.size)
        for i, (Uk, Cm) in out.items():
            M2[int(p2.Moff[i]):int(p2.Moff[i+1])] = cp.ascontiguousarray(Uk.T).ravel()
            if Cm is not None:
                Up2[int(pp2.Uoff[i]):int(pp2.Uoff[i+1])] = (Up[int(pp.Uoff[i]):int(pp.Uoff[i+1])].reshape(int(pp.R[i]), int(old[i])) @ Cm.T).ravel()
        Mp2 = cp.zeros(pp2.size); _pdev(pp2).u2m_add(Up2, Mp2); Mp2 /= cp.linalg.norm(Mp2)
        self._acc('mv_assemble', t); del res, out, blocks, H, x, Up, Up2
        self.stats['truncation'] += max(0.0, 1.0-kept); self.stats['t_svd'] += time.perf_counter()-t0
        self.plan[r] = p2; self.M[r] = M2; self.plan[r-1] = pp2; self.M[r-1] = Mp2
        t0 = time.perf_counter(); self._grow_right(r); self.stats['t_env'] += time.perf_counter()-t0


class DMRGGPU3d(DMRGGPU3):
    version = 'gpu-v3d'; factor = 'device'
