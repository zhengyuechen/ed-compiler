#!/usr/bin/env python3
"""Version 3 of the GPU layer DMRG with a record of every layer solve: how many Krylov vectors it used
and the residual estimate at its end against the tolerance.  A solve stops when the estimate b |y_last| falls below
tol * max(1, |E|), or when the 40 reserved vectors are used up; "used 40" alone does not say whether the tolerance was
reached at that step, the residual does.  The arithmetic of the solve is that of edc_dmrg_gpu3.DMRGGPU3 line by line
(same random stream, same energies digit by digit); only the record is added.

    eng.reslog   one entry per solve: sweep (counted from 1), D of the sweep, layer, tensor size n, vectors k,
                 residual estimate, tolerance (absolute), capped = k reached the reserve with the residual above tolerance
    cap_table(eng)   per sweep: solves, solves that used all vectors, of those above tolerance, largest residual / tolerance"""
import time
import numpy as np
import cupy as cp
import edc_dmrg_gpu as eg
import edc_dmrg_gpu3 as e3


class DMRGGPU3r(e3.DMRGGPU3):
    version = 'gpu-v3r'

    def sweep(self, Dmax, alpha, tol=1e-6, noise=0.0):
        if not hasattr(self, 'reslog'):
            self.reslog = []; self.sweep_no = 0
        self.sweep_no += 1; self.sweep_D = Dmax
        return super().sweep(Dmax, alpha, tol, noise)

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
        res = b*abs(y[-1]); ta = tol*max(1.0, abs(E))                  # the record: nothing below depends on it
        if hasattr(self, 'reslog'):
            self.reslog.append(dict(sweep=self.sweep_no, D=int(self.sweep_D), layer=r, n=int(n), k=k, residual=res, tolerance=ta, capped=bool(k == m and res >= ta and m < n)))
        x = cp.asarray(y[:k]) @ V[:k]; x /= cp.linalg.norm(x); self.M[r] = x; H._t('lanczos', t0)
        self.peak = max(self.peak, cp.get_default_memory_pool().total_bytes()); self._dev = (r, H, x)
        del V, v
        self.stats['solves'] += 1; self.stats['t_products'] += time.perf_counter()-t00
        return E


def cap_table(eng, maxit=40):
    rows = []
    for s in sorted({e['sweep'] for e in eng.reslog}):
        es = [e for e in eng.reslog if e['sweep'] == s]; full = [e for e in es if e['k'] >= min(maxit, e['n'])]; cap = [e for e in es if e['capped']]
        nmax = max(e['n'] for e in es); big = [e for e in es if e['n'] >= 0.8*nmax]
        rows.append(dict(sweep=s, D=es[0]['D'], tolerance=es[0]['tolerance'], solves=len(es), used_all=len(full), above_tolerance=len(cap),
                         largest_ratio=max((e['residual']/e['tolerance'] for e in cap), default=0.0), largest_residual=max((e['residual'] for e in cap), default=0.0),
                         largest_tensors=len(big), largest_tensors_above_tolerance=sum(e['capped'] for e in big), last_solve_ratio=es[-1]['residual']/es[-1]['tolerance']))
    return rows
