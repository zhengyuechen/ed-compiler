#!/usr/bin/env python3
"""CPU benchmark variant with an explicit BLAS thread setting for reorthogonalisation."""
import os, time
import numpy as np
import edc_dmrg as ed

_p = ed._p


def numpy_blas_threads(n=None):
    """Set the threads of numpy's BLAS (not scipy's) to n; returns {library file: threads} of every BLAS loaded."""
    from threadpoolctl import ThreadpoolController
    out = {}
    for lib in ThreadpoolController().lib_controllers:
        if lib.user_api == 'blas':
            if n and 'numpy' in (lib.filepath or ''):
                lib.set_num_threads(n)
            out[os.path.basename(lib.filepath or '?')] = lib.num_threads
    return out


class DMRGHost2(ed.DMRG):
    version = 'host-2'

    def solve(self, r, tol=1e-6, maxit=40):
        p = self.plan[r]; lib = self.lib; n = p.size; U = np.empty(n); oU = np.empty(n)
        LH, GL, RH, GR, QL, PR = (_p(a) for a in (self.LH[r], self.GL[r], self.RH[r+1], self.GR[r+1], self.Q[r], self.P[r+1]))
        t0 = time.perf_counter(); v = self.M[r]/np.linalg.norm(self.M[r])
        if self.noise > 0:
            xi = self.rng.standard_normal(n); v = v + (self.noise/np.linalg.norm(xi))*xi; v /= np.linalg.norm(v)
        m = min(maxit, n); V = np.empty((m, n)); V[0] = v; al = []; be = []; E = None; y = np.array([1.0]); k = 1
        for it in range(m):
            w = np.zeros(n); lib.heff(p.ref, LH, GL, RH, GR, QL, PR, _p(V[it]), _p(w), _p(U), _p(oU)); self.stats['products'] += 1
            a = float(V[it] @ w); al.append(a); w -= a*V[it]
            if be:
                w -= be[-1]*V[it-1]
            w -= (V[:k] @ w) @ V[:k]; b = float(np.linalg.norm(w))
            ev, Sv = np.linalg.eigh(np.diag(al) + np.diag(be, 1) + np.diag(be, -1)); E = float(ev[0]); y = Sv[:, 0]
            if b*abs(y[-1]) < tol*max(1.0, abs(E)) or b < 1e-13 or k == n or k == m:
                break
            be.append(b); V[k] = w/b; k += 1
        x = y[:k] @ V[:k]; self.M[r] = x/np.linalg.norm(x)
        self.stats['solves'] += 1; self.stats['t_products'] += time.perf_counter()-t0
        return E
