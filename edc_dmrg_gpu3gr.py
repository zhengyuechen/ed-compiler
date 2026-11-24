#!/usr/bin/env python3
"""Grouped GPU products with per-layer Krylov counts and residuals.

The sweep log records energy, discarded weight, elapsed time and pool size.
Set free_pool to False to retain unused CuPy allocations between sweeps.
"""
import time
import cupy as cp
import edc_dmrg_gpu3g as e3g
import edc_dmrg_gpu3r as e3r


class DMRGGPU3gr(e3r.DMRGGPU3r):
    version = 'gpu-v3gr'; product_cls = e3g.ProductG; free_pool = True

    def sweep(self, Dmax, alpha, tol=1e-6, noise=0.0):
        if not hasattr(self, 'sweep_log'):
            self.sweep_log = []
        t0 = time.perf_counter(); E = super().sweep(Dmax, alpha, tol, noise); pool = cp.get_default_memory_pool()
        self.sweep_log.append(dict(D=int(Dmax), alpha=float(alpha), tol=float(tol), E=float(E), discarded=float(self.stats['truncation']),
                                   largest_bond=int(max(b.dims.sum() for b in self.bond)), seconds=time.perf_counter()-t0, pool_gib=pool.total_bytes()/2**30))
        if self.free_pool:
            pool.free_all_blocks()
        return E
