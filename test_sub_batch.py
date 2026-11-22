#!/usr/bin/env python3
"""Batched product of the sub engine (R vectors at once, interleaved) against R single products, every sector of a
few clusters (real and complex sectors, with and without spin flip).  Usage: test_sub_batch.py [cpu|gpu]"""
import sys
import numpy as np
import edc_lattices as L, edc_lattice as el, edc_hubbard_sym as hs, edc_sub

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
xp = np
if DEVICE == 'gpu':
    import cupy as cp
    xp = cp


def check(name, cl, model, R=5):
    N = cl.N; nup = N//2; worst = 0.0; n = 0
    for k, ir, fl, pops, mult in hs.sectors(cl, nup, N-nup):
        e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=DEVICE, point_ops=pops)
        if e.dim == 0:
            continue
        dt = np.float64 if e.real else np.complex128
        rng = np.random.default_rng(n)
        X = rng.standard_normal((e.dim, R)).astype(dt)
        if not e.real:
            X = X+1j*rng.standard_normal((e.dim, R))
        Y0 = rng.standard_normal((e.dim, R)).astype(dt)
        Xd, Yd = xp.array(X, copy=True), xp.array(Y0, copy=True)
        apply = e.batched(R)
        apply(Xd, Yd, accumulate=True)
        Yb = Yd.get() if xp is not np else Yd
        ref = np.empty_like(Y0)
        for r in range(R):
            x = xp.asarray(np.ascontiguousarray(X[:, r])); y = xp.asarray(np.ascontiguousarray(Y0[:, r]))
            e.matvec(x, y, accumulate=True)
            ref[:, r] = y.get() if xp is not np else y
        err = np.max(np.abs(Yb-ref))/max(1.0, np.max(np.abs(ref)))
        worst = max(worst, err); n += 1
        assert err < 1e-13, (name, k, ir, fl, err)
    print(f'{name}: {n} sectors, R = {R}, max relative |batched - single| = {worst:.1e}', flush=True)


if __name__ == '__main__':
    kg = L.LatticeCluster('kagome', (2, 0), (0, 2)); check('kagome 12', kg, kg.heisenberg(1.0))
    tr = L.LatticeCluster('triangular', (2, 2), (-2, 4)); check('triangular 12 (J2 = 0.1)', tr, tr.heisenberg(1.0, 0.1))
    hc = L.LatticeCluster('honeycomb', (3, 0), (0, 3)); check('honeycomb 18', hc, hc.heisenberg(1.0), R=8)
    sq = el.Cluster((4, 0), (0, 4)); check('square 4x4 J1-J2', sq, sq.j1j2(1.0, 0.5), R=3)
    print('all passed')
