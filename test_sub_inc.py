#!/usr/bin/env python3
"""Incremental coset images (edc_sub source(..., incremental=True)) against the ordinary representative search: the
products must be identical in every sector of several clusters (real and complex, with and without spin flip).
Usage: test_sub_inc.py [cpu|gpu]"""
import sys
import numpy as np
import edc_lattices as L, edc_lattice as el, edc_hubbard_sym as hs, edc_sub

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
xp = np
if DEVICE == 'gpu':
    import cupy as cp
    xp = cp


def check(name, cl, model):
    N = cl.N; nup = N//2; worst = 0.0; n = 0
    for k, ir, fl, pops, mult in hs.sectors(cl, nup, N-nup):
        e0 = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=DEVICE, point_ops=pops, incremental=False)
        e1 = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=DEVICE, point_ops=pops, incremental=True)
        assert e0.dim == e1.dim
        if e0.dim == 0:
            continue
        rng = np.random.default_rng(n)
        x = rng.standard_normal(e0.dim) if e0.real else rng.standard_normal(e0.dim)+1j*rng.standard_normal(e0.dim)
        X = xp.asarray(x)
        y0 = e0.matvec(X, xp.zeros_like(X)); y1 = e1.matvec(X, xp.zeros_like(X))
        err = float(xp.max(xp.abs(y0-y1)))/max(1.0, float(xp.max(xp.abs(y0))))
        worst = max(worst, err); n += 1
        assert err < 1e-14, (name, k, ir, fl, err)
    print(f'{name}: {n} sectors, max relative |incremental - ordinary| = {worst:.1e}', flush=True)


if __name__ == '__main__':
    kg = L.LatticeCluster('kagome', (2, 0), (0, 2)); check('kagome 12', kg, kg.heisenberg(1.0))
    kg2 = L.LatticeCluster('kagome', (3, 0), (0, 2)); check('kagome 18', kg2, kg2.heisenberg(1.0, 0.2))
    tr = L.LatticeCluster('triangular', (2, 2), (-2, 4)); check('triangular 12 (J2 = 0.1)', tr, tr.heisenberg(1.0, 0.1))
    tr2 = L.LatticeCluster('triangular', (1, 4), (-4, 5)); check('triangular 21', tr2, tr2.heisenberg(1.0))
    hc = L.LatticeCluster('honeycomb', (3, 0), (0, 3)); check('honeycomb 18', hc, hc.heisenberg(1.0, 0.3))
    sq = el.Cluster((4, 0), (0, 4)); check('square 4x4 J1-J2', sq, sq.j1j2(1.0, 0.5))
    sq2 = el.Cluster((4, 2), (-2, 4)); check('square 20', sq2, sq2.j1j2(1.0, 0.5))
    print('all passed')
