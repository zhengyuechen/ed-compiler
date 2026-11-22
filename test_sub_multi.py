#!/usr/bin/env python3
"""Sublattice-table engine on several GPUs of one node against one GPU: every sector of small clusters (2 and 4
devices), then timing on larger sectors.  Run with 4 GPUs."""
import time, math
import edc_lattice as el, edc_lattices as L, edc_sub as es

for name, cl, m, nup in (('triangular (1,4)', L.LatticeCluster('triangular', (1, 4), (-4, 5)), None, 10),
                         ('kagome 27b', L.LatticeCluster('kagome', (3, 0), (0, 3)), None, 13),
                         ('honeycomb 24', L.LatticeCluster('honeycomb', (2, 2), (-2, 4)), None, 12)):
    m = cl.heisenberg(1.0)
    worst, n = 0.0, 0
    for k in cl.momenta():
        for ir in cl.irrep_table(k):
            for fl in ((1, -1) if 2*nup == cl.N else (0,)):
                one = es.SubEngine(m, nup, cl, k, ir, fl, device='gpu')
                if one.dim <= 400:
                    continue
                E1 = one.ground_energy(tol=1e-13)[0]
                for devs in ([0, 1], [0, 1, 2, 3]):
                    mul = es.SubEngine(m, nup, cl, k, ir, fl, device='gpu', devices=devs)
                    assert mul.dim == one.dim
                    worst = max(worst, abs(mul.ground_energy(tol=1e-13)[0]-E1)); n += 1
    print(f'{name:18s} N={cl.N}: {n} (sector, device set) pairs, max |E_multi - E_one| {worst:.1e}', flush=True)

for name, cl, m, nup, k, ir, fl in (('kagome 36d', L.LatticeCluster('kagome', (2, 2), (-2, 4)), 'h', 18, (0.0, 0.0), 'X0', 1),
                                    ('square 40', el.tilted(6, 2), 'j', 20, (0.0, 0.0), 'A1', 1)):
    model = cl.heisenberg(1.0) if m == 'h' else cl.j1j2(1.0, 0.5)
    for devs in (None, [0, 1], [0, 1, 2, 3]):
        t0 = time.perf_counter(); e = es.SubEngine(model, nup, cl, k, ir, fl, device='gpu', devices=devs); setup = time.perf_counter()-t0
        mv = e.time_matvec(); E, it, t = e.ground_energy(tol=1e-12)
        print(f'{name:12s} dim {e.dim} devices {devs}: setup {setup:.1f} s, product {mv:.3f} s, {it} steps, {t:.1f} s, E {E!r}', flush=True)
