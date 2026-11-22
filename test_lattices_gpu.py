#!/usr/bin/env python3
"""Lean GPU engine on triangular, honeycomb and kagome clusters (edc_lattices) against the CPU any-group engine,
every momentum and every 1D irrep of the little group."""
import edc_k, edc_lattices as L, edc_gpu_lean as gl

for kind, A1, A2, J, nup in (('triangular', (2, 2), (-2, 4), (1.0, 0.3), None), ('kagome', (2, 0), (0, 2), (1.0,), None),
                             ('kagome', (1, 1), (-1, 2), (1.0, 0.2), 4), ('honeycomb', (2, 2), (-2, 4), (1.0, 0.2), None),
                             ('kagome', (3, 0), (0, 2), (1.0,), None), ('triangular', (1, 4), (-4, 5), (1.0,), 10),
                             ('kagome', (3, 0), (0, 3), (1.0,), 13)):
    cl = L.LatticeCluster(kind, A1, A2); m = cl.heisenberg(*J); nup = nup if nup is not None else cl.N//2
    worst, n, nreal = 0.0, 0, 0
    for k in cl.momenta():
        for ir in cl.irrep_table(k):
            perms, chars, ops = cl.group(k, ir)
            c = edc_k.KEngine(m, nup, perms, chars, threads=8)
            g = gl.GPULean(m, nup, cl, k, ir)
            assert c.dim == g.dim, (kind, k, ir, c.dim, g.dim)
            if c.dim:
                worst = max(worst, abs(c.ground_energy(tol=1e-13)[0]-g.ground_energy(tol=1e-13)[0])); n += 1; nreal += g.real
    print(f'{kind:10s} {A1} {A2} N={cl.N:2d} nup={nup} J={J} n1={cl.n1} rotmin={g.rotmin}: {n:3d} sectors ({nreal} real), '
          f'max |E_gpu - E_cpu| {worst:.1e}', flush=True)
