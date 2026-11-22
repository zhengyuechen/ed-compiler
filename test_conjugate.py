#!/usr/bin/env python3
"""edc_hubbard_sym.conjugate_partners: every block marked as the complex conjugate of another has the same dimension
and the same lowest levels as its partner (spin models and the Hubbard model, several clusters and fillings).
Usage: test_conjugate.py"""
import numpy as np
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub

pairs = 0; worst = 0.0; rows_all = rows_saved = 0
for kind, a, nups, J in (('kagome', (3, 0, 0, 2), (9, 10, 12), (1.0, 0.2)), ('kagome', (2, 2, -2, 2), (12, 14), (1.0, -0.1)),
                         ('kagome', (3, 0, 0, 3), (16, 19), (1.0, 0.05)), ('triangular', (1, 4, -4, 5), (10, 12), (1.0, 0.1)),
                         ('triangular', (3, 0, 0, 3), (4, 5), (1.0,)), ('honeycomb', (2, 2, -2, 4), (12, 13), (1.0, 0.3))):
    cl = L.LatticeCluster(kind, a[:2], a[2:]); model = cl.heisenberg(*J)
    for nup in nups:
        secs = hs.sectors(cl, nup, cl.N-nup); partner = hs.conjugate_partners(cl, secs); E = {}; dims = {}
        for i, (k, ir, fl, pops, mult) in enumerate(secs):
            e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device='cpu', point_ops=pops); dims[i] = e.dim
            rows_all += e.dim*mult
            if e.dim:
                E[i] = np.atleast_1d(e.ground_energy(nev=min(3, e.dim), tol=1e-13)[0])
        for i, j in enumerate(partner):
            if j is None:
                continue
            assert dims[i] == dims[j], (kind, a, nup, i, j, dims[i], dims[j])
            pairs += 1; rows_saved += dims[i]*secs[i][4]
            if dims[i]:
                n = min(len(E[i]), len(E[j])); worst = max(worst, float(np.abs(E[i][:n]-E[j][:n]).max()))
        print(f'{kind} {a} N={cl.N} nup={nup}: {len(secs)} blocks, {sum(p is not None for p in partner)} conjugates', flush=True)
print(f'{pairs} conjugate pairs, largest difference of the lowest levels {worst:.1e}; rows saved {rows_saved/rows_all:.1%}')
assert pairs > 0 and worst < 1e-9, worst
print('conjugate partners: all checks passed')
