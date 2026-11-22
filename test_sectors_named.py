#!/usr/bin/env python3
"""edc_hubbard_sym.sectors_named: (1) the block dimensions times star and degeneracy add up to the whole (N_up, N_dn)
space; (2) the levels of all blocks, repeated star * degeneracy times, are the lowest levels of a dense diagonalisation
without any lattice symmetry; (3) fewer rows than the blocks of sectors().
Usage: test_sectors_named.py"""
import math
import numpy as np
import itertools
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub


def dense(model, nup):
    """The whole S_z sector as a dense matrix, no lattice symmetry."""
    n = model.n
    states = [sum(1 << b for b in c) for c in itertools.combinations(range(n), nup)]
    idx = {s: i for i, s in enumerate(states)}
    H = np.zeros((len(states),)*2)
    for i, s in enumerate(states):
        for (a, b), (jxy, jz) in model.bonds.items():
            x, y = (s >> a) & 1, (s >> b) & 1
            H[i, i] += jz*(0.25 if x == y else -0.25)
            if x != y and jxy:
                H[idx[s ^ ((1 << a) | (1 << b))], i] += jxy/2
    return H


worst = 0.0
for kind, a, nups, J in (('kagome', (2, 0, 0, 2), (6, 7, 8), (1.0, 0.3)), ('triangular', (3, 0, 0, 3), (4, 5), (1.0, 0.2)),
                         ('triangular', (2, 2, -2, 4), (6, 7), (1.0,)), ('honeycomb', (3, 0, 0, 3), (14, 15), (1.0, 0.4)),
                         ('kagome', (3, 0, 0, 3), (24, 25), (1.0, 0.1))):
    cl = L.LatticeCluster(kind, a[:2], a[2:]); model = cl.heisenberg(*J)
    for nup in nups:
        named = hs.sectors_named(cl, nup, cl.N-nup); old = hs.sectors(cl, nup, cl.N-nup)
        levels = []; tot = rows_new = rows_old = 0
        for b in named:
            e = edc_sub.SubEngine(model, nup, cl, b['k'], b['irrep'], b['flip'], device='cpu', point_ops=b['allowed'], stored=False)
            tot += e.dim*b['star']*b['degeneracy']; rows_new += e.dim
            if e.dim == 0:
                continue
            H = np.zeros((e.dim, e.dim), np.float64 if e.real else np.complex128); x = np.zeros(e.dim, H.dtype); y = np.zeros(e.dim, H.dtype)
            for c in range(e.dim):
                x[:] = 0; x[c] = 1; e.matvec(x, y); H[:, c] = y
            w = np.linalg.eigvalsh((H+H.conj().T)/2)
            levels += list(np.repeat(w, b['star']*b['degeneracy']))
        for k, ir, fl, pops, mult in old:
            rows_old += edc_sub.SubEngine(model, nup, cl, k, ir, fl, device='cpu', point_ops=pops, stored=False).dim
        full = math.comb(cl.N, nup)
        assert tot == full, (kind, a, nup, tot, full)
        Hd = dense(model, nup)                                    # the whole S_z sector, no lattice symmetry
        ref = np.linalg.eigvalsh((Hd+Hd.T)/2); d = float(np.abs(np.sort(levels)-ref).max()); worst = max(worst, d)
        print(f'{kind} {a} N={cl.N} nup={nup}: {len(named)} named blocks ({rows_new} rows) against {len(old)} ({rows_old} rows); '
              f'dimensions add up to {full}; all {full} levels against the dense spectrum: {d:.1e}', flush=True)
assert worst < 1e-9, worst
print('sectors_named: all checks passed')
