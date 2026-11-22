#!/usr/bin/env python3
"""Sublattice-table engine (edc_sub) against the any-group engine (edc_k): every momentum and 1D irrep of small square,
triangular, honeycomb and kagome clusters; with spin flip, dim(no flip) = dim(even) + dim(odd) and
E(no flip) = min(E(even), E(odd)).  Usage: test_sub.py [cpu|gpu]"""
import sys
import edc_k, edc_lattice as el, edc_lattices as L, edc_sub as es

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
CASES = [('square tilted (3,1)', el.tilted(3, 1), lambda c: c.j1j2(1.0, 0.3), None),
         ('square 4x4', el.Cluster((4, 0), (0, 4)), lambda c: c.j1j2(1.0, 0.5), None),
         ('triangular (2,2)', L.LatticeCluster('triangular', (2, 2), (-2, 4)), lambda c: c.heisenberg(1.0, 0.3), None),
         ('kagome (2,0)', L.LatticeCluster('kagome', (2, 0), (0, 2)), lambda c: c.heisenberg(1.0), None),
         ('kagome (1,1)', L.LatticeCluster('kagome', (1, 1), (-1, 2)), lambda c: c.heisenberg(1.0, 0.2), 4),
         ('kagome (3,0)x(0,2)', L.LatticeCluster('kagome', (3, 0), (0, 2)), lambda c: c.heisenberg(1.0), None),
         ('honeycomb (2,2)', L.LatticeCluster('honeycomb', (2, 2), (-2, 4)), lambda c: c.heisenberg(1.0, 0.2), None),
         ('triangular (1,4)', L.LatticeCluster('triangular', (1, 4), (-4, 5)), lambda c: c.heisenberg(1.0), 10)]
for name, cl, mk, nup in CASES:
    m = mk(cl); nup = nup if nup is not None else cl.N//2
    tabs = cl.irrep_table if hasattr(cl, 'irrep_table') else None
    worst, wflip, n, info = 0.0, 0.0, 0, None
    for k in cl.momenta():
        irs = list(tabs(k)) if tabs else [ir for ir in ('A1', 'A2', 'B1', 'B2', 'E+', 'E-', 'Bx', 'By')
                                          if [o for o in cl.little_group(k) if o in el.IRREPS[ir]] == cl.little_group(k)]
        seen = set()
        for ir in irs:
            perms, chars, ops = cl.group(k, ir)
            key = tuple(round(c.real, 9)+1j*round(c.imag, 9) for c in chars)
            if key in seen:
                continue
            seen.add(key)
            c = edc_k.KEngine(m, nup, perms, chars)
            s = es.SubEngine(m, nup, cl, k, ir, device=DEV)
            info = (s.w, s.nK, s.ncos)
            assert c.dim == s.dim, (name, k, ir, c.dim, s.dim)
            if not c.dim:
                continue
            Ec = c.ground_energy(tol=1e-13)[0]; Es = s.ground_energy(tol=1e-13)[0]
            worst = max(worst, abs(Ec-Es)); n += 1
            if 2*nup == cl.N:
                se, so = es.SubEngine(m, nup, cl, k, ir, 1, device=DEV), es.SubEngine(m, nup, cl, k, ir, -1, device=DEV)
                assert se.dim+so.dim == s.dim, (name, k, ir, se.dim, so.dim, s.dim)
                Ef = min(e.ground_energy(tol=1e-13)[0] for e in (se, so) if e.dim)
                wflip = max(wflip, abs(Ef-Es))
    print(f'{name:20s} N={cl.N:2d} nup={nup} (w, |K|, cosets) = {info}: {n:3d} sectors, max |E_sub - E_k| {worst:.1e}, '
          f'spin flip {wflip:.1e}', flush=True)
