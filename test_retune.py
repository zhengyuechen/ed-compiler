#!/usr/bin/env python3
"""SubEngine.retune: new couplings on the same bonds, with and without stored connectivity, against a freshly built
engine; every sector of two clusters, three coupling sets.  Usage: test_retune.py [cpu|gpu]"""
import sys
import numpy as np
import edc_lattices as L, edc_lattice as el, edc_hubbard_sym as hs, edc_sub

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'


def check(name, cl, make, Js):
    N = cl.N; nup = N//2; worst = 0.0; n = 0
    for k, ir, fl, pops, mult in hs.sectors(cl, nup, N-nup):
        for stored in (True, False):
            e = edc_sub.SubEngine(make(*Js[0]), nup, cl, k, ir, fl, device=DEVICE, point_ops=pops, stored=stored)
            if e.dim == 0:
                break
            for J in Js[1:]:
                e.retune(make(*J))
                ref = edc_sub.SubEngine(make(*J), nup, cl, k, ir, fl, device=DEVICE, point_ops=pops, stored=False)
                E1 = e.ground_energy(tol=1e-13, nev=2)[0]; E0 = ref.ground_energy(tol=1e-13, nev=2)[0]
                d = max(abs(a-b) for a, b in zip(E1, E0)) if isinstance(E0, list) else abs(E1-E0)
                worst = max(worst, d); n += 1
                assert d < 1e-10, (name, k, ir, fl, stored, J, E1, E0)
    print(f'{name}: {n} retunes (stored and on the fly), max |E - fresh engine| = {worst:.1e}', flush=True)


if __name__ == '__main__':
    sq = el.Cluster((4, 0), (0, 4))
    check('square 4x4 J1-J2', sq, lambda J1, J2: sq.j1j2(J1, J2), [(1.0, 0.3), (1.0, 0.6), (0.7, 0.1)])
    kg = L.LatticeCluster('kagome', (3, 0), (0, 2))
    check('kagome 18 J1-J2', kg, lambda J1, J2: kg.heisenberg(J1, J2), [(1.0, 0.2), (1.0, -0.4), (0.5, 0.9)])
    print('all passed')
