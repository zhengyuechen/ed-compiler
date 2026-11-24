#!/usr/bin/env python3
"""su2_dmrg against exact energies on cylinders that the bonds hold whole (D large): lowest level of total spin S and
momentum K.  Exact references: lowest S_z = S level of each K from edc_dmrg.exact_levels (the lowest S_z = S level has
spin S when it lies below every S_z = S + 1 level of that K, which is checked), and the singlet levels of
su2_layer (10 and 20 spins)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed, su2_dmrg as sd

cases = [((2, 3, True), [(0, 0), (0, 1), (2, 0), (2, 1)]),          # 18 spins
         ((2, 3, False), [(1, 0), (1, 1), (3, 0)]),                    # 15 spins, S = 1/2 and 3/2
         ((2, 4, False), [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)])]   # 20 spins, YC8
worst = 0.0
for (Lx, c, tail), targets in cases:
    lat = ed.kagome_cylinder(Lx, c, tail=tail); N = lat.N; model = sd.SU2Model(lat); ex = {}
    for S2, K in targets:
        nup = (N + S2)//2
        if nup not in ex:
            ex[nup] = ed.exact_levels(lat.global_bonds(), N, nup, 1.0, tperm=lat.translation(), c=c)
        if nup + 1 not in ex and nup + 1 <= N:
            ex[nup + 1] = ed.exact_levels(lat.global_bonds(), N, nup + 1, 1.0, tperm=lat.translation(), c=c)
        e_ex = ex[nup][K]; above = ex.get(nup + 1, {}).get(K, np.inf); ok = e_ex < above - 1e-9
        t0 = time.perf_counter(); E, eng, rows = sd.ground_state(model, 400, S2, K)
        d = E - e_ex; worst = max(worst, abs(d)) if ok else worst
        print(f'{lat.name}: N = {N}, S = {S2/2:g}, K = {K}: su2 {E:.10f} | exact lowest S_z = S level {e_ex:.10f} '
              f'({"spin S" if ok else "NOT shown to be spin S"}) | difference {d:+.1e} | largest bond {max(eng.bond_multiplets())} multiplets | {time.perf_counter()-t0:.1f} s', flush=True)
print(f'largest difference {worst:.1e}')
