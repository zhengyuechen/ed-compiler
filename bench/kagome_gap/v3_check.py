#!/usr/bin/env python3
"""The frozen v3 on this machine against exact diagonalisation, 24-spin YC6 (3 rings), S_z = 0 and 1, K = 0: energy,
<S_z> of every layer and the energy of every row of bonds (profiles.py).  Usage: v3_check.py V3_DIR"""
import sys, os
V3 = sys.argv[1]; sys.path.insert(0, V3); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, edc_dmrg as ed, edc_dmrg_gpu as eg, edc_dmrg_gpu3 as e3
from profiles import profiles, exact_rows
lat = ed.kagome_cylinder(3, 3); model = ed.Model(lat); worst = 0.0
for nup in (12, 13):
    Ex, szx, ex, single = exact_rows(lat, nup); E, eng, rows = eg.run(model, 512, nup=nup, K=0, cls=e3.DMRGGPU3, log=lambda *a: None)
    pr = profiles(eng, 512, nup, E); d1 = max(abs(a-b) for a, b in zip(pr['layer_sz'], szx)); d2 = max(abs(q['E'] - ex[(q['kind'], q['x'])]) for q in pr['rows'])
    worst = max(worst, abs(E-Ex), d1, d2, abs(pr['row_sum_minus_E']), pr['ring_mismatch'])
    print(f'v3, {lat.name}, S_z = {nup-12}: E - exact = {E-Ex:+.1e}; rows: sum - E = {pr["row_sum_minus_E"]:+.1e}, ring mismatch {pr["ring_mismatch"]:.1e}; '
          f'largest difference from exact: layer S_z {d1:.1e}, rows {d2:.1e}', flush=True)
print('v3 check ok' if worst < 1e-4 else 'v3 check FAILED')  # the profiles are first order in the error of the state
