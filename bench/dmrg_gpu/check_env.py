#!/usr/bin/env python3
"""Environments grown on the device (edc_dmrg_gpu2) against the host library, on the same tensors.
A host engine is swept twice; then every right environment RH[r], GR[r] is regrown on the device from the same
M[r], RH[r+1], GR[r+1] and compared; after a half sweep to the right every left environment LH[r+1], GL[r+1] is
regrown from M[r], LH[r], GL[r] and compared.   check_env.py [Lx c D tail]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, cupy as cp, edc_dmrg as ed, edc_dmrg_gpu as eg, edc_dmrg_gpu2 as e2

Lx, c, D = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (3, 3, 200); tail = len(sys.argv) > 4 and sys.argv[4] == 'tail'
lat = ed.kagome_cylinder(Lx, c, tail=tail); model = ed.Model(lat); worst = 0.0
for nup in (lat.N//2, lat.N//2+1):
    eng = eg.DMRGGPU(model, nup, 0); eng.host = True
    for Dw in (D//4, D//2, D):
        eng.sweep(Dw, 1e-2, 1e-6)
    rel = lambda a, b: float(np.abs(cp.asnumpy(a)[:b.size] - b).max()/max(1e-300, np.abs(b).max())) if b.size > 1 else float(np.abs(cp.asnumpy(a)[:1] - b).max())
    for r in range(eng.n-1, 0, -1):
        RH, GR = e2.grow_right_dev(model, eng.tables, r, eng.plan[r], cp.asarray(eng.M[r]), cp.asarray(eng.RH[r+1]), cp.asarray(eng.GR[r+1]))
        d = (rel(RH, eng.RH[r]), rel(GR, eng.GR[r])); worst = max(worst, *d)
        print(f'nup={nup} right environment of layer {r:2d}: RH {d[0]:.1e}, GR {d[1]:.1e} (relative to the largest entry; sizes {eng.RH[r].size}, {eng.GR[r].size})', flush=True)
    for r in range(eng.n-1):
        eng.solve(r, 1e-6); eng.move_right(r, D, 0.0)
        LH, GL = e2.grow_left_dev(model, eng.tables, r, eng.plan[r], cp.asarray(eng.M[r]), cp.asarray(eng.LH[r]), cp.asarray(eng.GL[r]))
        d = (rel(LH, eng.LH[r+1]), rel(GL, eng.GL[r+1])); worst = max(worst, *d)
        print(f'nup={nup} left environment of layer {r+1:2d}: LH {d[0]:.1e}, GL {d[1]:.1e} (sizes {eng.LH[r+1].size}, {eng.GL[r+1].size})', flush=True)
print(f'{lat.name}, D = {D}: largest relative difference device - host over all environments {worst:.1e}')
assert worst < 1e-11
