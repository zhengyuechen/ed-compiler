#!/usr/bin/env python3
"""Parameter scan J1-J2 ring: K values of J2 solved together vs one after another.  Usage: family_scan.py <L> <K> <threads>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc, edc_family
L, K, nt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lams = np.linspace(0.0, 0.7, K)
A = edc.ring(L); B = edc.SpinHalfModel(L)
for i in range(L):
    B.heisenberg(i, (i+2) % L, 1.0)
g = edc.ring_translations(L)
t0 = time.perf_counter(); f = edc_family.FamilyEngine(A, B, lams, L//2, group=g, threads=nt); setup_f = time.perf_counter()-t0
Ef, its, tf = f.ground_energies()
t0 = time.perf_counter()
Es, iters = [], []
e = None
for lam in lams:
    e = edc.Engine(edc.ring(L, 1.0, float(lam)), L//2, group=g, threads=nt)
    E, it, _ = e.ground_energy(); Es.append(E); iters.append(it)
ts = time.perf_counter()-t0
print(json.dumps(dict(L=L, K=K, threads=nt, dim=f.dim, J2=lams.tolist(), E_family=Ef.tolist(), E_separate=Es,
                      max_diff=float(np.max(np.abs(Ef-np.array(Es)))), iterations_family=its.tolist(), iterations_separate=iters,
                      family_setup_seconds=setup_f, family_lanczos_seconds=tf, family_total_seconds=setup_f+tf,
                      separate_total_seconds=ts)))
