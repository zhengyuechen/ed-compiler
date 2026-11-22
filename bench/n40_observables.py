#!/usr/bin/env python3
"""40-site J1-J2: ground state in Gamma A1 (spin-flip even), replayed eigenvector, M^2(pi,pi) and M^2(pi,0).
Usage: n40_observables.py <J2>"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cupy as cp, edc_lattice as el, edc_gpu as eg, edc_gpu_lean as gl
J2 = float(sys.argv[1]); cl = el.tilted(6, 2)
t0 = time.perf_counter(); g = gl.GPULean(cl.j1j2(1.0, J2), 20, cl, (0.0, 0.0), 'A1', flip=1)
E0, it, tl = g.ground_energy(tol=1e-13)
t1 = time.perf_counter(); psi, E, res = eg.ground_state_vector(g); t2 = time.perf_counter()
out = dict(N=40, J2=J2, dim=g.dim, E0=E0, iterations=it, lanczos_seconds=tl, replay_seconds=t2-t1, replay_energy=E, residual=res)
for q, Qs in (('pi_pi', [(math.pi, math.pi)]), ('pi_0', [(math.pi, 0.0), (0.0, math.pi)])):
    t3 = time.perf_counter(); out['M2_'+q] = gl.m2(g, psi, cl, Qs); out['M2_'+q+'_seconds'] = time.perf_counter()-t3
out['total_seconds'] = time.perf_counter()-t0
print(json.dumps(out), flush=True)
