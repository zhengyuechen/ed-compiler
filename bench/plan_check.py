#!/usr/bin/env python3
"""The planner on a few sectors: what it chooses and what that gains over the default settings; then a ground-state run with
and without the plan, split into products and the rest.  Usage: plan_check.py [cpu|gpu] [cases: e.g. 4x8,6x6,4x8c]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc_lattice as el, edc_sub, edc_plan
device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
cases = (sys.argv[2] if len(sys.argv) > 2 else '4x6,4x8,4x8c').split(',')
for c in cases:
    cplx = c.endswith('c'); Lx, Ly = (int(v) for v in c.rstrip('c').split('x')); cl = el.Cluster((Lx, 0), (0, Ly))
    k, flip = ((2*math.pi/Lx, 0.0), 0) if cplx else ((0.0, 0.0), 1 if (cl.N//2) % 2 == 0 else -1)
    e = edc_sub.SubEngine(cl.j1j2(1.0, 0.5), cl.N//2, cl, k, 'A1', flip, device=device, stored=True)
    if device == 'cpu':
        d0 = edc_plan.get_threads()
    with edc_plan.Profile(e) as pr0:
        E0 = e.ground_energy(tol=1e-12)
    t = time.perf_counter(); p = edc_plan.plan(e, vectors=8, cache=False); tp = time.perf_counter()-t
    with edc_plan.Profile(e) as pr1:
        E1 = e.ground_energy(tol=1e-12)
    out = dict(case=c, device=device, dim=int(e.dim), real=bool(e.real), machine=p['machine'], planning_seconds=tp,
               chosen={k2: p[k2] for k2 in ('threads', 'block', 'K') if k2 in p}, default_seconds=p['default_seconds'],
               planned_seconds_per_vector=p['seconds_per_vector'], gain=p['gain'], table=p['table'],
               solve_default=pr0.report(), solve_planned=pr1.report(), energy_difference=abs(float(E0[0])-float(E1[0])))
    if device == 'cpu':
        out['default_threads'] = d0; edc_plan.set_threads(d0)
    print(json.dumps(out), flush=True)
    del e
