#!/usr/bin/env python3
"""edc_levels.lowest_levels against existing records of bench/levels_run.py (read only): the same blocks are solved
again with the error-bound stopping rule and compared level by level.
Usage: levels_compare.py <records.jsonl[,more.jsonl]> <nup> [gpu|cpu] [max dim]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub, edc_levels

paths = sys.argv[1].split(','); nup = int(sys.argv[2]); device = sys.argv[3] if len(sys.argv) > 3 else 'gpu'
maxdim = float(sys.argv[4]) if len(sys.argv) > 4 else 1e18
recs = []
for p in paths:
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('nup') == nup and r.get('subgroup') == 'rotations' and r.get('converged') and len(r.get('E', [])) > 1 and r['dim'] <= maxdim:
            recs.append(r)
tot_old = tot_new = 0; worst = 0.0; t_old = t_new = 0.0
for r in sorted(recs, key=lambda r: (tuple(r['J']), r['block'])):
    a = r['vectors']; cl = L.LatticeCluster(r['kind'], (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*r['J'])
    k, ir, fl, pops, mult = hs.sectors(cl, nup, cl.N-nup)[r['block']]
    assert ir == r['irrep'] and fl == r['flip'] and max(abs(x/math.pi-y) for x, y in zip(k, r['k_over_pi'])) < 1e-9
    e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops, stored='partial', reserve_vectors=4)
    assert e.dim == r['dim']
    t0 = time.perf_counter(); lev, steps, conv, info = edc_levels.lowest_levels(e, len(r['E'])); dt = time.perf_counter()-t0
    d = max(abs(p-q) for p, q in zip(lev, r['E'])) if conv and len(lev) == len(r['E']) else float('nan')
    tot_old += r['iterations']; tot_new += steps; worst = max(worst, d if d == d else 9.9); t_old += r['solve_seconds']; t_new += dt
    print(json.dumps(dict(J=r['J'], nup=nup, block=r['block'], dim=r['dim'], steps_record=r['iterations'], steps=steps, seconds_record=r['solve_seconds'],
                          seconds=dt, stored_record=r.get('stored'), rows_stored=int(getattr(e, 'stored_rows', 0)), max_diff=d, converged=conv)), flush=True)
    del e
    if device == 'gpu':
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
print(json.dumps(dict(summary=True, nup=nup, blocks=len(recs), steps_record=tot_old, steps=tot_new, seconds_record=t_old, seconds=t_new, worst=worst)))
