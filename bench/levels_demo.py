#!/usr/bin/env python3
"""The lowest levels of one block with the error-bound stopping rule (edc_levels.lowest_levels), with the connectivity
stored as far as it fits (stored='partial') or not at all.  For a direct comparison with a record of bench/levels_run.py
(same block index of edc_hubbard_sym.sectors, same couplings).
Usage: levels_demo.py <kind> <A1x,A1y,A2x,A2y> <nup> <block> <J1,J2> <nev> [gpu|cpu] [modes: partial,fly,stored]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub, edc_levels

kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; nup = int(sys.argv[3]); block = int(sys.argv[4])
J = [float(v) for v in sys.argv[5].split(',')]; nev = int(sys.argv[6]); device = sys.argv[7] if len(sys.argv) > 7 else 'gpu'
modes = sys.argv[8].split(',') if len(sys.argv) > 8 else ['partial', 'fly']
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
k, ir, fl, pops, mult = hs.sectors(cl, nup, cl.N-nup)[block]


def sync():
    if device == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()


for mode in modes:
    t0 = time.perf_counter()
    e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops,
                          stored={'partial': 'partial', 'fly': False, 'stored': True}[mode], reserve_vectors=4)
    sync(); t1 = time.perf_counter()
    lev, steps, conv, info = edc_levels.lowest_levels(e, nev)
    sync(); t2 = time.perf_counter()
    print(json.dumps(dict(mode=mode, kind=kind, N=cl.N, nup=nup, block=block, k_over_pi=[v/math.pi for v in k], irrep=ir, flip=fl, J=J, dim=int(e.dim),
                          real=bool(e.real), rows_stored=int(getattr(e, 'stored_rows', 0)) if getattr(e, '_st', None) is not None else 0,
                          stored_gib=(getattr(e, 'stored_bytes', 0)/2**30 if getattr(e, '_st', None) is not None else 0.0),
                          setup_seconds=t1-t0, solve_seconds=t2-t1, steps=steps, converged=conv, E=lev, residuals=info['residuals'])), flush=True)
    del e
    if device == 'gpu':
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
