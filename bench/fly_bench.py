#!/usr/bin/env python3
"""Time of one product for single blocks: on-the-fly single vector, on-the-fly K vectors per pass, stored (if it fits).
Usage: fly_bench.py <kind> <A1x,A1y,A2x,A2y> <J1[,J2]> <nup:block> [<nup:block> ...]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cupy as cp
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub

kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J); N = cl.N
sync = lambda: cp.cuda.Device().synchronize()
for item in sys.argv[4:]:
    nup, bi = (int(v) for v in item.split(':'))
    k, ir, fl, pops, mult = hs.sectors(cl, nup, N-nup)[bi]
    out = dict(J=J, nup=nup, block=bi, irrep=ir, flip=fl, mult=mult)
    t0 = time.perf_counter()
    e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device='gpu', point_ops=pops, stored=0)
    sync(); out.update(dim=e.dim, real=e.real, group_order=e.group_order, setup_fly=time.perf_counter()-t0)
    out['fly_single'] = e.time_matvec(3)
    dt = np.float64 if e.real else np.complex128
    for K in (2, 4):
        try:
            ap = e.batched(K)
            V = cp.ones((e.dim, K), dt); U = cp.zeros_like(V)
            ap(V, U); sync(); best = 1e9
            for _ in range(2):
                t = time.perf_counter(); ap(V, U); sync(); best = min(best, time.perf_counter()-t)
            out[f'fly_batched_{K}'] = best
            del V, U, ap
        except Exception as ex:
            out[f'fly_batched_{K}'] = repr(ex)[:80]
        cp.get_default_memory_pool().free_all_blocks()
    del e; cp.get_default_memory_pool().free_all_blocks()
    t0 = time.perf_counter()
    e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device='gpu', point_ops=pops)
    sync(); st = getattr(e, '_st', None) is not None
    out.update(stored=st, setup_default=time.perf_counter()-t0, stored_gib=getattr(e, 'stored_bytes', 0)/2**30)
    if st:
        out['stored_single'] = e.time_matvec(3)
    print(json.dumps(out), flush=True)
    del e; cp.get_default_memory_pool().free_all_blocks()
