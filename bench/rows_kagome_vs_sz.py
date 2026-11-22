#!/usr/bin/env python3
"""Kagome strip of 2 x Ly cells: ring-of-rows engine (total spin 0, momentum 0, trimer row basis) against the S_z engine in
the sector of the same torus that holds the same level (momentum 0, its point-group irrep and spin flip), same machine.
Prints dimension, stored bytes, set-up, time per product and time to the level for both.  Usage: rows_kagome_vs_sz.py Ly [cpu|gpu]"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc_lattices as L, edc_hubbard_sym as hs, edc_sub, edc_rows
Ly = int(sys.argv[1]); device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'; N = 6*Ly
def sync():
    if device == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()
t = time.perf_counter(); r = edc_rows.RowsEngine(2, Ly, lattice='kagome', device=device); sync(); t_set = time.perf_counter()-t
t = time.perf_counter(); en, prod, ts, res = r.ground_energy(); sync(); t_sol = time.perf_counter()-t; E0 = float(np.atleast_1d(en)[0])
vb = 8 if r.real else 16
rows = dict(dim=int(r.dim), nnz=int(r.nnz), real=bool(r.real), stored_bytes=int(r.stored_bytes), vector_bytes=int(r.dim*vb), setup_seconds=t_set,
            product_seconds=float(r.time_product(5)), solve_seconds=t_sol, products=int(prod), E0=E0)
cl = L.LatticeCluster('kagome', (2, 0), (0, Ly)); nup = N//2; best = None
for k, ir, fl, pops, mult in hs.sectors(cl, nup, N-nup):
    if abs(k[0]) > 1e-9 or abs(k[1]) > 1e-9:
        continue
    sync(); t = time.perf_counter(); e = edc_sub.SubEngine(cl.heisenberg(1.0), nup, cl, k, ir, fl, device=device, point_ops=pops, stored=True); sync(); ts_ = time.perf_counter()-t
    if e.dim == 0:
        continue
    t = time.perf_counter(); E, it, _ = e.ground_energy(tol=1e-12, nev=3); sync(); tsol = time.perf_counter()-t
    lev = [float(x) for x in np.atleast_1d(E)]
    if min(abs(x-E0) for x in lev) < 1e-8:
        t = time.perf_counter(); E1, it1, _ = e.ground_energy(tol=1e-12); sync(); tsol1 = time.perf_counter()-t
        vb2 = 8 if e.real else 16
        best = dict(irrep=ir, flip=fl, dim=int(e.dim), real=bool(e.real), stored_bytes=int(getattr(e, 'stored_bytes', 0)), vector_bytes=int(e.dim*vb2), setup_seconds=ts_,
                    product_seconds=float(e.time_matvec(5)), solve_seconds=tsol1, products=int(it1), lowest_of_sector=float(np.atleast_1d(E1)[0]), level_index=int(np.argmin([abs(x-E0) for x in lev])))
        break
    del e
out = dict(N=N, device=device, rows=rows, sz=best)
if best:
    out['ratios_sz_over_rows'] = dict(vector=best['vector_bytes']/rows['vector_bytes'], stored=best['stored_bytes']/max(rows['stored_bytes'], 1),
                                      product=best['product_seconds']/rows['product_seconds'], end_to_end=(best['setup_seconds']+best['solve_seconds'])/(rows['setup_seconds']+rows['solve_seconds']))
print(json.dumps(out))
