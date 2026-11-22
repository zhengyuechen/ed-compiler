#!/usr/bin/env python3
"""Benchmark: ring-of-rows engine (edc_rows) against the S_z engine (edc_sub), same torus, same sector
(K = (0,0), mirrors even / A1, lowest singlet), same device.  One JSON line per engine and torus.
Usage: rows_vs_sz.py cpu|gpu LxxLy [LxxLy ...]      e.g.  rows_vs_sz.py gpu 4x8 6x5 6x6"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc_lattice as el, edc_sub, edc_rows
device = sys.argv[1]

def sync():
    if device == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()

def sz(Lx, Ly, pops, label):
    cl = el.Cluster((Lx, 0), (0, Ly)); flip = 1 if (cl.N//2) % 2 == 0 else -1     # the flip sector that contains the singlets
    t0 = time.perf_counter(); e = edc_sub.SubEngine(cl.j1j2(1.0), cl.N//2, cl, (0.0, 0.0), 'A1', flip, device=device, point_ops=pops)
    sync(); t_setup = time.perf_counter()-t0
    xp = np
    if device == 'gpu':
        import cupy as xp
    x = xp.asarray(np.random.default_rng(1).standard_normal(e.dim)); y = xp.zeros(e.dim); e.matvec(x, y); sync(); best = 1e9
    for _ in range(20):
        y[:] = 0; sync(); t0 = time.perf_counter(); e.matvec(x, y); sync(); best = min(best, time.perf_counter()-t0)
    del x, y
    # levels needed to reach the lowest singlet of this sector: one where it is the lowest state, three on 6x5
    nev = {(6, 5): 3}.get((Lx, Ly), 1)
    t0 = time.perf_counter(); r = e.ground_energy(tol=1e-12, nev=nev); sync(); t_solve = time.perf_counter()-t0
    r = (r[0] if nev > 1 else [r[0]], r[1])
    print(json.dumps(dict(engine='sz', group=label, torus=f'{Lx}x{Ly}', device=device, flip=flip, dim=int(e.dim), entries=int(getattr(e, 'stored_nnz', 0) or 0),
                          stored_bytes=int(getattr(e, 'stored_bytes', 0) or 0), product_seconds=best, setup_seconds=t_setup, solve_seconds=t_solve,
                          products=int(r[1]), levels=[float(v) for v in r[0]])), flush=True)

def rows(Lx, Ly):
    e = edc_rows.RowsEngine(Lx, Ly, device=device); sync()
    best = e.time_product(20)
    en, products, t_solve, res = e.ground_energy(nev=1, tol=1e-12); sync()
    print(json.dumps(dict(engine='rows', torus=f'{Lx}x{Ly}', device=device, dim=e.dim, entries=e.nnz, stored_bytes=e.stored_bytes, real=e.real,
                          product_seconds=best, setup_seconds=e.setup_seconds, solve_seconds=t_solve, products=products, residual=res, levels=en)), flush=True)

for t in sys.argv[2:]:
    Lx, Ly = [int(v) for v in t.split('x')]
    rows(Lx, Ly); rows(Lx, Ly)                                  # second line: warm caches (compiled kernels, local patterns)
    sz(Lx, Ly, ['E', 'C2', 'Mx', 'My'], 'same group (translations x D2)'); sz(Lx, Ly, ['E', 'C2', 'Mx', 'My'], 'same group (translations x D2)')
    if Lx == Ly:
        sz(Lx, Ly, None, 'full group (with C4)'); sz(Lx, Ly, None, 'full group (with C4)')
