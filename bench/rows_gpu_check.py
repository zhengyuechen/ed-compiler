#!/usr/bin/env python3
"""Check of the GPU set-up of the rows engine against the CPU set-up on the same node: dimension, non-zeros, the product
on a random vector, the energy; then timings.  Usage: rows_gpu_check.py LxxLy [LxxLy ...]"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cupy as cp, edc_rows
os.environ['EDC_ROWS_GPU_MIN'] = '0'
for t in sys.argv[1:]:
    Lx, Ly = [int(v) for v in t.split('x')]
    t0 = time.perf_counter(); g = edc_rows.RowsEngine(Lx, Ly, device='gpu'); cp.cuda.Device().synchronize(); tg = time.perf_counter()-t0
    out = dict(torus=t, gpu_setup=bool(g.gpu_setup), dim=g.dim, nnz=g.nnz, compact=g.compact, stored_bytes=g.stored_bytes, lanes=g.lanes,
               setup_seconds_gpu_path=tg, product_seconds=g.time_product(20))
    if g.dim <= 400000:
        os.environ['EDC_ROWS_GPU_MIN'] = str(10**18)
        t0 = time.perf_counter(); c = edc_rows.RowsEngine(Lx, Ly, device='gpu'); cp.cuda.Device().synchronize(); tc = time.perf_counter()-t0
        os.environ['EDC_ROWS_GPU_MIN'] = '0'
        x = cp.asarray(np.random.default_rng(3).standard_normal(g.dim)); yg = g.matvec(x); yc = c.matvec(x)
        out.update(setup_seconds_cpu_path=tc, cpu_path_dim=c.dim, cpu_path_nnz=c.nnz, product_difference=float(cp.max(cp.abs(yg-yc))/cp.max(cp.abs(yc))))
    en, products, ts, res = g.ground_energy()
    out.update(energy=en[0], products=products, solve_seconds=ts, residual=res)
    print(json.dumps(out), flush=True)
