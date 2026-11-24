#!/usr/bin/env python3
"""v3's product (edc_dmrg_gpu.Product) against the grouped-GEMM product (edc_dmrg_gpu3g.ProductG) on the same saved
input, layer by layer, one GPU.

    gemm_check.py state.npz pass.npz out.json

For every layer of the pass file (levels.py): both products of the same engine state applied to v = C_r / |C_r|;
relative difference of the results; minimum and median time of each (alternating, after one untimed call each); and the
time split (copy, slab kernels, matrix products, layouts) of each."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, cupy as cp
import levels, edc_dmrg_gpu as eg, edc_dmrg_gpu3g as eg3g

state, passf, outf = sys.argv[1:4]
R = levels.Replay('gpu3', state, passf); rows = []
print(f'{R.lat.name}, D = {R.head["D"]}, nup = {R.head["nup"]}; grouped call available: {eg3g._cublas() is not None}', flush=True)
R.eng.solve(0, 3e-5, maxit=2); R.restore(0)                                 # kernels compiled before any clock
for r in range(R.n):
    R.at(r); eng = R.eng; v = eng.M[r]/cp.linalg.norm(eng.M[r]); pa, pb = {}, {}
    A = eg.Product(eng, r, pa); B = eg3g.ProductG(eng, r, pb)
    wa = A.apply(v); wb = B.apply(v); cp.cuda.Device().synchronize()
    d = float(cp.linalg.norm(wb - wa)/cp.linalg.norm(wa))
    for p in (pa, pb):
        p.clear()
    ta, tb = [], []; t0 = time.perf_counter()
    while len(ta) < 50 and (len(ta) < 5 or time.perf_counter() - t0 < 1.0):
        t = time.perf_counter(); A.apply(v); cp.cuda.Device().synchronize(); ta.append(time.perf_counter() - t)
        t = time.perf_counter(); B.apply(v); cp.cuda.Device().synchronize(); tb.append(time.perf_counter() - t)
    k = len(ta); w = levels.work(R.model, eng.plan[r], r)
    row = dict(layer=r, size=int(A.n), flops=w['flops'], labels_left=w['labels_left'], labels_right=w['labels_right'], problems=int(B.gL1.n + B.gL2.n + B.gR1.n + B.gR2.n) if B.grouped else None,
               relative_difference=d, v3_min=min(ta), v3_median=float(np.median(ta)), g_min=min(tb), g_median=float(np.median(tb)), reps=k,
               v3_parts={q: x/k for q, x in pa.items()}, g_parts={q: x/k for q, x in pb.items()})
    rows.append(row)
    print(f'  layer {r:2d}: tensor {A.n:9d}, {row["problems"]} products | v3 {1e3*row["v3_min"]:7.3f} ms, grouped {1e3*row["g_min"]:7.3f} ms ({row["v3_min"]/row["g_min"]:4.2f}x) | '
          f'matrix products {1e3*row["v3_parts"].get("gemm", 0):6.3f} -> {1e3*row["g_parts"].get("gemm", 0):6.3f} ms | difference {d:.1e}', flush=True)
    del A, B
sa = sum(x['v3_min'] for x in rows); sb = sum(x['g_min'] for x in rows); fl = sum(x['flops'] for x in rows)
print(f'one product of every layer: v3 {1e3*sa:.2f} ms ({fl/sa/1e12:.2f} TFLOP/s), grouped {1e3*sb:.2f} ms ({fl/sb/1e12:.2f} TFLOP/s), {sa/sb:.2f}x; '
      f'largest difference {max(x["relative_difference"] for x in rows):.1e}', flush=True)
json.dump(dict(problem=R.head, device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), layers=rows, v3_sum=sa, grouped_sum=sb), open(outf, 'w'), indent=1)
