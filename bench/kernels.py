#!/usr/bin/env python3
"""Time per matrix-vector product for several generated kernels, batch sizes K and thread counts.
Usage: kernels.py <model> <kernels comma> <K comma> <threads comma>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, edc, models
name = sys.argv[1]; kernels = sys.argv[2].split(','); Ks = [int(k) for k in sys.argv[3].split(',')]
threads = [int(t) for t in sys.argv[4].split(',')]
m, g = models.build(name)
for K in Ks:
    e = edc.Engine(m, m.n//2, group=g, K=K)
    x = np.random.default_rng(0).standard_normal(e.dim*K); y = np.empty_like(x)
    if 'stored' in kernels:
        e.build_stored()
    for kern in kernels:
        for nt in threads:
            e.threads(nt); e.matvec(x, y, kernel=kern)
            best = min((lambda t0: (e.matvec(x, y, kernel=kern), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(3))
            print(json.dumps(dict(model=name, dim=e.dim, kernel=kern, K=K, threads=nt, seconds=best, seconds_per_vector=best/K)), flush=True)
    del e, x, y
