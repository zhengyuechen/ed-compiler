#!/usr/bin/env python3
"""Stored product on one GPU: double vs single precision vectors (compact entries when possible).
Usage: gpu_fp32_bench.py <kind|square> <A1x,A1y,A2x,A2y> <J1[,J2]> <k1/pi,k2/pi> <irrep> <flip> [label]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cupy as cp
import edc_lattices as L, edc_lattice as el, edc_sub
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]; flip = int(sys.argv[6])
label = sys.argv[7] if len(sys.argv) > 7 else kind
if kind == 'square':
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); model = cl.j1j2(*J)
else:
    cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
e = edc_sub.SubEngine(model, cl.N//2, cl, k, ir, flip, device='gpu', stored=True)
out = dict(label=label, N=cl.N, dim=e.dim, real=e.real, nnz=e.stored_nnz, format=e.stored_format, gib=e.stored_bytes/2**30,
           build_seconds=e.stored_build_seconds)
rng = np.random.default_rng(0)
for name, dt in (('fp64', np.float64 if e.real else np.complex128), ('fp32', np.float32 if e.real else np.complex64)):
    x = rng.standard_normal(e.dim) if e.real else rng.standard_normal(e.dim)+1j*rng.standard_normal(e.dim)
    x = cp.asarray(x.astype(dt)); y = cp.empty_like(x); best = math.inf
    for rep in range(6):
        cp.cuda.Device().synchronize(); t = time.perf_counter(); e.matvec(x, y); cp.cuda.Device().synchronize()
        if rep:
            best = min(best, time.perf_counter()-t)
    out[name+'_matvec'] = best; out[name+'_entries_per_second'] = e.stored_nnz/best
print(json.dumps(out), flush=True)
