#!/usr/bin/env python3
"""Time against memory with partly stored connectivity (SubEngine(stored='partial')): one product of one double-precision
vector with a fraction 0, 1/4, 1/2, 3/4, 1 of the full store (a negative fraction: as many rows as fit on the device).
Usage: partial_bench.py <kind> <A1x,A1y,A2x,A2y> <nup> <k1/pi,k2/pi> <irrep> <flip> [cpu|gpu] [fractions] [J1,J2]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_sub

kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; nup = int(sys.argv[3])
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]; flip = int(sys.argv[6])
device = sys.argv[7] if len(sys.argv) > 7 else 'gpu'
fracs = [float(v) for v in sys.argv[8].split(',')] if len(sys.argv) > 8 else [0.0, 0.25, 0.5, 0.75, 1.0]
J = [float(v) for v in sys.argv[9].split(',')] if len(sys.argv) > 9 else [1.0]
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
ref = None
for f in fracs:
    if f > 0 and f < 1:
        os.environ['EDC_STORED_FRACTION'] = str(f)
    t0 = time.perf_counter()
    e = edc_sub.SubEngine(model, nup, cl, k, ir, flip, device=device, stored=(False if f == 0 else True if f == 1 else 'partial'))   # f < 0: what fits
    os.environ.pop('EDC_STORED_FRACTION', None)
    xp = e.xp; n = e.dim; rng = np.random.default_rng(0)
    x = rng.standard_normal(n) if e.real else rng.standard_normal(n)+1j*rng.standard_normal(n)
    x = xp.asarray(x); y = xp.empty_like(x); best = math.inf

    def sync():
        if device == 'gpu':
            import cupy as cp
            cp.cuda.Device().synchronize()
    for rep in range(5):
        sync(); t = time.perf_counter(); e.matvec(x, y); sync()
        if rep:
            best = min(best, time.perf_counter()-t)
    chk = float(abs(xp.vdot(x, y)))
    ref = chk if ref is None else ref
    print(json.dumps(dict(kind=kind, N=cl.N, J=J, dim=n, real=bool(e.real), fraction=f, rows_stored=int(getattr(e, 'stored_rows', 0)) if f else 0,
                          stored_gib=(getattr(e, 'stored_bytes', 0)/2**30 if f else 0.0), matvec_seconds=best, setup_seconds=time.perf_counter()-t0,
                          check=abs(chk-ref)/abs(ref))), flush=True)
    del e, x, y
    if device == 'gpu':
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
