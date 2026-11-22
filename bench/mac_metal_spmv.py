#!/usr/bin/env python3
"""Stored-connectivity product on the Mac GPU (Metal, through MLX's metal_kernel), single precision, compact 6-byte
entries, real sectors.  The connectivity is built by the CPU engine (edc_sub) and copied to the GPU; the Metal kernel
is the same loop as the CUDA spmv32 (one thread per row; Metal has no double, so the row sum is accumulated in float).
Also times the CPU product (OpenMP) on the same data and checks the GPU result against it.
Usage: mac_metal_spmv.py <kind|square> <A1x,A1y,A2x,A2y> <J1[,J2]> <k1/pi,k2/pi> <irrep> <flip> [label]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import edc_lattices as L, edc_lattice as el, edc_sub

kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]; flip = int(sys.argv[6])
label = sys.argv[7] if len(sys.argv) > 7 else kind
if kind == 'square':
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); model = cl.j1j2(*J)
else:
    cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
t0 = time.perf_counter()
e = edc_sub.SubEngine(model, cl.N//2, cl, k, ir, flip, device='cpu', stored=True, precision='fp32')
build = time.perf_counter()-t0
st = e._st
assert e.real and st['compact'] and e.stored_nnz < 2**32
n = e.dim
x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
# CPU product (fp32 vectors, double accumulation), best of 3
y = np.zeros(n, np.float32); tc = math.inf
for _ in range(4):
    t = time.perf_counter(); e.matvec(x, y); tc = min(tc, time.perf_counter()-t)
yref = y.copy()

src = '''
    uint i = thread_position_in_grid.x;
    float acc = 0.0f;
    uint e1 = rp[i + 1];
    for (uint e = rp[i]; e < e1; e++) { acc += vt[mm[e]] * x[jj[e]]; }
    out[i] = dg[i] * x[i] + acc * ist[i];
'''
kern = mx.fast.metal_kernel(name='edc_spmv32', input_names=['x', 'rp', 'jj', 'mm', 'vt', 'dg', 'ist'], output_names=['out'], source=src)
dev = dict(rp=mx.array(st['RP'].astype(np.uint32)), jj=mx.array(st['JJ']), mm=mx.array(st['MM']),
           vt=mx.array(st['VT'].astype(np.float32)), dg=mx.array(st['DG'].astype(np.float32)),
           ist=mx.array((1.0/np.sqrt(st['STR'].astype(np.float64))).astype(np.float32)))
mx.eval(*dev.values())
xg = mx.array(x)
def product(v):
    return kern(inputs=[v, dev['rp'], dev['jj'], dev['mm'], dev['vt'], dev['dg'], dev['ist']],
                grid=(n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
yg = product(xg); mx.eval(yg)
err = float(np.max(np.abs(np.array(yg)-yref))/np.max(np.abs(yref)))
tg = math.inf
for _ in range(10):
    t = time.perf_counter(); yg = product(xg); mx.eval(yg); tg = min(tg, time.perf_counter()-t)
# chained products (y feeds the next product), as in a Lanczos run
t = time.perf_counter(); v = xg
for _ in range(20):
    v = product(v); v = v*(1.0/mx.sqrt(mx.sum(v*v)))
mx.eval(v); tchain = (time.perf_counter()-t)/20
print(json.dumps(dict(label=label, N=cl.N, dim=n, nnz=e.stored_nnz, gib=e.stored_bytes/2**30, build_seconds=build,
                      cpu_threads=os.environ.get('OMP_NUM_THREADS'), cpu_matvec=tc, metal_matvec=tg, metal_chained=tchain,
                      metal_entries_per_second=e.stored_nnz/tg, cpu_entries_per_second=e.stored_nnz/tc,
                      relative_difference_gpu_cpu=err)), flush=True)
