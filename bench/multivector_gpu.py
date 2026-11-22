#!/usr/bin/env python3
"""GPU version of bench/multivector.py: one stored connectivity applied to K interleaved vectors (8 lanes per row).
Cases: the ring-of-rows sector matrix of an Lx x Ly torus, and a synthetic matrix with random columns (n rows, m per row).
Usage: multivector_gpu.py Lx Ly n m"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cupy as cp, edc_rows
SRC = r'''
#define W 8
extern "C" __global__ void spmm(const long long n, const long long *rp, const int *ci, const unsigned short *cd, const double *vt,
                                const double *x, double *y) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, i = tid / W;
    int lane = (int)(tid % W);
    double s[K];
    for (int k = 0; k < K; k++) s[k] = 0.0;
    if (i < n) for (long long e = rp[i] + lane; e < rp[i + 1]; e += W) {
        double v = vt[cd[e]]; const double *xc = x + (long long)ci[e] * K;
        for (int k = 0; k < K; k++) s[k] += v * xc[k];
    }
    for (int k = 0; k < K; k++) for (int off = W / 2; off > 0; off >>= 1) s[k] += __shfl_down_sync(0xffffffff, s[k], off, W);
    if (lane == 0 && i < n) for (int k = 0; k < K; k++) y[i * K + k] = s[k];
}
'''
def bench(label, rp, ci, cd, vt):
    n = len(rp)-1; out = dict(case=label, rows=int(n), entries=int(len(ci))); ref = None
    for K in (1, 2, 4, 8, 16):
        k = cp.RawKernel(SRC, 'spmm', options=(f'-DK={K}',)); x = cp.asarray(np.random.default_rng(1).standard_normal(n*K)); y = cp.zeros(n*K); best = 1e9
        for _ in range(8):
            cp.cuda.Device().synchronize(); t0 = time.perf_counter(); k(((n*8+255)//256,), (256,), (np.int64(n), rp, ci, cd, vt, x, y))
            cp.cuda.Device().synchronize(); best = min(best, time.perf_counter()-t0)
        if K == 1:
            ref = (x.copy(), y.copy())
        elif K == 2:                                             # same numbers as the single product on the first vector
            x1 = cp.zeros(n*2); x1[0::2] = ref[0]; y1 = cp.zeros(n*2); k(((n*8+255)//256,), (256,), (np.int64(n), rp, ci, cd, vt, x1, y1))
            out['check'] = float(cp.max(cp.abs(y1[0::2]-ref[1])))
        out[f'K{K}_ms_per_vector'] = round(1e3*best/K, 5)
        del x, y
    out['gain_K8'] = round(out['K1_ms_per_vector']/out['K8_ms_per_vector'], 2); out['gain_K16'] = round(out['K1_ms_per_vector']/out['K16_ms_per_vector'], 2)
    print(json.dumps(out), flush=True)

Lx, Ly, n, m = [int(v) for v in sys.argv[1:5]]
e = edc_rows.RowsEngine(Lx, Ly, device='gpu')
bench(f'rows {Lx}x{Ly}', e._rp, e._ci, e._vals[0], e._vals[1]); del e
rng = cp.random.default_rng(2)
rp = cp.arange(0, (n+1)*m, m, dtype=cp.int64); ci = rng.integers(0, n, n*m, dtype=cp.int32); cd = rng.integers(0, 200, n*m, dtype=cp.int32).astype(cp.uint16)
bench(f'random columns n={n} m={m}', rp, ci, cd, cp.asarray(np.random.default_rng(3).standard_normal(200)))
