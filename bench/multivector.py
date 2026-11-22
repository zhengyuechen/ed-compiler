#!/usr/bin/env python3
"""General test: one stored connectivity applied to K vectors at once, vectors interleaved (x[state][k]), against K
single products.  The connectivity (column, value code) is read once for all K vectors and one gather fetches K numbers.
Case 1: the ring-of-rows sector matrix of an Lx x Ly torus.  Case 2: a synthetic matrix with random columns (n rows, m
entries per row) whose vector does not fit the caches -- the regime of large S_z sectors.
Usage: multivector.py Lx Ly [n m]"""
import sys, os, time, ctypes, subprocess, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc, edc_rows
SRC = r'''
#include <stdint.h>
void spmm(int64_t n, const int64_t *rp, const int32_t *ci, const uint16_t *cd, const double *vt, const double *x, double *y) {
    #pragma omp parallel for schedule(dynamic, 2048)
    for (int64_t i = 0; i < n; i++) {
        double s[K]; for (int k = 0; k < K; k++) s[k] = 0.0;
        for (int64_t e = rp[i]; e < rp[i + 1]; e++) {
            double v = vt[cd[e]]; const double *xc = x + (int64_t)ci[e] * K;
            for (int k = 0; k < K; k++) s[k] += v * xc[k];
        }
        for (int k = 0; k < K; k++) y[i * K + k] = s[k];
    }
}
'''
def kernel(K):
    d = tempfile.mkdtemp(); c = os.path.join(d, 'k.c'); so = os.path.join(d, 'k.dylib' if sys.platform == 'darwin' else 'k.so')
    open(c, 'w').write(SRC); subprocess.check_call(edc.CC+[f'-DK={K}', c, '-o', so])
    lib = ctypes.CDLL(so); lib.spmm.argtypes = [ctypes.c_int64]+[ctypes.c_void_p]*6
    return lib

def bench(label, rp, ci, cd, vt):
    n = len(rp)-1; out = dict(case=label, rows=int(n), entries=int(len(ci)))
    for K in (1, 2, 4, 8):
        lib = kernel(K); x = np.random.default_rng(1).standard_normal(n*K); y = np.zeros(n*K); best = 1e9
        for _ in range(8):
            t0 = time.perf_counter(); lib.spmm(n, rp.ctypes.data, ci.ctypes.data, cd.ctypes.data, vt.ctypes.data, x.ctypes.data, y.ctypes.data)
            best = min(best, time.perf_counter()-t0)
        out[f'K{K}_ms_per_vector'] = round(1e3*best/K, 4)
    out['gain_K8'] = round(out['K1_ms_per_vector']/out['K8_ms_per_vector'], 2)
    print(json.dumps(out), flush=True)

Lx, Ly = int(sys.argv[1]), int(sys.argv[2])
e = edc_rows.RowsEngine(Lx, Ly)
bench(f'rows {Lx}x{Ly}', e._rp, e._ci, e._vals[0], e._vals[1])
if len(sys.argv) > 4:
    n, m = int(sys.argv[3]), int(sys.argv[4]); rng = np.random.default_rng(2)
    rp = np.arange(0, (n+1)*m, m, dtype=np.int64); ci = rng.integers(0, n, n*m, dtype=np.int32); cd = rng.integers(0, 200, n*m).astype(np.uint16)
    bench(f'random columns n={n} m={m}', rp, ci, cd, rng.standard_normal(200))
