#!/usr/bin/env python3
"""Time one product with a stored ring-of-rows sector matrix (real CSR saved beforehand, env SAVE) in a
minimal C/OpenMP kernel compiled like the project's kernels, for a clean reading at a fixed thread count.
Usage: rows_spmv.py matrix.npz [threads ...]"""
import sys, os, time, ctypes, subprocess, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc
SRC = r'''
#include <omp.h>
#include <stdint.h>
void spmv(int64_t n, const int64_t *rp, const int32_t *ci, const double *va, const double *x, double *y, int threads) {
  #pragma omp parallel for schedule(dynamic, 4096) num_threads(threads)
  for (int64_t i = 0; i < n; i++) {
    double s = 0.0;
    for (int64_t e = rp[i]; e < rp[i + 1]; e++) s += va[e] * x[ci[e]];
    y[i] = s;
  }
}
'''
z = np.load(sys.argv[1]); n = int(z['n']); va = np.ascontiguousarray(z['data'], np.float64)
ci = np.ascontiguousarray(z['indices'], np.int32); rp = np.ascontiguousarray(z['indptr'], np.int64)
d = tempfile.mkdtemp(); c = os.path.join(d, 'k.c'); so = os.path.join(d, 'k.dylib' if sys.platform == 'darwin' else 'k.so')
open(c, 'w').write(SRC); subprocess.check_call(edc.CC+[c, '-o', so])
lib = ctypes.CDLL(so); P = ctypes.c_void_p
lib.spmv.argtypes = [ctypes.c_int64, P, P, P, P, P, ctypes.c_int]
x = np.random.default_rng(1).standard_normal(n); y = np.zeros(n)
out = dict(n=n, nnz=int(len(va)), bytes=int(va.nbytes+ci.nbytes+rp.nbytes))
for t in [int(v) for v in sys.argv[2:]] or [1, 8]:
    best = 1e9
    for rep in range(40):
        t0 = time.perf_counter(); lib.spmv(n, rp.ctypes.data, ci.ctypes.data, va.ctypes.data, x.ctypes.data, y.ctypes.data, t)
        best = min(best, time.perf_counter()-t0)
    out[f'ms_{t}_threads'] = round(best*1e3, 3)
import scipy.sparse as sp
A = sp.csr_matrix((va, ci, rp), shape=(n, n)); out['check'] = float(np.max(np.abs(A @ x-y)))
print(json.dumps(out))
