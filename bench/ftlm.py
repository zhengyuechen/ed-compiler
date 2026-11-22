#!/usr/bin/env python3
"""Finite-temperature Lanczos (FTLM) over every symmetry block of one S_z sector, the computation of
arXiv:2607.17429 Table 3: R random vectors per block, N_L Lanczos steps each.  Blocks from edc_hubbard_sym.sectors
(complete, with multiplicities), engine edc_sub.SubEngine on one device.
  mode 'batched': the R chains run together; one pass computes every matrix element once for all R vectors.
  mode 'single' : the R chains run one after another with the ordinary product.
Recurrence (both modes, per chain): u <- u + H v; alpha = <v|u>; u <- u - alpha v; beta = |u|; (v, u) <- (u/beta, -beta v).
Usage: ftlm.py <kind> <A1x,A1y,A2x,A2y> <J1[,J2,..]> <R> <NL> <mode> [part/nparts] [device]
Prints one JSON line per block (alphas, betas per chain; timings)."""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub

kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
R, NL, mode = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
part, nparts = (int(v) for v in (sys.argv[7] if len(sys.argv) > 7 else '0/1').split('/'))
device = sys.argv[8] if len(sys.argv) > 8 else 'gpu'
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J); nup = cl.N//2
if device == 'gpu':
    import cupy as cp
    xp = cp
    sync = lambda: cp.cuda.Device().synchronize()
    # column sums of x*y for (n, C) arrays, C a divisor of 32: thread (chunk, lane) adds 1024 entries of column lane % C (a warp
    # reads 32 neighbouring entries per step).  A library reduction along axis 0 runs on 512 threads per column and takes
    # seconds on 1e9 entries.
    _coldots = cp.RawKernel(r'''
extern "C" __global__ void coldots(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ part, long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long f = (t >> 5) * 32768 + (t & 31); double s = 0.0;
    for (int k = 0; k < 1024 && f < n; k++, f += 32) s += x[f] * y[f];
    part[t] = s;
}''', 'coldots')
    def coldot(X, Y, axis=0):
        n, C = X.shape; assert 32 % C == 0 and X.flags.c_contiguous and Y.flags.c_contiguous
        nt = ((n*C+32767)//32768)*32
        part = cp.empty(((nt+255)//256)*256, np.float64)
        k = _coldots16 if X.dtype == np.float16 else (_coldots32 if X.dtype == np.float32 else _coldots)
        k((part.size//256,), (256,), (X, Y, part, np.int64(n*C)))
        return part.reshape(-1, 32).sum(axis=0).reshape(-1, C).sum(axis=0)
    _coldots32 = cp.RawKernel(r'''
extern "C" __global__ void coldots32(const float* __restrict__ x, const float* __restrict__ y, double* __restrict__ part, long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long f = (t >> 5) * 32768 + (t & 31); double s = 0.0;
    for (int k = 0; k < 1024 && f < n; k++, f += 32) s += (double)x[f] * (double)y[f];
    part[t] = s;
}''', 'coldots32')
    _coldots16 = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__ void coldots16(const __half* __restrict__ x, const __half* __restrict__ y, double* __restrict__ part, long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long f = (t >> 5) * 32768 + (t & 31); double s = 0.0;
    for (int k = 0; k < 1024 && f < n; k++, f += 32) s += (double)__half2float(x[f]) * (double)__half2float(y[f]);
    part[t] = s;
}''', 'coldots16')
    axpy32 = cp.ElementwiseKernel('T v, float32 a', 'T u', 'u = u - a * v', 'ftlm_axpy32')
    scal32 = cp.ElementwiseKernel('float32 a', 'T u', 'u = u * a', 'ftlm_scal32')
    axpy_c = cp.ElementwiseKernel('T v, float64 a', 'T u', 'u = u - a * v', 'ftlm_axpy')
    scal_c = cp.ElementwiseKernel('float64 a', 'T u', 'u = u * a', 'ftlm_scal')
else:
    xp = np
    sync = lambda: None
    coldot = lambda x, y, axis: np.einsum('ij,ij->j', x, y)
    def axpy_c(v, a, u): u -= a*v
    def scal_c(a, u): u *= a

if os.environ.get('EDC_FTLM_SELFTEST') and device == 'gpu':            # column dots against the library dot
    for dt in (np.float64, np.complex128):
        X = (cp.random.standard_normal((1000003, 4))+(1j*cp.random.standard_normal((1000003, 4)) if dt is np.complex128 else 0)).astype(dt)
        Y = (cp.random.standard_normal((1000003, 4))+(1j*cp.random.standard_normal((1000003, 4)) if dt is np.complex128 else 0)).astype(dt)
        ref = cp.array([cp.vdot(cp.ascontiguousarray(X[:, r]), cp.ascontiguousarray(Y[:, r])).real for r in range(4)])
        d = coldot(X.view(np.float64), Y.view(np.float64), axis=0)
        got = d if dt is np.float64 else d.reshape(-1, 2).sum(axis=1)
        print('selftest coldot', dt.__name__, float(cp.abs(got-ref).max()), file=sys.stderr, flush=True)

blocks = hs.sectors(cl, nup, cl.N-nup)
order = sorted(range(len(blocks)), key=lambda b: -blocks[b][4])     # spread the blocks over the parts
mine = [b for i, b in enumerate(order) if i % nparts == part]


def re_dots(X, Y):
    """Re <x_r|y_r> for every column r of interleaved (dim, R) arrays."""
    if X.dtype == np.float16:                                       # half-precision storage; complex vectors are (dim, R, 2)
        if X.ndim == 2:
            return coldot(X, Y, axis=0)
        return coldot(X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1), axis=0).reshape(-1, 2).sum(axis=1)
    if X.dtype in (np.float64, np.float32):
        return coldot(X, Y, axis=0)
    ft = np.float32 if X.dtype == np.complex64 else np.float64
    d = coldot(X.view(ft), Y.view(ft), axis=0)                      # (dim, 2R): re, im of each column
    return d.reshape(-1, 2).sum(axis=1)


def start_vectors(e, seed, R=R):
    """R random start vectors, interleaved (dim, R), filled in chunks (no full-size temporaries); both modes use them."""
    dt = np.float64 if e.real else np.complex128
    rng = (cp.random.default_rng(seed) if xp is not np else np.random.default_rng(seed))
    V = xp.empty((e.dim, R), dt)
    ch = max(1, (1 << 24)//R)
    for i in range(0, e.dim, ch):
        c = min(ch, e.dim-i)
        blk = rng.standard_normal((c, R))
        V[i:i+c] = blk if e.real else blk+1j*rng.standard_normal((c, R))
    return V


TIMES = dict(apply=0.0)


def run_batched(e, seed, R=R, V=None):
    apply = e.batched(R)
    if V is None:
        V = start_vectors(e, seed, R)
    f32 = V.dtype in (np.float32, np.complex64, np.float16)         # reduced-precision vectors: scalars as float32 (GPU only)
    ax, sc, cast = (axpy32, scal32, np.float32) if f32 else (axpy_c, scal_c, np.float64)
    bc = (lambda a: a[None, :, None].astype(cast)) if V.ndim == 3 else (lambda a: a[None, :].astype(cast))
    s2 = float(getattr(e, 'fp16_scale', 1.0))**2 if V.dtype == np.float16 else 1.0   # stored vectors = true vectors * scale
    dots = (lambda X, Y: re_dots(X, Y)/s2)
    sc(bc(1.0/xp.sqrt(dots(V, V))), V)
    U = xp.zeros_like(V)
    A, B = [], []
    for it in range(NL):
        sync(); ta = time.perf_counter()
        apply(V, U, accumulate=True)
        sync(); TIMES['apply'] += time.perf_counter()-ta
        al = dots(V, U)
        ax(V, bc(al), U)
        be = xp.sqrt(dots(U, U))
        A.append(al); B.append(be)
        if it == NL-1:
            break
        sc(bc(1.0/be), U); sc(bc(-be), V)
        V, U = U, V
    A = np.array([x.get() if xp is not np else x for x in A]).T; B = np.array([x.get() if xp is not np else x for x in B]).T
    return A, B


def run_grouped(e, seed):
    """The R chains in groups of K run together on the stored connectivity; K = the largest of 8, 4, 2 whose vectors fit."""
    K = R
    if xp is not np:
        cp.get_default_memory_pool().free_all_blocks(); free = cp.cuda.Device().mem_info[0]
        vs = (8 if e.real else 16)//(4 if getattr(e, 'fp16', False) else 2 if getattr(e, 'fp32', False) else 1)
        K = next((k for k in (8, 4, 2) if k <= R and R % k == 0 and 2.3*e.dim*k*vs < free), 1)
        if os.environ.get('EDC_PLAN'):                             # the planner times K = 1, 2, 4, 8 and the launch block
            import edc_plan
            p = edc_plan.plan(e, vectors=R); K = p['K'] if R % p['K'] == 0 else K; e.planned = dict(block=p.get('block'), K=p['K'], cached=p.get('cached', False))
    e.group = K
    if K == 1:
        return run_single(e, seed)
    As, Bs = [], []
    for g in range(R//K):
        f32 = getattr(e, 'fp32', False); f16 = getattr(e, 'fp16', False)
        if f16:                                                         # half precision: stored = true * 2^k, k = round(log2 sqrt(dim))
            e.fp16_scale = 2.0**round(math.log2(math.sqrt(e.dim)))
            V = xp.empty((e.dim, K) if e.real else (e.dim, K, 2), np.float16)
            for j in range(K):
                w = hs._random_vector(xp, e.dim, e.real, seed*R+g*K+j, True)
                w *= np.float32(e.fp16_scale/math.sqrt(float(xp.vdot(w, w).real)))
                if e.real:
                    V[:, j] = w
                else:
                    V[:, j, 0] = w.real; V[:, j, 1] = w.imag
                del w
        else:
            V = xp.empty((e.dim, K), ((np.float32 if e.real else np.complex64) if f32 else (np.float64 if e.real else np.complex128)))
        for j in range(K if not f16 else 0):                            # the start vectors of the single mode
            V[:, j] = hs._random_vector(xp, e.dim, e.real, seed*R+g*K+j, f32)
        A, B = run_batched(e, seed*1000+g, K, V); As.append(A); Bs.append(B); del V
        if xp is not np:
            cp.get_default_memory_pool().free_all_blocks()
    return np.concatenate(As), np.concatenate(Bs)


def run_single(e, seed):
    fp32 = getattr(e, 'fp32', False)
    dt = (np.float32 if e.real else np.complex64) if fp32 else (np.float64 if e.real else np.complex128)
    axpy, dot = hs._vec_ops(xp, e.real, fp32)
    As, Bs = [], []
    for r in range(R):
        v = hs._random_vector(xp, e.dim, e.real, seed*R+r, fp32)          # one vector at a time (memory)
        v *= 1.0/math.sqrt(dot(v, v).real)
        u = xp.zeros(e.dim, dt)
        A, B = [], []
        for it in range(NL):
            e.matvec(v, u, accumulate=True)
            al = float(np.real(dot(v, u))); axpy(-al, v, u); be = math.sqrt(float(np.real(dot(u, u))))
            A.append(al); B.append(be)
            if it == NL-1:
                break
            u *= 1.0/be; v *= -be
            v, u = u, v
        As.append(A); Bs.append(B)
        del v, u
    return np.array(As), np.array(Bs)


for b in mine:
    k, ir, fl, pops, mult = blocks[b]
    sync(); t0 = time.perf_counter()
    e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops, precision=os.environ.get('EDC_PRECISION', 'fp64'),
                          stored=(False if os.environ.get('EDC_STORED') == '0' else None))   # EDC_STORED=0: matrix elements on the fly
    sync(); t1 = time.perf_counter()
    if e.dim == 0:
        print(json.dumps(dict(block=b, k_over_pi=[v/math.pi for v in k], irrep=ir, flip=fl, mult=mult, dim=0)), flush=True)
        continue
    TIMES['apply'] = 0.0
    A, B = dict(batched=run_batched, grouped=run_grouped).get(mode, run_single)(e, 1000+b)
    sync(); t2 = time.perf_counter()
    print(json.dumps(dict(block=b, k_over_pi=[v/math.pi for v in k], irrep=ir, flip=fl, mult=mult, dim=e.dim, real=e.real,
                          R=R, NL=NL, mode=mode, group=getattr(e, 'group', None), plan=getattr(e, 'planned', None), apply_seconds=TIMES['apply'],
                          setup_seconds=t1-t0, lanczos_seconds=t2-t1,
                          stored=getattr(e, '_st', None) is not None, stored_gib=getattr(e, 'stored_bytes', 0)/2**30,
                          build_seconds=getattr(e, 'stored_build_seconds', None),
                          device_gib=e.device_gib, pool_peak_gib=(cp.get_default_memory_pool().total_bytes()/2**30 if xp is not np else None),
                          alphas=A.tolist(), betas=B.tolist())), flush=True)
    del e
    if xp is not np:
        cp.get_default_memory_pool().free_all_blocks()
