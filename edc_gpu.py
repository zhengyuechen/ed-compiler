#!/usr/bin/env python3
"""edc_gpu — the same generated, structure-aware matrix-vector kernel emitted as CUDA C (compiled at run time by
NVRTC through CuPy), with the Lanczos vector operations on the GPU.  Basis, symmetry table and orbit norms are built
by the CPU engine (OpenMP) and copied once to device memory.  One GPU thread per row; matrix-free.
"""
import time
import numpy as np
import cupy as cp
import edc


def cuda_source(g, K=1):
    L, LA = g.L, g.LA
    sym = g.group is not None
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    JAT = 'unsigned int' if g.JAT == 'uint32_t' else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    FFS = '__ffs' if L <= 32 else '__ffsll'
    low = (1 << LA)-1
    o = [f'typedef {T} st_t; typedef {JAT} ja_t;',
         f'#define FULLIDX(t) ((long long)JA[(t) >> {LA}] + JB[(t) & {g.c(low)}])',
         f'#define IDX(t) ({"(long long)TAB[FULLIDX(t)]" if sym else "FULLIDX(t)"})']
    body = []
    dconst = 0.0
    for fi, f in enumerate(g.fams):
        ff = {a: v for a, v in f.items() if a != '_meta'}
        meta = f['_meta']
        flip = [0]*L; cxy = [0.0]*L; wz = [0.0]*L
        for a, (b, jxy, jz) in ff.items():
            flip[a] = (1 << a) | (1 << b); cxy[a] = jxy/2; wz[a] = jz/4; dconst += jz/4
        o.append(f'__device__ const st_t FLIP{fi}[{L}] = {{{", ".join(g.c(v) for v in flip)}}};')
        o.append(f'__device__ const double CXY{fi}[{L}] = {{{", ".join(repr(float(v)) for v in cxy)}}};')
        o.append(f'__device__ const double WZ{fi}[{L}] = {{{", ".join(repr(float(v)) for v in wz)}}};')
        body.append(f'{{ st_t u = {g.u_expr(ff)};')
        if meta['mz']:
            if meta['wz'] is not None:
                body.append(f'd -= {2*meta["wz"]!r} * (double){POP}(u & {g.c(meta["mz"])});')
            else:
                body.append(f'{{ st_t z = u & {g.c(meta["mz"])}; while (z) {{ int a = {FFS}(z) - 1; z &= z - 1; d -= 2.0 * WZ{fi}[a]; }} }}')
        if meta['mxy']:
            cexpr = repr(meta['cxy']) if meta['cxy'] is not None else f'CXY{fi}[a]'
            body.append(f'{{ st_t fl = u & {g.c(meta["mxy"])}; while (fl) {{ int a = {FFS}(fl) - 1; fl &= fl - 1; '
                        f'st_t t = s ^ FLIP{fi}[a]; long long j = IDX(t); double c = {cexpr}; '
                        f'for (int q = 0; q < {K}; q++) acc[q] += c * X[j*{K}+q]; }} }}')
        body.append('}')
    h = g.m.h
    dconst = float(dconst)-float(h.sum())/2
    hu = {float(v) for v in h if v != 0}
    if len(hu) == 1:
        body.append(f'd += {next(iter(hu))!r} * (double){POP}(s);')
    elif len(hu) > 1:
        body += [f'd += {float(h[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if h[i] != 0]
    scale = 'SQ[i] * ' if sym else ''
    o.append('extern "C" __global__ void mv_free(const double* __restrict__ x, const double* __restrict__ X, '
             'double* __restrict__ y, const st_t* __restrict__ ST, const ja_t* __restrict__ JA, '
             'const unsigned int* __restrict__ JB, const unsigned int* __restrict__ TAB, '
             'const double* __restrict__ SQ, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append(f'  st_t s = ST[i]; double d = {dconst!r}; double acc[{K}]; for (int q = 0; q < {K}; q++) acc[q] = 0.0;')
    o += ['  '+b for b in body]
    o.append(f'  for (int q = 0; q < {K}; q++) y[i*{K}+q] = d * x[i*{K}+q] + {scale}acc[q];\n}}')
    return '\n'.join(o)+'\n'


def _lowest_distinct(A, B, nev, merge=1e-9):
    """Lowest nev distinct eigenvalues of the Lanczos tridiagonal; copies closer than merge*|E| are one level
    (plain Lanczos produces ghost copies of converged levels)."""
    from scipy.linalg import eigvalsh_tridiagonal
    if len(A) == 1:
        return [A[0]]
    ev = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, min(len(A)-1, 3*nev+2)))
    out = []
    for e in ev:
        if not out or abs(e-out[-1]) > merge*max(1.0, abs(e)):
            out.append(float(e))
        if len(out) == nev:
            break
    return out


SMALL = 400   # sectors up to this size are diagonalised densely: Lanczos run to the full dimension in floating point
              # point arithmetic makes ghost copies and is not exact there


def _dense_small(engine, nev, complex_):
    """Exact lowest levels of a small sector: H from n matvecs on unit vectors, dense Hermitian eigensolver.
    Returns (lowest level or list of the lowest nev distinct levels, n, seconds); keeps the lowest vector for
    ground_state_vector."""
    t0 = time.perf_counter()
    n = engine.dim
    dt = cp.complex128 if complex_ else cp.float64
    H = cp.empty((n, n), dt); x = cp.zeros(n, dt); y = cp.empty(n, dt)
    for c in range(n):
        x[:] = 0; x[c] = 1; engine.matvec(x, y); H[:, c] = y
    w, V = np.linalg.eigh(cp.asnumpy((H+H.conj().T)/2))
    engine._lanczos = dict(dense=(w, V[:, 0]))
    if nev == 1:
        return float(w[0]), n, time.perf_counter()-t0
    out = []
    for e in w:
        if not out or abs(e-out[-1]) > 1e-9*max(1.0, abs(e)):
            out.append(float(e))
        if len(out) == nev:
            break
    return out, n, time.perf_counter()-t0


_update = cp.ElementwiseKernel('float64 v, float64 vp, float64 a, float64 b', 'float64 w',
                               'w = w - a * v - b * vp', 'lanczos_update')


class GPUEngine:
    def __init__(self, model, nup, group=None, threads=None, block=256):
        t0 = time.perf_counter()
        self.cpu = edc.Engine(model, nup, group=group, threads=threads)
        self.cpu_setup_seconds = time.perf_counter()-t0
        g = self.cpu.gen
        self.sym = group is not None
        self.src = cuda_source(g)
        t1 = time.perf_counter()
        self.mod = cp.RawModule(code=self.src, options=('-std=c++17',))
        self.kern = self.mod.get_function('mv_free')
        self.compile_seconds = time.perf_counter()-t1
        t2 = time.perf_counter()
        c = self.cpu
        self.dim = c.dim
        self.ST = cp.asarray(c.st); self.JA = cp.asarray(c.ja); self.JB = cp.asarray(c.jb)
        self.TAB = cp.asarray(c.tab)
        self.SQ = cp.asarray(c.sq) if self.sym else cp.zeros(1)
        self.ISQ = cp.asarray(c.isq) if self.sym else None
        self.XS = cp.empty(self.dim) if self.sym else None
        cp.cuda.Stream.null.synchronize()
        self.upload_seconds = time.perf_counter()-t2
        self.block = block
        self.grid = (self.dim+block-1)//block
        self.device_gib = sum(a.nbytes for a in (self.ST, self.JA, self.JB, self.TAB, self.SQ))/2**30

    def matvec(self, x, y):
        if self.sym:
            cp.multiply(x, self.ISQ, out=self.XS); X = self.XS
        else:
            X = x
        self.kern((self.grid,), (self.block,), (x, X, y, self.ST, self.JA, self.JB, self.TAB, self.SQ, np.int64(self.dim)))
        return y

    def time_matvec(self, reps=5):
        x = cp.random.default_rng(0).standard_normal(self.dim); y = cp.empty_like(x)
        self.matvec(x, y); cp.cuda.Stream.null.synchronize()
        best = np.inf
        for _ in range(reps):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record(); self.matvec(x, y); e1.record(); e1.synchronize()
            best = min(best, cp.cuda.get_elapsed_time(e0, e1)/1e3)
        return best

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3):
        from scipy.linalg import eigvalsh_tridiagonal
        n = self.dim
        if n <= SMALL:
            return _dense_small(self, nev, False)
        v = cp.asarray(np.random.default_rng(seed).standard_normal(n)); v /= np.sqrt(_ddot(v, v))
        vp = cp.zeros(n); w = cp.empty(n)
        A, B = [], []; bprev = 0.0; e_old = None; hits = 0
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.matvec(v, w)
            al = _ddot(v, w)
            _update(v, vp, al, bprev, w)
            be = float(np.sqrt(_ddot(w, w)))
            A.append(al); B.append(be)
            if nev == 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0] if it > 1 else al
                hits = hits+1 if (e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e))) else 0; done = hits >= stable
            else:
                e = _lowest_distinct(A, B, nev)
                ok = (e_old is not None and len(e) == nev and len(e_old) == nev and
                      all(abs(a-b) < tol*max(1.0, abs(a)) for a, b in zip(e, e_old)))
                hits = hits+1 if ok else 0; done = hits >= stable
            # converged, or the Krylov space is exhausted (tiny sectors): the tridiagonal spectrum is then exact
            if done or be < 1e-12 or it >= n:
                self._lanczos = dict(A=list(A), B=list(B), seed=seed)
                return e, it, time.perf_counter()-t0
            e_old = e
            w *= 1.0/be
            bprev = be
            vp, v, w = v, w, vp
        return e_old, it, time.perf_counter()-t0


# ---------------------------------------------------------------------------------------------------------------
# Whole setup on the GPU: Lin tables (tiny) on the host; representatives, orbit sizes and the orbit table on the GPU.
def lin_tables(L, nup):
    import math
    LA = L//2
    v = np.arange(1 << LA, dtype=np.int64)
    pl = np.zeros(1 << LA, np.int64)
    for b in range(LA):
        pl += (v >> b) & 1
    order = np.lexsort((v, pl))
    lows = v[order]
    off = np.zeros(LA+2, np.int64); off[1:] = np.cumsum(np.bincount(pl, minlength=LA+1))
    jb = np.zeros(1 << LA, np.uint32); jb[order] = (np.arange(1 << LA)-off[pl[order]]).astype(np.uint32)
    vh = np.arange(1 << (L-LA), dtype=np.int64)
    ph = np.zeros(1 << (L-LA), np.int64)
    for b in range(L-LA):
        ph += (vh >> b) & 1
    need = nup-ph
    comb = np.array([math.comb(LA, int(q)) if 0 <= q <= LA else 0 for q in range(-L, L+1)])
    cnt = np.where((need >= 0) & (need <= LA), comb[need+L], 0)
    ja = np.zeros(1 << (L-LA), np.int64); ja[1:] = np.cumsum(cnt)[:-1]
    return lows, off, ja, jb, int(cnt.sum())


def setup_source(g):
    L, LA = g.L, g.LA
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    G = g.group
    o = [f'typedef {T} st_t;']
    for gi, perm in enumerate(G):
        o.append(f'__device__ __forceinline__ st_t G{gi}(st_t s) {{ return {g.apply_expr(perm, "s")}; }}')
    o.append('__device__ __forceinline__ st_t state_of(long long i, const long long* JA, const st_t* LOWS, const long long* OFF) {')
    o.append(f'  int lo = 0, hi = {1 << (L-LA)} - 1;')
    o.append('  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (JA[mid] <= i) lo = mid; else hi = mid - 1; }')
    o.append(f'  int p = {g.nup} - {POP}((st_t)lo); return ((st_t)lo << {LA}) | LOWS[OFF[p] + (i - JA[lo])]; }}')
    tests = ' '.join(f'{{ st_t r = G{gi}(s); if (r < s) return 0; st += (r == s); }}' for gi in range(1, len(G)))
    o.append(f'__device__ __forceinline__ int stab_if_rep(st_t s) {{ int st = 1; {tests} return st; }}')
    o.append('extern "C" __global__ void rep_flags(unsigned char* flags, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append('  flags[i] = stab_if_rep(state_of(i, JA, LOWS, OFF)) ? 1 : 0; }')
    o.append('extern "C" __global__ void make_reps(const long long* idx, st_t* ST, double* SQ, double* ISQ, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;')
    o.append(f'  st_t s = state_of(idx[k], JA, LOWS, OFF); ST[k] = s; double m = {len(G)}.0 / stab_if_rep(s); SQ[k] = sqrt(m); ISQ[k] = 1.0 / sqrt(m); }}')
    o.append('extern "C" __global__ void fill_tab(const st_t* ST, unsigned int* TAB, const long long* JA, const unsigned int* JB, long long n) {')
    o.append('  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return; st_t s = ST[k];')
    o += [f'  {{ st_t t = G{gi}(s); TAB[JA[t >> {LA}] + JB[t & {g.c((1 << LA)-1)}]] = (unsigned int)k; }}' for gi in range(len(G))]
    o.append('}')
    return '\n'.join(o)+'\n'


class GPUEngineFull(GPUEngine):
    """Like GPUEngine, but basis, representatives and orbit table are built on the GPU (symmetric case only)."""

    def __init__(self, model, nup, group, block=256):
        t0 = time.perf_counter()
        g = edc.Gen(model, nup, group, 1, 'static')
        g.source()                                   # fills family metadata used by cuda_source
        g.JAT = 'uint64_t'                           # the GPU kernels use 64-bit offsets throughout
        L = model.n
        lows, off, ja, jb, full = lin_tables(L, nup)
        st_np = np.uint32 if L <= 32 else np.uint64
        self.sym = True
        t1 = time.perf_counter()
        smod = cp.RawModule(code=setup_source(g), options=('-std=c++17',))
        self.mod = cp.RawModule(code=cuda_source(g), options=('-std=c++17',))
        self.kern = self.mod.get_function('mv_free')
        cp.cuda.Stream.null.synchronize()
        self.compile_seconds = time.perf_counter()-t1
        JA, LOWS, OFF = cp.asarray(ja), cp.asarray(lows.astype(st_np)), cp.asarray(off)
        self.JA, self.JB = JA, cp.asarray(jb)
        t2 = time.perf_counter()
        flags = cp.empty(full, cp.uint8)
        smod.get_function('rep_flags')(((full+block-1)//block,), (block,), (flags, JA, LOWS, OFF, np.int64(full)))
        idx = cp.flatnonzero(flags); del flags
        cp.get_default_memory_pool().free_all_blocks()
        self.dim = n = int(idx.size)
        self.ST = cp.empty(n, st_np); self.SQ = cp.empty(n); self.ISQ = cp.empty(n)
        smod.get_function('make_reps')(((n+block-1)//block,), (block,), (idx, self.ST, self.SQ, self.ISQ, JA, LOWS, OFF, np.int64(n)))
        del idx
        cp.get_default_memory_pool().free_all_blocks()
        self.TAB = cp.empty(full, cp.uint32)
        smod.get_function('fill_tab')(((n+block-1)//block,), (block,), (self.ST, self.TAB, JA, self.JB, np.int64(n)))
        cp.cuda.Stream.null.synchronize()
        self.gpu_setup_seconds = time.perf_counter()-t2
        self.cpu_setup_seconds = t1-t0
        self.upload_seconds = 0.0
        self.XS = cp.empty(n)
        self.block, self.grid = block, (n+block-1)//block
        self.full_dim = full
        self.device_gib = sum(a.nbytes for a in (self.ST, self.JA, self.JB, self.TAB, self.SQ, self.ISQ))/2**30


# ---------------------------------------------------------------------------------------------------------------
# Any momentum on the GPU: complex vectors, table = orbit index + translation index; whole setup on the GPU.
def cuda_source_k(g, chars):
    L, LA = g.L, g.LA
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    FFS = '__ffs' if L <= 32 else '__ffsll'
    G = g.group
    one = [abs(complex(c)-1) < 1e-9 for c in chars]
    o = ['#include <cupy/complex.cuh>', f'typedef {T} st_t;',
         f'__device__ const double CHR[{len(G)}] = {{{", ".join(repr(complex(c).real) for c in chars)}}};',
         f'__device__ const double CHI[{len(G)}] = {{{", ".join(repr(-complex(c).imag) for c in chars)}}};']   # conj(chi)
    for gi, perm in enumerate(G):
        o.append(f'__device__ __forceinline__ st_t G{gi}(st_t s) {{ return {g.apply_expr(perm, "s")}; }}')
    o.append('__device__ __forceinline__ st_t state_of(long long i, const long long* JA, const st_t* LOWS, const long long* OFF) {')
    o.append(f'  int lo = 0, hi = {1 << (L-LA)} - 1;')
    o.append('  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (JA[mid] <= i) lo = mid; else hi = mid - 1; }')
    o.append(f'  int p = {g.nup} - {POP}((st_t)lo); return ((st_t)lo << {LA}) | LOWS[OFF[p] + (i - JA[lo])]; }}')
    tests = ' '.join(f'{{ st_t r = G{gi}(s); if (r < s) return 0; if (r == s) {{ st++; {"" if one[gi] else "bad = 1;"} }} }}'
                     for gi in range(1, len(G)))
    o.append(f'__device__ __forceinline__ int stab_if_rep(st_t s) {{ int st = 1, bad = 0; {tests} return bad ? 0 : st; }}')
    o.append('extern "C" __global__ void rep_flags(unsigned char* flags, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append('  flags[i] = stab_if_rep(state_of(i, JA, LOWS, OFF)) ? 1 : 0; }')
    o.append('extern "C" __global__ void make_reps(const long long* idx, st_t* ST, double* SQ, double* ISQ, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;')
    o.append(f'  st_t s = state_of(idx[k], JA, LOWS, OFF); ST[k] = s; double m = {len(G)}.0 / stab_if_rep(s); SQ[k] = sqrt(m); ISQ[k] = 1.0 / sqrt(m); }}')
    o.append('extern "C" __global__ void fill_tab(const st_t* ST, unsigned int* TAB, unsigned char* GIX, const long long* JA, const unsigned int* JB, long long n) {')
    o.append('  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return; st_t s = ST[k];')
    o += [f'  {{ st_t t = G{gi}(s); long long f = JA[t >> {LA}] + JB[t & {g.c((1 << LA)-1)}]; TAB[f] = (unsigned int)k; GIX[f] = {gi}; }}'
          for gi in range(len(G))]
    o.append('}')
    # matrix-vector product
    body, dconst = [], 0.0
    for fi, f in enumerate(g.fams):
        ff = {a: v for a, v in f.items() if a != '_meta'}
        meta = f['_meta']
        flip = [0]*L; cxy = [0.0]*L; wz = [0.0]*L
        for a, (b, jxy, jz) in ff.items():
            flip[a] = (1 << a) | (1 << b); cxy[a] = jxy/2; wz[a] = jz/4; dconst += jz/4
        o.append(f'__device__ const st_t FLIP{fi}[{L}] = {{{", ".join(g.c(v) for v in flip)}}};')
        o.append(f'__device__ const double CXY{fi}[{L}] = {{{", ".join(repr(float(v)) for v in cxy)}}};')
        o.append(f'__device__ const double WZ{fi}[{L}] = {{{", ".join(repr(float(v)) for v in wz)}}};')
        body.append(f'{{ st_t u = {g.u_expr(ff)};')
        if meta['mz']:
            if meta['wz'] is not None:
                body.append(f'd -= {2*meta["wz"]!r} * (double){POP}(u & {g.c(meta["mz"])});')
            else:
                body.append(f'{{ st_t z = u & {g.c(meta["mz"])}; while (z) {{ int a = {FFS}(z) - 1; z &= z - 1; d -= 2.0 * WZ{fi}[a]; }} }}')
        if meta['mxy']:
            cexpr = repr(meta['cxy']) if meta['cxy'] is not None else f'CXY{fi}[a]'
            body.append(f'{{ st_t fl = u & {g.c(meta["mxy"])}; while (fl) {{ int a = {FFS}(fl) - 1; fl &= fl - 1; '
                        f'st_t t = s ^ FLIP{fi}[a]; long long f = (long long)JA[t >> {LA}] + JB[t & {g.c((1 << LA)-1)}]; '
                        f'unsigned char q = GIX[f]; acc += {cexpr} * complex<double>(CHR[q], CHI[q]) * XS[TAB[f]]; }} }}')
        body.append('}')
    h = g.m.h
    dconst = float(dconst)-float(h.sum())/2
    body += [f'd += {float(h[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if h[i] != 0]
    o.append('extern "C" __global__ void mv_free(const complex<double>* __restrict__ x, const complex<double>* __restrict__ XS, '
             'complex<double>* __restrict__ y, const st_t* __restrict__ ST, const long long* __restrict__ JA, '
             'const unsigned int* __restrict__ JB, const unsigned int* __restrict__ TAB, const unsigned char* __restrict__ GIX, '
             'const double* __restrict__ SQ, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append(f'  st_t s = ST[i]; double d = {dconst!r}; complex<double> acc(0.0, 0.0);')
    o += ['  '+b for b in body]
    o.append('  y[i] = d * x[i] + SQ[i] * acc; }')
    return '\n'.join(o)+'\n'


_CHUNK = 1 << 30      # cuBLAS takes 32-bit lengths: vectors longer than 2^31 - 1 must be split


def _dotc(x, y):
    """<x|y> for complex vectors through cuBLAS dotc (no conjugated copy), in chunks of 2^30."""
    from cupy import cublas
    n = x.size
    if n <= _CHUNK:
        return complex(cublas.dotc(x, y))
    return sum(complex(cublas.dotc(x[i:i+_CHUNK], y[i:i+_CHUNK])) for i in range(0, n, _CHUNK))


def _ddot(x, y):
    """x . y for real vectors through cuBLAS, in chunks of 2^30."""
    n = x.size
    if n <= _CHUNK:
        return float(cp.dot(x, y))
    return sum(float(cp.dot(x[i:i+_CHUNK], y[i:i+_CHUNK])) for i in range(0, n, _CHUNK))


_update_c = cp.ElementwiseKernel('complex128 v, complex128 vp, float64 a, float64 b', 'complex128 w',
                                 'w = w - a * v - b * vp', 'lanczos_update_c')


class GPUKEngine:
    """Translation symmetry at any momentum, everything on the GPU."""

    def __init__(self, model, nup, group, chars, block=256):
        t0 = time.perf_counter()
        g = edc.Gen(model, nup, group, 1, 'static'); g.source()
        L = model.n
        lows, off, ja, jb, full = lin_tables(L, nup)
        st_np = np.uint32 if L <= 32 else np.uint64
        mod = cp.RawModule(code=cuda_source_k(g, chars), options=('-std=c++17',))
        self.kern = mod.get_function('mv_free')
        JA, LOWS, OFF = cp.asarray(ja), cp.asarray(lows.astype(st_np)), cp.asarray(off)
        self.JA, self.JB = JA, cp.asarray(jb)
        flags = cp.empty(full, cp.uint8)
        mod.get_function('rep_flags')(((full+block-1)//block,), (block,), (flags, JA, LOWS, OFF, np.int64(full)))
        idx = cp.flatnonzero(flags); del flags
        cp.get_default_memory_pool().free_all_blocks()
        self.dim = n = int(idx.size)
        self.ST = cp.empty(max(n, 1), st_np); self.SQ = cp.empty(max(n, 1)); self.ISQ = cp.empty(max(n, 1))
        if n:
            mod.get_function('make_reps')(((n+block-1)//block,), (block,), (idx, self.ST, self.SQ, self.ISQ, JA, LOWS, OFF, np.int64(n)))
        del idx
        cp.get_default_memory_pool().free_all_blocks()
        self.TAB = cp.full(full, n, cp.uint32); self.GIX = cp.zeros(full, cp.uint8)
        if n:
            mod.get_function('fill_tab')(((n+block-1)//block,), (block,), (self.ST, self.TAB, self.GIX, JA, self.JB, np.int64(n)))
        self.XS = cp.zeros(n+1, cp.complex128)
        self.block, self.grid = block, (max(n, 1)+block-1)//block
        cp.cuda.Stream.null.synchronize()
        self.setup_seconds = time.perf_counter()-t0
        self.full_dim = full

    def matvec(self, x, y):
        cp.multiply(x, self.ISQ[:self.dim], out=self.XS[:self.dim])
        self.kern((self.grid,), (self.block,), (x, self.XS, y, self.ST, self.JA, self.JB, self.TAB, self.GIX, self.SQ, np.int64(self.dim)))
        return y

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3):
        from scipy.linalg import eigvalsh_tridiagonal
        n = self.dim
        if n == 0:
            return None, 0, 0.0
        if n <= SMALL:
            return _dense_small(self, nev, True)
        rng = np.random.default_rng(seed)
        v = cp.asarray(rng.standard_normal(n)+1j*rng.standard_normal(n)); v *= 1.0/float(cp.sqrt(_dotc(v, v).real))
        vp = cp.zeros(n, cp.complex128); w = cp.empty(n, cp.complex128); A, B = [], []; bprev = 0.0; e_old = None; hits = 0
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.matvec(v, w); al = float(_dotc(v, w).real); _update_c(v, vp, al, bprev, w)
            be = float(np.sqrt(_dotc(w, w).real)); A.append(al); B.append(be)   # cuBLAS dotc: no conj copy, no |w|^2 temporaries
            if nev == 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0] if it > 1 else al
                hits = hits+1 if (e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e))) else 0; done = hits >= stable
            else:
                e = _lowest_distinct(A, B, nev)
                ok = (e_old is not None and len(e) == nev and len(e_old) == nev and
                      all(abs(a-b) < tol*max(1.0, abs(a)) for a, b in zip(e, e_old)))
                hits = hits+1 if ok else 0; done = hits >= stable
            if done or be < 1e-12 or it >= n:
                self._lanczos = dict(A=list(A), B=list(B), seed=seed)
                return e, it, time.perf_counter()-t0
            e_old = e
            w *= 1.0/be
            bprev = be; vp, v, w = v, w, vp
        return e_old, it, time.perf_counter()-t0


# ---------------------------------------------------------------------------------------------------------------
# Memory-lean symmetric engine (momentum zero): no orbit table.  For every flipped state the representative is found
# on the fly (minimum over all group images, which also gives the stabiliser size), and its orbit index is the rank
# of its full-sector index in a bitmap of representatives (1 bit per state + one 32-bit count per 64-bit word).
def lean_source(g):
    L, LA = g.L, g.LA
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    FFS = '__ffs' if L <= 32 else '__ffsll'
    G = g.group
    low = g.c((1 << LA)-1)
    o = [f'typedef {T} st_t;']
    for gi, perm in enumerate(G):
        o.append(f'__device__ __forceinline__ st_t G{gi}(st_t s) {{ return {g.apply_expr(perm, "s")}; }}')
    mins = ' '.join(f'{{ st_t r = G{gi}(t); if (r < m) {{ m = r; st = 1; }} else if (r == m) st++; }}' for gi in range(1, len(G)))
    o.append(f'__device__ __forceinline__ st_t rep_of(st_t t, int* stab) {{ st_t m = t; int st = 1; {mins} *stab = st; return m; }}')
    tests = ' '.join(f'{{ st_t r = G{gi}(s); if (r < s) return 0; st += (r == s); }}' for gi in range(1, len(G)))
    o.append(f'__device__ __forceinline__ int stab_if_rep(st_t s) {{ int st = 1; {tests} return st; }}')
    o.append('__device__ __forceinline__ st_t state_of(long long i, const long long* JA, const st_t* LOWS, const long long* OFF) {')
    o.append(f'  int lo = 0, hi = {1 << (L-LA)} - 1;')
    o.append('  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (JA[mid] <= i) lo = mid; else hi = mid - 1; }')
    o.append(f'  int p = {g.nup} - {POP}((st_t)lo); return ((st_t)lo << {LA}) | LOWS[OFF[p] + (i - JA[lo])]; }}')
    # one thread per 64-bit word of the bitmap
    o.append('extern "C" __global__ void rep_bits(unsigned long long* bits, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long i0 = w * 64; if (i0 >= n) return;')
    o.append('  unsigned long long word = 0; for (int b = 0; b < 64 && i0 + b < n; b++) if (stab_if_rep(state_of(i0 + b, JA, LOWS, OFF))) word |= 1ull << b;')
    o.append('  bits[w] = word; }')
    o.append('extern "C" __global__ void word_counts(const unsigned long long* bits, unsigned int* cnt, long long nw) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (w < nw) cnt[w] = __popcll(bits[w]); }')
    o.append('extern "C" __global__ void make_reps(const unsigned long long* bits, const unsigned int* rank, st_t* ST, const long long* JA, const st_t* LOWS, const long long* OFF, long long nw, long long n) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (w >= nw) return; unsigned long long word = bits[w]; long long k = rank[w];')
    o.append('  while (word) { int b = __ffsll(word) - 1; word &= word - 1; ST[k++] = state_of(w * 64 + b, JA, LOWS, OFF); } }')
    body, dconst = [], 0.0
    for fi, f in enumerate(g.fams):
        ff = {a: v for a, v in f.items() if a != '_meta'}
        meta = f['_meta']
        flip = [0]*L
        for a, (b, jxy, jz) in ff.items():
            flip[a] = (1 << a) | (1 << b); dconst += jz/4
        o.append(f'__device__ const st_t FLIP{fi}[{L}] = {{{", ".join(g.c(v) for v in flip)}}};')
        assert meta['wz'] is not None and meta['cxy'] is not None, 'lean prototype: uniform couplings only'
        body.append(f'{{ st_t u = {g.u_expr(ff)};')
        if meta['mz']:
            body.append(f'd -= {2*meta["wz"]!r} * (double){POP}(u & {g.c(meta["mz"])});')
        if meta['mxy']:
            body.append(f'{{ st_t fl = u & {g.c(meta["mxy"])}; while (fl) {{ int a = {FFS}(fl) - 1; fl &= fl - 1; '
                        f'st_t t = s ^ FLIP{fi}[a]; int sj; st_t r = rep_of(t, &sj); '
                        f'long long f = (long long)JA[r >> {LA}] + JB[r & {low}]; long long wd = f >> 6; '
                        f'long long j = (long long)RANK[wd] + __popcll(BITS[wd] & ((1ull << (f & 63)) - 1)); '
                        f'acc += {meta["cxy"]!r} * sqrt((double)sj) * x[j]; }} }}')
        body.append('}')
    o.append('extern "C" __global__ void mv_lean(const double* __restrict__ x, double* __restrict__ y, const st_t* __restrict__ ST, '
             'const long long* __restrict__ JA, const unsigned int* __restrict__ JB, const unsigned long long* __restrict__ BITS, '
             'const unsigned int* __restrict__ RANK, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append(f'  st_t s = ST[i]; double d = {float(dconst)!r}; double acc = 0.0;')
    o += ['  '+b for b in body]
    o.append('  y[i] = d * x[i] + acc / sqrt((double)stab_if_rep(s)); }')
    return '\n'.join(o)+'\n'


class GPUEngineLean(GPUEngine):
    def __init__(self, model, nup, group, block=256):
        t0 = time.perf_counter()
        g = edc.Gen(model, nup, group, 1, 'static'); g.source()
        L = model.n
        lows, off, ja, jb, full = lin_tables(L, nup)
        st_np = np.uint32 if L <= 32 else np.uint64
        mod = cp.RawModule(code=lean_source(g), options=('-std=c++17',))
        self.kern = mod.get_function('mv_lean')
        JA, LOWS, OFF = cp.asarray(ja), cp.asarray(lows.astype(st_np)), cp.asarray(off)
        self.JA, self.JB = JA, cp.asarray(jb)
        nw = (full+63)//64
        self.BITS = cp.empty(nw, cp.uint64)
        mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (self.BITS, JA, LOWS, OFF, np.int64(full)))
        cnt = cp.empty(nw, cp.uint32)
        mod.get_function('word_counts')(((nw+block-1)//block,), (block,), (self.BITS, cnt, np.int64(nw)))
        incl = cp.cumsum(cnt, dtype=cp.uint64)
        self.dim = n = int(incl[-1])
        self.RANK = (incl-cnt).astype(cp.uint32); del incl, cnt
        self.ST = cp.empty(n, st_np)
        mod.get_function('make_reps')(((nw+block-1)//block,), (block,), (self.BITS, self.RANK, self.ST, JA, LOWS, OFF, np.int64(nw), np.int64(n)))
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        self.setup_seconds = time.perf_counter()-t0
        self.sym = False                                # matvec below does its own scaling
        self.block, self.grid = block, (n+block-1)//block
        self.full_dim = full
        self.device_gib = sum(a.nbytes for a in (self.ST, self.JA, self.JB, self.BITS, self.RANK))/2**30

    def matvec(self, x, y):
        self.kern((self.grid,), (self.block,), (x, y, self.ST, self.JA, self.JB, self.BITS, self.RANK, np.int64(self.dim)))
        return y


# ---------------------------------------------------------------------------------------------------------------
# Memory-lean engine at any momentum.  rep = G_m(t) for the minimising m, so t = G_m^{-1}(rep) and the phase
# conj(chi(g_t)) with g_t = m^{-1} equals chi(m).  Orbits absent at this momentum have no bit in the bitmap.
def lean_source_k(g, chars):
    L, LA = g.L, g.LA
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    FFS = '__ffs' if L <= 32 else '__ffsll'
    G = g.group
    low = g.c((1 << LA)-1)
    one = [abs(complex(c)-1) < 1e-9 for c in chars]
    o = ['#include <cupy/complex.cuh>', f'typedef {T} st_t;',
         f'__device__ const double CHR[{len(G)}] = {{{", ".join(repr(complex(c).real) for c in chars)}}};',
         f'__device__ const double CHI[{len(G)}] = {{{", ".join(repr(complex(c).imag) for c in chars)}}};']
    for gi, perm in enumerate(G):
        o.append(f'__device__ __forceinline__ st_t G{gi}(st_t s) {{ return {g.apply_expr(perm, "s")}; }}')
    mins = ' '.join(f'{{ st_t r = G{gi}(t); if (r < m) {{ m = r; st = 1; gm = {gi}; }} else if (r == m) st++; }}'
                    for gi in range(1, len(G)))
    o.append(f'__device__ __forceinline__ st_t rep_of(st_t t, int* stab, int* gmin) {{ st_t m = t; int st = 1, gm = 0; {mins} *stab = st; *gmin = gm; return m; }}')
    tests = ' '.join(f'{{ st_t r = G{gi}(s); if (r < s) return 0; if (r == s) {{ st++; {"" if one[gi] else "bad = 1;"} }} }}'
                     for gi in range(1, len(G)))
    o.append(f'__device__ __forceinline__ int stab_if_rep(st_t s) {{ int st = 1, bad = 0; {tests} return bad ? 0 : st; }}')
    o.append('__device__ __forceinline__ st_t state_of(long long i, const long long* JA, const st_t* LOWS, const long long* OFF) {')
    o.append(f'  int lo = 0, hi = {1 << (L-LA)} - 1;')
    o.append('  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (JA[mid] <= i) lo = mid; else hi = mid - 1; }')
    o.append(f'  int p = {g.nup} - {POP}((st_t)lo); return ((st_t)lo << {LA}) | LOWS[OFF[p] + (i - JA[lo])]; }}')
    o.append('extern "C" __global__ void rep_bits(unsigned long long* bits, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long i0 = w * 64; if (i0 >= n) return;')
    o.append('  unsigned long long word = 0; for (int b = 0; b < 64 && i0 + b < n; b++) if (stab_if_rep(state_of(i0 + b, JA, LOWS, OFF))) word |= 1ull << b;')
    o.append('  bits[w] = word; }')
    o.append('extern "C" __global__ void word_counts(const unsigned long long* bits, unsigned int* cnt, long long nw) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (w < nw) cnt[w] = __popcll(bits[w]); }')
    o.append('extern "C" __global__ void make_reps(const unsigned long long* bits, const unsigned int* rank, st_t* ST, const long long* JA, const st_t* LOWS, const long long* OFF, long long nw, long long n) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (w >= nw) return; unsigned long long word = bits[w]; long long k = rank[w];')
    o.append('  while (word) { int b = __ffsll(word) - 1; word &= word - 1; ST[k++] = state_of(w * 64 + b, JA, LOWS, OFF); } }')
    body, dconst = [], 0.0
    for fi, f in enumerate(g.fams):
        ff = {a: v for a, v in f.items() if a != '_meta'}
        meta = f['_meta']
        flip = [0]*L
        for a, (b, jxy, jz) in ff.items():
            flip[a] = (1 << a) | (1 << b); dconst += jz/4
        o.append(f'__device__ const st_t FLIP{fi}[{L}] = {{{", ".join(g.c(v) for v in flip)}}};')
        assert meta['wz'] is not None and meta['cxy'] is not None, 'lean prototype: uniform couplings only'
        body.append(f'{{ st_t u = {g.u_expr(ff)};')
        if meta['mz']:
            body.append(f'd -= {2*meta["wz"]!r} * (double){POP}(u & {g.c(meta["mz"])});')
        if meta['mxy']:
            body.append(f'{{ st_t fl = u & {g.c(meta["mxy"])}; while (fl) {{ int a = {FFS}(fl) - 1; fl &= fl - 1; '
                        f'st_t t = s ^ FLIP{fi}[a]; int sj, gm; st_t r = rep_of(t, &sj, &gm); '
                        f'long long f = (long long)JA[r >> {LA}] + JB[r & {low}]; long long wd = f >> 6; unsigned long long bw = BITS[wd]; '
                        f'if (!((bw >> (f & 63)) & 1ull)) continue; '
                        f'long long j = (long long)RANK[wd] + __popcll(bw & ((1ull << (f & 63)) - 1)); '
                        f'acc += {meta["cxy"]!r} * sqrt((double)sj) * complex<double>(CHR[gm], CHI[gm]) * x[j]; }} }}')
        body.append('}')
    o.append('extern "C" __global__ void mv_lean(const complex<double>* __restrict__ x, complex<double>* __restrict__ y, '
             'const st_t* __restrict__ ST, const long long* __restrict__ JA, const unsigned int* __restrict__ JB, '
             'const unsigned long long* __restrict__ BITS, const unsigned int* __restrict__ RANK, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append(f'  st_t s = ST[i]; double d = {float(dconst)!r}; complex<double> acc(0.0, 0.0);')
    o += ['  '+b for b in body]
    o.append('  y[i] = d * x[i] + acc / sqrt((double)stab_if_rep(s)); }')
    return '\n'.join(o)+'\n'


class GPUKEngineLean(GPUKEngine):
    def __init__(self, model, nup, group, chars, block=256):
        t0 = time.perf_counter()
        g = edc.Gen(model, nup, group, 1, 'static'); g.source()
        L = model.n
        lows, off, ja, jb, full = lin_tables(L, nup)
        st_np = np.uint32 if L <= 32 else np.uint64
        mod = cp.RawModule(code=lean_source_k(g, chars), options=('-std=c++17',))
        self.kern = mod.get_function('mv_lean')
        JA, LOWS, OFF = cp.asarray(ja), cp.asarray(lows.astype(st_np)), cp.asarray(off)
        self.JA, self.JB = JA, cp.asarray(jb)
        nw = (full+63)//64
        self.BITS = cp.empty(nw, cp.uint64)
        mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (self.BITS, JA, LOWS, OFF, np.int64(full)))
        cnt = cp.empty(nw, cp.uint32)
        mod.get_function('word_counts')(((nw+block-1)//block,), (block,), (self.BITS, cnt, np.int64(nw)))
        incl = cp.cumsum(cnt, dtype=cp.uint64)
        self.dim = n = int(incl[-1])
        self.RANK = (incl-cnt).astype(cp.uint32); del incl, cnt
        self.ST = cp.empty(max(n, 1), st_np)
        mod.get_function('make_reps')(((nw+block-1)//block,), (block,), (self.BITS, self.RANK, self.ST, JA, LOWS, OFF, np.int64(nw), np.int64(n)))
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        self.setup_seconds = time.perf_counter()-t0
        self.block, self.grid = block, (max(n, 1)+block-1)//block
        self.full_dim = full
        self.device_gib = sum(a.nbytes for a in (self.ST, self.JA, self.JB, self.BITS, self.RANK))/2**30

    def matvec(self, x, y):
        self.kern((self.grid,), (self.block,), (x, y, self.ST, self.JA, self.JB, self.BITS, self.RANK, np.int64(self.dim)))
        return y


# ---------------------------------------------------------------------------------------------------------------
# Memory-lean engine with a rectangular space group (momentum zero, A1).  Instead of applying each of the
# |P| * Lx * Ly operations as its own permutation, apply each point operation once and reach all translates by
# repeated one-step translations (two shift-mask groups each).  Group order: p, then dy, then dx (as rect_space_group).
def lean_source_rect(g, Lx, Ly, point_perms):
    L, LA = g.L, g.LA
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    FFS = '__ffs' if L <= 32 else '__ffsll'
    low = g.c((1 << LA)-1)
    tx = [((i % Lx+1) % Lx)+Lx*(i//Lx) for i in range(L)]
    ty = [(i % Lx)+Lx*((i//Lx+1) % Ly) for i in range(L)]
    nG = len(point_perms)*Lx*Ly
    o = [f'typedef {T} st_t;',
         f'__device__ __forceinline__ st_t TX(st_t s) {{ return {g.apply_expr(tx, "s")}; }}',
         f'__device__ __forceinline__ st_t TY(st_t s) {{ return {g.apply_expr(ty, "s")}; }}']
    for pi, perm in enumerate(point_perms):
        o.append(f'__device__ __forceinline__ st_t P{pi}(st_t s) {{ return {g.apply_expr(perm, "s")}; }}')
    # minimum over the whole group, with stabiliser count
    loops = []
    for pi in range(len(point_perms)):
        loops.append(f'{{ st_t u = P{pi}(t); for (int iy = 0; iy < {Ly}; iy++) {{ st_t v = u; for (int ix = 0; ix < {Lx}; ix++) {{ '
                     f'if (v < m) {{ m = v; st = 1; }} else if (v == m) st++; v = TX(v); }} u = TY(u); }} }}')
    o.append('__device__ __forceinline__ st_t rep_of(st_t t, int* stab) { st_t m = t; int st = 0; ' + ' '.join(loops) + ' *stab = st; return m; }')
    tests = []
    for pi in range(len(point_perms)):
        tests.append(f'{{ st_t u = P{pi}(s); for (int iy = 0; iy < {Ly}; iy++) {{ st_t v = u; for (int ix = 0; ix < {Lx}; ix++) {{ '
                     f'if (v < s) return 0; st += (v == s); v = TX(v); }} u = TY(u); }} }}')
    o.append('__device__ __forceinline__ int stab_if_rep(st_t s) { int st = 0; ' + ' '.join(tests) + ' return st; }')
    o.append('__device__ __forceinline__ st_t state_of(long long i, const long long* JA, const st_t* LOWS, const long long* OFF) {')
    o.append(f'  int lo = 0, hi = {1 << (L-LA)} - 1;')
    o.append('  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (JA[mid] <= i) lo = mid; else hi = mid - 1; }')
    o.append(f'  int p = {g.nup} - {POP}((st_t)lo); return ((st_t)lo << {LA}) | LOWS[OFF[p] + (i - JA[lo])]; }}')
    o.append('extern "C" __global__ void rep_bits(unsigned long long* bits, const long long* JA, const st_t* LOWS, const long long* OFF, long long n) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long i0 = w * 64; if (i0 >= n) return;')
    o.append('  unsigned long long word = 0; for (int b = 0; b < 64 && i0 + b < n; b++) if (stab_if_rep(state_of(i0 + b, JA, LOWS, OFF))) word |= 1ull << b;')
    o.append('  bits[w] = word; }')
    o.append('extern "C" __global__ void word_counts(const unsigned long long* bits, unsigned int* cnt, long long nw) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (w < nw) cnt[w] = __popcll(bits[w]); }')
    o.append('extern "C" __global__ void make_reps(const unsigned long long* bits, const unsigned int* rank, st_t* ST, const long long* JA, const st_t* LOWS, const long long* OFF, long long nw, long long n) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (w >= nw) return; unsigned long long word = bits[w]; long long k = rank[w];')
    o.append('  while (word) { int b = __ffsll(word) - 1; word &= word - 1; ST[k++] = state_of(w * 64 + b, JA, LOWS, OFF); } }')
    body, dconst = [], 0.0
    for fi, f in enumerate(g.fams):
        ff = {a: v for a, v in f.items() if a != '_meta'}
        meta = f['_meta']
        flip = [0]*L
        for a, (b, jxy, jz) in ff.items():
            flip[a] = (1 << a) | (1 << b); dconst += jz/4
        o.append(f'__device__ const st_t FLIP{fi}[{L}] = {{{", ".join(g.c(v) for v in flip)}}};')
        body.append(f'{{ st_t u = {g.u_expr(ff)};')
        if meta['mz']:
            body.append(f'd -= {2*meta["wz"]!r} * (double){POP}(u & {g.c(meta["mz"])});')
        if meta['mxy']:
            body.append(f'{{ st_t fl = u & {g.c(meta["mxy"])}; while (fl) {{ int a = {FFS}(fl) - 1; fl &= fl - 1; '
                        f'st_t t = s ^ FLIP{fi}[a]; int sj; st_t r = rep_of(t, &sj); '
                        f'long long f = (long long)JA[r >> {LA}] + JB[r & {low}]; long long wd = f >> 6; '
                        f'long long j = (long long)RANK[wd] + __popcll(BITS[wd] & ((1ull << (f & 63)) - 1)); '
                        f'acc += {meta["cxy"]!r} * sqrt((double)sj) * x[j]; }} }}')
        body.append('}')
    o.append('extern "C" __global__ void mv_lean(const double* __restrict__ x, double* __restrict__ y, const st_t* __restrict__ ST, '
             'const long long* __restrict__ JA, const unsigned int* __restrict__ JB, const unsigned long long* __restrict__ BITS, '
             'const unsigned int* __restrict__ RANK, long long n) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append(f'  st_t s = ST[i]; double d = {float(dconst)!r}; double acc = 0.0;')
    o += ['  '+b for b in body]
    o.append('  y[i] = d * x[i] + acc / sqrt((double)stab_if_rep(s)); }')
    return '\n'.join(o)+'\n', nG


class GPUEngineLeanRect(GPUEngineLean):
    def __init__(self, model, nup, Lx, Ly, block=256):
        t0 = time.perf_counter()
        pts = edc.rect_point_ops(Lx, Ly)
        g = edc.Gen(model, nup, None, 1, 'static'); g.source()
        L = model.n
        lows, off, ja, jb, full = lin_tables(L, nup)
        st_np = np.uint32 if L <= 32 else np.uint64
        src, self.group_order = lean_source_rect(g, Lx, Ly, pts)
        mod = cp.RawModule(code=src, options=('-std=c++17',))
        self.kern = mod.get_function('mv_lean')
        JA, LOWS, OFF = cp.asarray(ja), cp.asarray(lows.astype(st_np)), cp.asarray(off)
        self.JA, self.JB = JA, cp.asarray(jb)
        nw = (full+63)//64
        self.BITS = cp.empty(nw, cp.uint64)
        mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (self.BITS, JA, LOWS, OFF, np.int64(full)))
        cnt = cp.empty(nw, cp.uint32)
        mod.get_function('word_counts')(((nw+block-1)//block,), (block,), (self.BITS, cnt, np.int64(nw)))
        incl = cp.cumsum(cnt, dtype=cp.uint64)
        self.dim = n = int(incl[-1])
        self.RANK = (incl-cnt).astype(cp.uint32); del incl, cnt
        self.ST = cp.empty(n, st_np)
        mod.get_function('make_reps')(((nw+block-1)//block,), (block,), (self.BITS, self.RANK, self.ST, JA, LOWS, OFF, np.int64(nw), np.int64(n)))
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        self.setup_seconds = time.perf_counter()-t0
        self.sym = False
        self.block, self.grid = block, (n+block-1)//block
        self.full_dim = full
        self.device_gib = sum(a.nbytes for a in (self.ST, self.JA, self.JB, self.BITS, self.RANK))/2**30



def ground_state_vector(engine):
    """Replay the last real Lanczos run of `engine` (same seed, same coefficients) and build the lowest Ritz vector.
    Needs four vectors of the sector on the device.  Returns (psi, energy, residual norm |H psi - E psi|)."""
    from scipy.linalg import eigh_tridiagonal
    assert getattr(engine, 'real', True), 'real sectors only'
    L = engine._lanczos
    if 'dense' in L:                                  # small sector, diagonalised densely
        psi = cp.asarray(L['dense'][1].real.copy())
    else:
        psi = _replay(engine, L)
    hp = cp.empty(engine.dim); engine.matvec(psi, hp)
    E = _ddot(psi, hp)
    hp -= E*psi
    return psi, E, float(np.sqrt(_ddot(hp, hp)))


def _replay(engine, L):
    from scipy.linalg import eigh_tridiagonal
    A, B = np.array(L['A']), np.array(L['B'])
    m = len(A)
    ev, U = eigh_tridiagonal(A, B[:m-1], select='i', select_range=(0, 0))
    c = U[:, 0]
    n = engine.dim
    v = cp.asarray(np.random.default_rng(L['seed']).standard_normal(n)); v /= np.sqrt(_ddot(v, v))
    vp = cp.zeros(n); w = cp.empty(n); psi = c[0]*v
    for k in range(1, m):
        engine.matvec(v, w)
        _update(v, vp, float(A[k-1]), float(B[k-2]) if k > 1 else 0.0, w)
        w *= 1.0/float(B[k-1])
        psi += c[k]*w
        vp, v, w = v, w, vp
    del vp, w
    psi /= np.sqrt(_ddot(psi, psi))
    return psi
