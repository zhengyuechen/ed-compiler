#!/usr/bin/env python3
"""Memory-lean GPU engine for any square-lattice cluster (edc_lattice.Cluster), any momentum k and any 1D irrep of the
little group.  No orbit table: for a flipped state t the representative is the minimum over the group, found by applying
each point operation once and walking the translations tx^i ty^j step by step; its index is the rank of its full-sector
index in a bitmap of representatives.  The phase of the matrix element is chi(g_min) = chi_P(p) wx^i wy^j.
Real arithmetic is generated when every character is real (k in {0, pi}^2 and a real irrep)."""
import time, cmath, math
import numpy as np
import cupy as cp
import edc
import edc_lattice as el
from edc_gpu import lin_tables, GPUEngine, GPUKEngine, SMALL, _dense_small, _ddot


def lean_general_source(g, cl, ops, irrep, k, flip=0, rotmin=False):
    L, LA = g.L, g.LA
    T = 'unsigned int' if L <= 32 else 'unsigned long long'
    POP = '__popc' if L <= 32 else '__popcll'
    FFS = '__ffs' if L <= 32 else '__ffsll'
    low = g.c((1 << LA)-1)
    n1, m = cl.n1, cl.m
    table = cl.irrep_table(k)[irrep] if hasattr(cl, 'irrep_table') else el.IRREPS[irrep]
    chp = [complex(table[o]) for o in ops]
    wx = [cmath.exp(-1j*k[0]*i) for i in range(n1)]
    wy = [cmath.exp(-1j*k[1]*j) for j in range(m)]
    real = all(abs(z.imag) < 1e-12 for z in chp+wx+wy)
    FM = g.c((1 << L)-1)                         # spin flip: complement of all L bits
    V = 'double' if real else 'complex<double>'
    o = ['#include <cupy/complex.cuh>', f'typedef {T} st_t; typedef {V} vt;',
         f'__device__ __forceinline__ st_t TX(st_t s) {{ return {g.apply_expr(cl.tx, "s")}; }}',
         f'__device__ __forceinline__ st_t TY(st_t s) {{ return {g.apply_expr(cl.ty, "s")}; }}']
    for pi, name in enumerate(ops):
        o.append(f'__device__ __forceinline__ st_t P{pi}(st_t s) {{ return {g.apply_expr(cl.point_perm(name), "s")}; }}')
    for nm, arr in (('CP', chp), ('WX', wx), ('WY', wy)):
        o.append(f'__device__ const double {nm}R[{len(arr)}] = {{{", ".join(repr(z.real) for z in arr)}}};')
        o.append(f'__device__ const double {nm}I[{len(arr)}] = {{{", ".join(repr(z.imag) for z in arr)}}};')
    # character of g = tx^i ty^j P_p, and test chi(g) == 1
    o.append(f'#define FS {float(flip if flip else 1)!r}')
    o.append('__device__ __forceinline__ bool chi_is_one(int p, int i, int j, int f) {'
             ' complex<double> c = complex<double>(CPR[p], CPI[p]) * complex<double>(WXR[i], WXI[i]) * complex<double>(WYR[j], WYI[j]) * (f ? FS : 1.0);'
             ' double a = c.real() - 1.0, b = c.imag(); return a * a + b * b < 1e-18; }')
    if real:
        o.append('__device__ __forceinline__ vt chi(int p, int i, int j, int f) { return CPR[p] * WXR[i] * WYR[j] * (f ? FS : 1.0); }')
    else:
        o.append('__device__ __forceinline__ vt chi(int p, int i, int j, int f) { return complex<double>(CPR[p], CPI[p]) * '
                 'complex<double>(WXR[i], WXI[i]) * complex<double>(WYR[j], WYI[j]) * (f ? FS : 1.0); }')
    if rotmin:
        # tx is a simultaneous one-step rotation of the m blocks of n1 bits; min over tx^i by a table of block minima
        assert n1 <= 30
        BMk = (1 << n1)-1
        o.append(f'#define N1 {n1}\n#define MB {getattr(cl, "nblocks", m)}\n#define BM {BMk}u')
        o.append('__device__ __forceinline__ unsigned rotl_b(unsigned b, int i) { return i ? (((b << i) | (b >> (N1 - i))) & BM) : b; }')
        o.append('__device__ __forceinline__ st_t coset_min(st_t u, const unsigned* __restrict__ ARG, unsigned* candp) {'
                 ' unsigned cand = ARG[(unsigned)((u >> ((MB - 1) * N1)) & BM)];'
                 ' for (int jb = MB - 2; jb >= 0 && (cand & (cand - 1)); jb--) {'
                 '  unsigned bj = (unsigned)((u >> (jb * N1)) & BM), best = 0xFFFFFFFFu, nc = 0, c = cand;'
                 '  while (c) { int i = __ffs(c) - 1; c &= c - 1; unsigned r = rotl_b(bj, i); if (r < best) { best = r; nc = 1u << i; } else if (r == best) nc |= 1u << i; }'
                 '  cand = nc; }'
                 ' int i0 = __ffs(cand) - 1; st_t v = 0;'
                 ' for (int jb = 0; jb < MB; jb++) v |= (st_t)rotl_b((unsigned)((u >> (jb * N1)) & BM), i0) << (jb * N1);'
                 ' *candp = cand; return v; }')
        rm = []
        for pi in range(len(ops)):
            fl = (f'{{ unsigned cb; st_t vb = coset_min(u ^ {FM}, ARG, &cb); if (vb < mn) {{ mn = vb; st = __popc(cb); pm = {pi}; im = __ffs(cb) - 1; jm = j; fm = 1; }} '
                  f'else if (vb == mn) st += __popc(cb); }} ' if flip else '')
            rm.append(f'{{ st_t u = P{pi}(t); for (int j = 0; j < {m}; j++) {{ unsigned c; st_t v = coset_min(u, ARG, &c); '
                      f'if (v < mn) {{ mn = v; st = __popc(c); pm = {pi}; im = __ffs(c) - 1; jm = j; fm = 0; }} else if (v == mn) st += __popc(c); '
                      f'{fl}u = TY(u); }} }}')
        o.append('__device__ __forceinline__ st_t rep_of(st_t t, int* stab, int* P, int* I, int* J, int* F, const unsigned* __restrict__ ARG) {'
                 ' st_t mn = ~(st_t)0; int st = 0, pm = 0, im = 0, jm = 0, fm = 0; ' + ' '.join(rm) +
                 ' *stab = st; *P = pm; *I = im; *J = jm; *F = fm; return mn; }')
        rt = []
        for pi in range(len(ops)):
            flt = (f'{{ unsigned cb; st_t vb = coset_min(u ^ {FM}, ARG, &cb); if (vb < s) return 0; if (vb == s) {{ st += __popc(cb); '
                   f'unsigned c2 = cb; while (c2) {{ int i = __ffs(c2) - 1; c2 &= c2 - 1; if (!chi_is_one({pi}, i, j, 1)) bad = 1; }} }} }} ' if flip else '')
            rt.append(f'{{ st_t u = P{pi}(s); for (int j = 0; j < {m}; j++) {{ unsigned c; st_t v = coset_min(u, ARG, &c); '
                      f'if (v < s) return 0; if (v == s) {{ st += __popc(c); unsigned c2 = c; while (c2) {{ int i = __ffs(c2) - 1; c2 &= c2 - 1; '
                      f'if (!chi_is_one({pi}, i, j, 0)) bad = 1; }} }} {flt}u = TY(u); }} }}')
        o.append('__device__ __forceinline__ int stab_if_rep(st_t s, const unsigned* __restrict__ ARG) { int st = 0, bad = 0; '
                 + ' '.join(rt) + ' return bad ? 0 : st; }')
    RA = ', ARG' if rotmin else ''
    walk_min = []
    for pi in range(len(ops) if not rotmin else 0):
        fl = (f'{{ st_t vb = v ^ {FM}; if (vb < mn) {{ mn = vb; st = 1; pm = {pi}; im = i; jm = j; fm = 1; }} else if (vb == mn) st++; }} '
              if flip else '')
        walk_min.append(f'{{ st_t u = P{pi}(t); for (int j = 0; j < {m}; j++) {{ st_t v = u; for (int i = 0; i < {n1}; i++) {{ '
                        f'if (v < mn) {{ mn = v; st = 1; pm = {pi}; im = i; jm = j; fm = 0; }} else if (v == mn) st++; {fl}v = TX(v); }} u = TY(u); }} }}')
    if not rotmin:
        o.append('__device__ __forceinline__ st_t rep_of(st_t t, int* stab, int* P, int* I, int* J, int* F) { st_t mn = ~(st_t)0; '
                 'int st = 0, pm = 0, im = 0, jm = 0, fm = 0; ' + ' '.join(walk_min) + ' *stab = st; *P = pm; *I = im; *J = jm; *F = fm; return mn; }')
    walk_test = []
    for pi in range(len(ops) if not rotmin else 0):
        flt = (f'{{ st_t vb = v ^ {FM}; if (vb < s) return 0; if (vb == s) {{ st++; if (!chi_is_one({pi}, i, j, 1)) bad = 1; }} }} '
               if flip else '')
        walk_test.append(f'{{ st_t u = P{pi}(s); for (int j = 0; j < {m}; j++) {{ st_t v = u; for (int i = 0; i < {n1}; i++) {{ '
                         f'if (v < s) return 0; if (v == s) {{ st++; if (!chi_is_one({pi}, i, j, 0)) bad = 1; }} {flt}v = TX(v); }} u = TY(u); }} }}')
    if not rotmin:
        o.append('__device__ __forceinline__ int stab_if_rep(st_t s) { int st = 0, bad = 0; ' + ' '.join(walk_test) + ' return bad ? 0 : st; }')
    o.append('__device__ __forceinline__ st_t state_of(long long i, const long long* JA, const st_t* LOWS, const long long* OFF) {')
    o.append(f'  int lo = 0, hi = {1 << (L-LA)} - 1;')
    o.append('  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (JA[mid] <= i) lo = mid; else hi = mid - 1; }')
    o.append(f'  int p = {g.nup} - {POP}((st_t)lo); return ((st_t)lo << {LA}) | LOWS[OFF[p] + (i - JA[lo])]; }}')
    o.append('extern "C" __global__ void rep_bits(unsigned long long* bits, const long long* JA, const st_t* LOWS, const long long* OFF, long long n, const unsigned* __restrict__ ARG) {')
    o.append('  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long i0 = w * 64; if (i0 >= n) return;')
    o.append(f'  unsigned long long word = 0; for (int b = 0; b < 64 && i0 + b < n; b++) if (stab_if_rep(state_of(i0 + b, JA, LOWS, OFF){RA})) word |= 1ull << b;')
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
        flipm = [0]*L; cxy = [0.0]*L; wz = [0.0]*L
        for a, (b, jxy, jz) in ff.items():
            flipm[a] = (1 << a) | (1 << b); cxy[a] = jxy/2; wz[a] = jz/4; dconst += jz/4
        o.append(f'__device__ const st_t FLIP{fi}[{L}] = {{{", ".join(g.c(v) for v in flipm)}}};')
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
                        f'st_t t = s ^ FLIP{fi}[a]; int sj, pp, ii, jj, ff; st_t r = rep_of(t, &sj, &pp, &ii, &jj, &ff{RA}); '
                        f'long long f = (long long)JA[r >> {LA}] + JB[r & {low}]; long long wd = f >> 6; unsigned long long bw = BITS[wd]; '
                        f'if (!((bw >> (f & 63)) & 1ull)) continue; '
                        f'long long j = (long long)RANK[wd] + __popcll(bw & ((1ull << (f & 63)) - 1)); '
                        f'acc += ({cexpr} * sqrt((double)sj)) * chi(pp, ii, jj, ff) * x[j]; }} }}')
        body.append('}')
    h = g.m.h
    dconst = float(dconst)-float(h.sum())/2
    body += [f'd += {float(h[i])!r} * (double)((s >> {i}) & 1);' for i in range(L) if h[i] != 0]
    o.append('extern "C" __global__ void mv_lean(const vt* __restrict__ x, vt* __restrict__ y, const st_t* __restrict__ ST, '
             'const long long* __restrict__ JA, const unsigned int* __restrict__ JB, const unsigned long long* __restrict__ BITS, '
             'const unsigned int* __restrict__ RANK, long long n, const unsigned* __restrict__ ARG, int accumulate) {')
    o.append('  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;')
    o.append(f'  st_t s = ST[i]; double d = {dconst!r}; vt acc = 0.0;')
    o += ['  '+b for b in body]
    o.append(f'  vt r = d * x[i] + acc / sqrt((double)stab_if_rep(s{RA})); y[i] = accumulate ? y[i] + r : r; }}')
    return '\n'.join(o)+'\n', real, len(ops)*n1*m*(2 if flip else 1)


_axpy_neg = cp.ElementwiseKernel('T v, float64 a', 'T u', 'u = u - a * v', 'edc_axpy_neg')


def rotation_argmin_table(n1):
    """For every n1-bit block b: bit mask of the rotation amounts i with rotl(b, i) minimal."""
    b = np.arange(1 << n1, dtype=np.int64); mask = (1 << n1)-1
    rots = np.stack([((b << i) | (b >> (n1-i))) & mask if i else b for i in range(n1)])
    mn = rots.min(axis=0)
    arg = np.zeros(1 << n1, np.uint32)
    for i in range(n1):
        arg |= ((rots[i] == mn).astype(np.uint32) << np.uint32(i))
    return arg


class GPULean:
    """Ground state (lowest level) of one symmetry sector (k, irrep) of a model on an edc_lattice.Cluster."""

    def __init__(self, model, nup, cl, k, irrep='A1', flip=0, rotmin=None, block=256):
        t0 = time.perf_counter()
        assert flip in (0, 1, -1) and (flip == 0 or 2*nup == model.n), 'spin flip only in the Sz = 0 sector'
        assert flip == 0 or not model.h.any(), 'a magnetic field breaks spin-flip symmetry'
        table = cl.irrep_table(k)[irrep] if hasattr(cl, 'irrep_table') else el.IRREPS[irrep]
        ops = [o for o in cl.little_group(k) if o in table]
        assert ops[0] == 'E' and len(ops) == len(cl.little_group(k)), 'irrep must be defined on the whole little group'
        g = edc.Gen(model, nup, None, 1, 'static'); g.source()
        L = model.n
        lows, off, ja, jb, full = lin_tables(L, nup)
        st_np = np.uint32 if L <= 32 else np.uint64
        self.rotmin = rotmin if rotmin is not None else (10 <= cl.n1 <= 24)   # measured: faster for n1 = 20 (7.9x), slower for n1 = 6
        src, self.real, self.group_order = lean_general_source(g, cl, ops, irrep, k, flip, self.rotmin)
        self.ARG = cp.asarray(rotation_argmin_table(cl.n1)) if self.rotmin else cp.zeros(1, cp.uint32)
        self.flip = flip
        self.ops = ops
        self.cl, self.irrep, self.k, self.nup = cl, irrep, k, nup
        mod = cp.RawModule(code=src, options=('-std=c++17',))
        self.kern = mod.get_function('mv_lean')
        JA, LOWS, OFF = cp.asarray(ja), cp.asarray(lows.astype(st_np)), cp.asarray(off)
        self.JA, self.JB = JA, cp.asarray(jb)
        nw = (full+63)//64
        self.BITS = cp.empty(nw, cp.uint64)
        mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (self.BITS, JA, LOWS, OFF, np.int64(full), self.ARG))
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
        self.kern((self.grid,), (self.block,), (x, y, self.ST, self.JA, self.JB, self.BITS, self.RANK, np.int64(self.dim), self.ARG, np.int32(0)))
        return y

    def matvec_add(self, x, y):
        """y += H x"""
        self.kern((self.grid,), (self.block,), (x, y, self.ST, self.JA, self.JB, self.BITS, self.RANK, np.int64(self.dim), self.ARG, np.int32(1)))
        return y

    def operator(self, op_model):
        """Compile another symmetric operator (a model on the same sites, invariant under the same group) for this
        sector; returns a function y = O x that reuses the representative tables."""
        g = edc.Gen(op_model, self.nup, None, 1, 'static'); g.source()
        src, _, _ = lean_general_source(g, self.cl, self.ops, self.irrep, self.k, self.flip, self.rotmin)
        kern = cp.RawModule(code=src, options=('-std=c++17',)).get_function('mv_lean')
        def apply(x, y):
            kern((self.grid,), (self.block,), (x, y, self.ST, self.JA, self.JB, self.BITS, self.RANK, np.int64(self.dim), self.ARG, np.int32(0)))
            return y
        return apply

    def time_matvec(self, reps=2):
        dt = cp.float64 if self.real else cp.complex128
        x = cp.ones(self.dim, dt); y = cp.empty_like(x)
        self.matvec(x, y); cp.cuda.Stream.null.synchronize(); best = np.inf
        for _ in range(reps):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event(); e0.record(); self.matvec(x, y); e1.record(); e1.synchronize()
            best = min(best, cp.cuda.get_elapsed_time(e0, e1)/1e3)
        return best

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, nev=1, two_vectors=None, stable=3):
        if self.dim == 0:
            return None, 0, 0.0
        if self.dim <= SMALL:
            return _dense_small(self, nev, not self.real)
        if two_vectors is None:                      # use the two-vector form when three vectors would not fit
            free, total = cp.cuda.runtime.memGetInfo()
            two_vectors = 3*self.dim*(8 if self.real else 16) > 0.9*free
        if not two_vectors:
            return (GPUEngine.ground_energy if self.real else GPUKEngine.ground_energy)(self, tol, maxit, seed, nev, stable)
        return self.ground_energy_2v(tol, maxit, seed, nev, stable)

    def ground_energy_2v(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3):
        """Lanczos with two vectors (Paige): u holds -beta_{k-1} v_{k-1}; u += H v; alpha = <v|u>; u -= alpha v;
        beta = |u|; then (v, u) <- (u / beta, -beta v).  Same coefficients as the three-vector form."""
        from scipy.linalg import eigvalsh_tridiagonal
        from edc_gpu import _lowest_distinct, _dotc
        n = self.dim
        rng = np.random.default_rng(seed)
        if self.real:
            v = cp.asarray(rng.standard_normal(n)); dot = _ddot
        else:
            v = cp.asarray(rng.standard_normal(n)+1j*rng.standard_normal(n)); dot = lambda a, b: _dotc(a, b).real
        v *= 1.0/np.sqrt(dot(v, v))
        u = cp.zeros_like(v)
        A, B = [], []; e_old = None; hits = 0
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        for it in range(1, maxit+1):
            self.matvec_add(v, u)
            al = dot(v, u)
            _axpy_neg(v, al, u)                         # u -= al v in place, no temporaries of size n
            be = float(np.sqrt(dot(u, u)))
            A.append(al); B.append(be)
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
            u *= 1.0/be; v *= -be
            v, u = u, v
        return e_old, it, time.perf_counter()-t0



def structure_factor_model(cl, Qs):
    """O = sum_{i<j} c_ij S_i.S_j with c_ij = mean over Q in Qs of cos(Q.(r_i - r_j)).  For a state psi,
    M^2(Q) = (2 <O> + 3N/4) / (N (N + 2))  (Richter & Schulenburg 2010).  Average Q over a point-group orbit so that O
    is invariant (e.g. [(pi, 0), (0, pi)])."""
    m = edc.SpinHalfModel(cl.N)
    for i in range(cl.N):
        for j in range(i+1, cl.N):
            dx = cl.coord[i][0]-cl.coord[j][0]; dy = cl.coord[i][1]-cl.coord[j][1]
            c = float(np.mean([math.cos(Q[0]*dx+Q[1]*dy) for Q in Qs]))
            if abs(c) > 1e-12:
                m.heisenberg(i, j, c)
    return m


def m2(engine, psi, cl, Qs):
    O = engine.operator(structure_factor_model(cl, Qs))
    y = cp.empty_like(psi); O(psi, y)
    from edc_gpu import _dotc
    val = _ddot(psi, y) if engine.real else _dotc(psi, y).real
    return (2*val+0.75*cl.N)/(cl.N*(cl.N+2))
