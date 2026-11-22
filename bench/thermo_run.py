#!/usr/bin/env python3
"""Finite-temperature Lanczos (FTLM) over every S_z sector and every symmetry block of a spin-1/2 cluster: the input of
the thermodynamics (thermo_post.py).  Sectors N_up >= N/2 only (S_z -> -S_z is added by the post-processing); blocks
from edc_hubbard_sym.sectors (complete, with multiplicities; spin flip at S_z = 0); engine edc_sub.SubEngine on one
device, stored connectivity where it fits, the on-the-fly product otherwise.
  blocks up to --dense states: the whole block is built column by column and diagonalised (exact);
  larger blocks: R random vectors, NL Lanczos steps each, K chains per pass (K = the largest of 8, 4, 2 that fits).
R depends on M = N_up - N/2 (large sectors need few vectors): --R '0:8,1:4,2:2,9:8' means R = 8 at M = 0, 4 at M = 1,
2 for M = 2..8, 8 from M = 9 on.
One JSON line per block is appended to --out; blocks already in that file are skipped (restart after a time limit).
Recurrence per chain: u <- u + H v; alpha = <v|u>; u <- u - alpha v; beta = |u|; (v, u) <- (u/beta, -beta v).

Usage: thermo_run.py <kind> <A1x,A1y,A2x,A2y> <J1[,J2,..]> --out file.jsonl [--NL 160] [--R spec] [--nup a:b]
                     [--part i/n] [--device gpu|cpu] [--dense 2000] [--max-seconds S]"""
import sys, os, json, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub

ap = argparse.ArgumentParser()
ap.add_argument('kind'); ap.add_argument('vectors'); ap.add_argument('J')
ap.add_argument('--out', required=True); ap.add_argument('--NL', type=int, default=160)
ap.add_argument('--R', default='0:8,1:4,2:2,9:8'); ap.add_argument('--nup', default='')
ap.add_argument('--part', default='0/1'); ap.add_argument('--device', default='gpu')
ap.add_argument('--dense', type=int, default=2000); ap.add_argument('--max-seconds', type=float, default=1e18)
ap.add_argument('--list', action='store_true', help='print the work list of this part and stop')
args = ap.parse_args()
T_START = time.perf_counter()
a = [int(v) for v in args.vectors.split(',')]; J = [float(v) for v in args.J.split(',')]
NL, device = args.NL, args.device
part, nparts = (int(v) for v in args.part.split('/'))
cl = L.LatticeCluster(args.kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J); N = cl.N
assert N % 2 == 0, 'even N only (S_z = 0 sector with spin flip)'
lo, hi = (int(v) for v in args.nup.split(':')) if args.nup else (N//2, N)
assert N//2 <= lo <= hi <= N
RSPEC = sorted((int(m), int(r)) for m, r in (p.split(':') for p in args.R.split(',')))
assert RSPEC[0][0] == 0


def R_of(M):
    return [r for m, r in RSPEC if m <= M][-1]


if device == 'gpu':
    import cupy as cp
    xp = cp
    sync = lambda: cp.cuda.Device().synchronize()
    # column sums of x*y for (n, C) arrays, C a divisor of 32: thread (chunk, lane) adds 1024 entries of column lane % C
    _coldots = cp.RawKernel(r'''
extern "C" __global__ void coldots(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ part, long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long f = (t >> 5) * 32768 + (t & 31); double s = 0.0;
    for (int k = 0; k < 1024 && f < n; k++, f += 32) s += x[f] * y[f];
    part[t] = s;
}''', 'coldots')
    def coldot(X, Y):
        n, C = X.shape; assert 32 % C == 0 and X.flags.c_contiguous and Y.flags.c_contiguous
        nt = ((n*C+32767)//32768)*32
        part_ = cp.empty(((nt+255)//256)*256, np.float64)
        _coldots((part_.size//256,), (256,), (X, Y, part_, np.int64(n*C)))
        return part_.reshape(-1, 32).sum(axis=0).reshape(-1, C).sum(axis=0)
    axpy_c = cp.ElementwiseKernel('T v, float64 a', 'T u', 'u = u - a * v', 'thermo_axpy')
    scal_c = cp.ElementwiseKernel('float64 a', 'T u', 'u = u * a', 'thermo_scal')
else:
    xp = np
    sync = lambda: None
    coldot = lambda x, y: np.einsum('ij,ij->j', x, y)
    def axpy_c(v, a_, u): u -= a_*v
    def scal_c(a_, u): u *= a_


def re_dots(X, Y):
    """Re <x_r|y_r> for every column r of interleaved (dim, R) arrays."""
    if X.dtype == np.float64:
        return coldot(X, Y)
    d = coldot(X.view(np.float64), Y.view(np.float64))               # (dim, 2R): re, im of each column
    return d.reshape(-1, 2).sum(axis=1)


TIMES = dict(apply=0.0)


def run_batched(e, V, nl):
    K = V.shape[1]
    apply = e.batched(K)
    scal_c((1.0/xp.sqrt(re_dots(V, V)))[None, :], V)
    U = xp.zeros_like(V)
    A, B = [], []
    for it in range(nl):
        sync(); ta = time.perf_counter()
        apply(V, U, accumulate=True)
        sync(); TIMES['apply'] += time.perf_counter()-ta
        al = re_dots(V, U)
        axpy_c(V, al[None, :], U)
        be = xp.sqrt(re_dots(U, U))
        A.append(al); B.append(be)
        if it == nl-1:
            break
        bs = xp.where(be > 1e-300, be, 1.0)                           # an exhausted chain is cut in the post-processing
        scal_c((1.0/bs)[None, :], U); scal_c((-be)[None, :], V)
        V, U = U, V
    g = (lambda x: x.get()) if xp is not np else (lambda x: x)
    return np.array([g(x) for x in A]).T, np.array([g(x) for x in B]).T


def run_single(e, seed, R, nl):
    axpy, dot = hs._vec_ops(xp, e.real, False)
    dt = np.float64 if e.real else np.complex128
    As, Bs = [], []
    for r in range(R):
        v = hs._random_vector(xp, e.dim, e.real, seed*64+r, False)
        v *= 1.0/math.sqrt(dot(v, v).real)
        u = xp.zeros(e.dim, dt)
        A, B = [], []
        for it in range(nl):
            sync(); ta = time.perf_counter()
            e.matvec(v, u, accumulate=True)
            sync(); TIMES['apply'] += time.perf_counter()-ta
            al = float(np.real(dot(v, u))); axpy(-al, v, u); be = math.sqrt(float(np.real(dot(u, u))))
            A.append(al); B.append(be)
            if it == nl-1:
                break
            u *= 1.0/(be if be > 1e-300 else 1.0); v *= -be
            v, u = u, v
        As.append(A); Bs.append(B)
        del v, u
    return np.array(As), np.array(Bs)


def run_lanczos(e, seed, R, nl):
    """R chains, K at a time (same start vectors whatever K is)."""
    vs = 8 if e.real else 16
    if xp is not np:
        cp.get_default_memory_pool().free_all_blocks(); free = cp.cuda.Device().mem_info[0]
    else:
        free = 0.25*os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    K = next((k for k in (8, 4, 2) if k <= R and R % k == 0 and 2.3*e.dim*k*vs < free), 1)
    e.group = K
    if K == 1:
        return run_single(e, seed, R, nl)
    As, Bs = [], []
    for g in range(R//K):
        V = xp.empty((e.dim, K), np.float64 if e.real else np.complex128)
        for j in range(K):
            V[:, j] = hs._random_vector(xp, e.dim, e.real, seed*64+g*K+j, False)
        A, B = run_batched(e, V, nl); As.append(A); Bs.append(B); del V
        if xp is not np:
            cp.get_default_memory_pool().free_all_blocks()
    return np.concatenate(As), np.concatenate(Bs)


def dense_spectrum(e):
    """Every eigenvalue of a small block: H e_j column by column, then a dense Hermitian diagonalisation."""
    n = e.dim; dt = np.float64 if e.real else np.complex128
    H = np.empty((n, n), dt)
    x = xp.zeros(n, dt); y = xp.zeros(n, dt)
    for j in range(n):
        x[j] = 1.0; y[:] = 0.0
        e.matvec(x, y, accumulate=True)
        H[:, j] = y.get() if xp is not np else y
        x[j] = 0.0
    herm = float(np.abs(H-H.conj().T).max()) if n else 0.0
    return np.linalg.eigvalsh(0.5*(H+H.conj().T)), herm


# ------------------------------------------------------------------ work list
tasks = []
for nup in range(lo, hi+1):
    blocks = hs.sectors(cl, nup, N-nup)
    per_k = {}
    for b in blocks:
        per_k[b[0]] = per_k.get(b[0], 0)+1
    for bi, b in enumerate(blocks):
        est = math.comb(N, nup)/(cl.Nc*per_k[b[0]])                  # estimate of the block dimension
        tasks.append((est*R_of(nup-N//2), nup, bi, b))
tasks.sort(key=lambda t: (-t[0], t[1], t[2]))
mine = [t for i, t in enumerate(tasks) if i % nparts == part]
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        line = line.strip()
        if line:
            try:
                r = json.loads(line); done.add((r['nup'], r['block']))
            except (ValueError, KeyError):
                pass                                                     # a line cut by a time limit
if args.list:
    for cost, nup, bi, b in mine:
        print(nup, bi, b[1], b[2], b[4], f'{cost:.3g}', 'done' if (nup, bi) in done else '')
    sys.exit(0)

for cost, nup, bi, (k, ir, fl, pops, mult) in mine:
    if (nup, bi) in done:
        continue
    if time.perf_counter()-T_START > args.max_seconds:
        print('time limit reached before', nup, bi, file=sys.stderr, flush=True)
        break
    M = nup-N//2; R = R_of(M)
    sync(); t0 = time.perf_counter()
    e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops)
    sync(); t1 = time.perf_counter()
    rec = dict(nup=nup, block=bi, k_over_pi=[v/math.pi for v in k], irrep=ir, flip=fl, mult=mult, dim=e.dim, J=J, N=N)
    if e.dim == 0:
        pass
    elif e.dim <= args.dense:
        ev, herm = dense_spectrum(e)
        rec.update(real=e.real, eigs=ev.tolist(), hermiticity=herm, setup_seconds=t1-t0, solve_seconds=time.perf_counter()-t1)
    else:
        TIMES['apply'] = 0.0
        A, B = run_lanczos(e, 100000*nup+bi, R, min(NL, e.dim))
        sync(); t2 = time.perf_counter()
        rec.update(real=e.real, R=R, NL=A.shape[1], group=e.group, apply_seconds=TIMES['apply'], setup_seconds=t1-t0,
                   lanczos_seconds=t2-t1, stored=getattr(e, '_st', None) is not None,
                   stored_gib=getattr(e, 'stored_bytes', 0)/2**30, alphas=A.tolist(), betas=B.tolist())
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec)+'\n')
    print(f'nup {nup} block {bi} {ir} flip {fl} dim {e.dim} R {rec.get("R", "dense")} '
          f'stored {rec.get("stored")} K {rec.get("group")} {time.perf_counter()-t0:.1f} s', file=sys.stderr, flush=True)
    del e
    if xp is not np:
        cp.get_default_memory_pool().free_all_blocks()
print('part finished', file=sys.stderr, flush=True)
