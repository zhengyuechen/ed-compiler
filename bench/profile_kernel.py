#!/usr/bin/env python3
"""Where the time of one product goes on the GPU (sub engine): kernel variants that remove one stage at a time.
  full     : the production kernel
  nogather : the read of x[j] replaced by a cheap function of j (index search still done)
  noindex  : also no index/bitmap/rank look-ups (j taken from the representative's bits)
  norep    : also no representative search (r = the flipped state itself)
  diag     : no off-diagonal loop at all (row state, its own stabiliser, diagonal term)
Differences between consecutive variants estimate the cost of gather, index search, representative search and the
bond loop itself.  Results of the variants other than 'full' are meaningless numbers; only their times matter.
Usage: profile_kernel.py <kind|square> <A1x,A1y,A2x,A2y> <J1[,J2]> <k1/pi,k2/pi> <irrep> <flip> [store 0|1] [label]"""
import sys, os, re, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cupy as cp
import edc_lattices as L, edc_lattice as el, edc_sub

kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]; flip = int(sys.argv[6])
store = {'0': False, '1': True}[sys.argv[7]] if len(sys.argv) > 7 else None
label = sys.argv[8] if len(sys.argv) > 8 else f'{kind}{a}'
if kind == 'square':
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); model = cl.j1j2(*J)
else:
    cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
e = edc_sub.SubEngine(model, cl.N//2, cl, k, ir, flip, device='gpu', store_states=store)

LOOKUP = ('u64 f = (u64)index_of(r); u64 wd = BITS[f >> 6]; if (!((wd >> (f & 63)) & 1ull)) continue; '
          'i64 j = (i64)RANK[f >> 6] + POPC64(wd & ((1ull << (f & 63)) - 1));')
assert LOOKUP in e.src


def variant(name):
    s = e.src
    if name == 'full':
        return s
    s = s.replace('* ph * XVAL(j);', '* ph * (double)(j & 7);')
    if name == 'nogather':
        return s
    s = s.replace(LOOKUP, 'i64 j = (i64)(r & 0xffffull);')
    if name == 'noindex':
        return s
    s = re.sub(r'st_t r = rep_of\(s \^ FLIP(\d+)\[a\], &sj, &ph\);', r'st_t r = s ^ FLIP\1[a]; sj = 1; ph = 1.0;', s)
    if name == 'norep':
        return s
    return s.replace('while (fl) {', 'while (0 && fl) {')


dt = cp.float64 if e.real else cp.complex128
rng = np.random.default_rng(0)
x = cp.asarray(rng.standard_normal(e.dim)) if e.real else cp.asarray(rng.standard_normal(e.dim)+1j*rng.standard_normal(e.dim))
y = cp.empty_like(x)
out = dict(label=label, kind=kind, cluster=a, N=cl.N, J=J, k_over_pi=[v/math.pi for v in k], irrep=ir, flip=flip,
           dim=e.dim, real=e.real, store=e.store, group_order=e.group_order, w=e.w, cosets=e.ncos, times={})
for name in ('full', 'nogather', 'noindex', 'norep', 'diag'):
    mod = cp.RawModule(code=variant(name), options=('-std=c++17',))
    kern = mod.get_function('mv')
    e._bind_gpu(mod)
    best = math.inf
    for rep in range(4):
        cp.cuda.Device().synchronize(); t0 = time.perf_counter()
        kern((e.grid,), (e.block,), (x, y, np.int64(e.dim), np.int32(0)))
        cp.cuda.Device().synchronize(); t = time.perf_counter()-t0
        if rep:
            best = min(best, t)
    out['times'][name] = best
    if name == 'full':
        ref = y.copy()
        e._ensure_bound(); e.matvec(x, y)                      # the variant 'full' must equal the engine's product
        out['full_check'] = float(cp.max(cp.abs(y-ref)))
T = out['times']
out['ns_per_row'] = {n: T[n]/e.dim*1e9 for n in T}
out['split_ns_per_row'] = dict(gather=(T['full']-T['nogather'])/e.dim*1e9, index=(T['nogather']-T['noindex'])/e.dim*1e9,
                               representative=(T['noindex']-T['norep'])/e.dim*1e9, bond_loop=(T['norep']-T['diag'])/e.dim*1e9,
                               row_overhead=T['diag']/e.dim*1e9)
print(json.dumps(out), flush=True)
