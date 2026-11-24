#!/usr/bin/env python3
"""Kagome YC cylinder (ring ends) on a frozen GPU engine: lowest state of chosen (S_z, K) sectors with its profiles
(profiles.py: <S_z> per layer, energy of every row of bonds).

    kg_run.py ENGINE_DIR Lx c D out.json [sz=0,1] [K=0,1] [seeds=0] [start=random|A|B] [end_ring=x] [end_out=x]

  start=A / B   warm-up at a small bond dimension with a Hamiltonian whose rows of bonds touching the even (A) or odd
                (B) rings are 30 % stronger and the others 30 % weaker; that state is then the start of the run with
                the uniform Hamiltonian (no random warm-up sweeps).  Two starts with opposite ring-to-ring patterns.
  end_ring=x    coupling of the bonds inside the two end rings is 1 + x (V_end = those 4c bonds);
  end_out=x     coupling of the bonds between each end ring and its neighbouring c-spin layer is 1 + x."""
import sys, os, json, time
V = sys.argv[1]; sys.path.insert(0, V); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, edc_dmrg as ed
from fractions import Fraction
from profiles import profiles
opt = dict(a.split('=', 1) for a in sys.argv[6:]); Lx, c, D = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); out = sys.argv[5]
ints = lambda k, d: [int(x) for x in opt.get(k, d).split(',')]
szs, seeds, start = ints('sz', '0,1'), ints('seeds', '0'), opt.get('start', 'random'); Ks = ints('K', ','.join(str(k) for k in range(c//2+1)))
end_ring, end_out = float(opt.get('end_ring', 0)), float(opt.get('end_out', 0))


def lattice(Lx, c, row=lambda kind, ring: 1.0):
    """ed.kagome_cylinder(Lx, c) with the coupling of every row of bonds given by row(kind, ring), kind in ring/in/out."""
    layers = []; bonds = []
    for j in range(Lx):
        lr = len(layers); layers.append([Fraction(n, 2) for n in range(2*c)])
        bonds += [(lr, n, lr, (n+1) % (2*c), row('ring', j)) for n in range(2*c)]
        if j+1 < Lx:
            off = Fraction(1, 4) if j % 2 == 0 else Fraction(3, 4); layers.append([i+off for i in range(c)])
            for i in range(c):
                for dy in (Fraction(-1, 4), Fraction(1, 4)):
                    n = int(((i+off+dy) % c)*2); bonds.append((lr, n, lr+1, i, row('out', j))); bonds.append((lr+1, i, lr+2, n, row('in', j+1)))
    return ed.Lattice(c, layers, bonds, Fraction(1, 2), f'kagome cylinder YC{2*c}, {Lx} rings')


def ends(kind, ring):
    e = ring in (0, Lx-1)
    return 1.0 + (end_ring if (e and kind == 'ring') else 0.0) + (end_out if ((ring == 0 and kind == 'out') or (ring == Lx-1 and kind == 'in')) else 0.0)


lat = lattice(Lx, c, ends); model = ed.Model(lat); N = lat.N
ref = ed.kagome_cylinder(Lx, c)                                      # the unperturbed lattice must be the engine's own
assert (end_ring or end_out) or sorted(lat.global_bonds()) == sorted(ref.global_bonds())
import cupy as cp, edc_dmrg_gpu as eg, edc_dmrg_gpu3 as e3
cls = e3.DMRGGPU3; pool = cp.get_default_memory_pool(); version = 'dmrg-gpu-v3'


def run_from(pattern, nup, K, seed):
    """Warm-up on the patterned Hamiltonian at a small bond dimension, then the usual sweeps at D on the uniform one."""
    sgn = 1.0 if pattern == 'A' else -1.0; pat = ed.Model(lattice(Lx, c, lambda kind, ring: 1.0 + 0.3*sgn*(1 if ring % 2 == 0 else -1)))
    Dw = max(64, D//4); _, src, _ = eg.run(pat, Dw, nup=nup, K=K, cls=cls, sweeps=4, seed=seed, log=None)
    eng = cls(model, nup, K, seed=seed); n = eng.n
    for r in range(n+1):
        eng.bond[r].dims[:] = src.bond[r].dims
    for r in range(n):
        eng.plan[r] = ed.Plan(model, r, eng.bond[r], eng.bond[r+1]); eng.M[r] = src.M[r].copy()
    for r in range(n-1, 0, -1):                                        # centre on layer 0, the other layers right-orthonormal
        eng._grow_right(r)
    del src; rows = []; t0 = time.perf_counter()
    for al in ed.ALPHAS:
        E = eng.sweep(D, al, 3e-5 if al > 0 else 1e-6, noise=0.0); rows.append(dict(E=E, elapsed=time.perf_counter()-t0))
    rows[-1]['peak_pool_GiB'] = getattr(eng, 'peak', 0)/2**30
    return E, eng, rows


res = dict(lattice=lat.name, N=N, c=c, Lx=Lx, D=D, version=version, start=start, end_ring=end_ring, end_out=end_out, layer_sizes=[len(y) for y in lat.layers], runs=[])
for seed in seeds:
    for sz in szs:
        for K in Ks:
            nup = N//2 + sz; t0 = time.perf_counter()
            E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=cls, seed=seed, log=None) if start == 'random' else run_from(start, nup, K, seed)
            t1 = time.perf_counter(); pr = profiles(eng, D, nup, E); t2 = time.perf_counter()
            r = dict(seed=seed, Sz=sz, K=K, E=float(E), seconds=t1-t0, profile_seconds=t2-t1, pool_gib=float(rows[-1].get('peak_pool_GiB', 0.0)),
                     sweep_E=[float(x.get('E', np.nan)) for x in rows], **pr)
            res['runs'].append(r); json.dump(res, open(out, 'w'), indent=1)
            z = np.array(pr['layer_sz'])
            print(f'{lat.name} D {D} start {start} seed {seed} S_z {sz} K {K}: E = {E:.10f}, {t1-t0:.0f} s, pool {r["pool_gib"]:.1f} GiB, rows sum - E = {pr["row_sum_minus_E"]:+.1e}, '
                  f'ring mismatch {pr["ring_mismatch"]:.1e}\n   S_z per c-spin layer: ' + ' '.join(f'{x:+.3f}' for x in z[1::2]) + f' | all rings {z[0::2].sum():+.3f}\n   ring rows: '
                  + ' '.join(f'{q["E"]:+.4f}' for q in pr['rows'] if q['kind'] == 'ring'), flush=True)
            del eng
            pool.free_all_blocks()
