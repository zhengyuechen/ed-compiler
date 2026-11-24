#!/usr/bin/env python3
"""GPU back end of the layer DMRG against exact diagonalisation on a cylinder that ED can solve (needs a GPU).

1. One product with the effective Hamiltonian of every layer, device against host, on random vectors.
2. YC6, two rings and a last layer (18 spins), nearest-neighbour Heisenberg: lowest singlet (lowest S_z = 0 level),
   lowest triplet (lowest S_z = 1 level) and their difference, in every momentum, against the product-basis ED;
   at D = 512 the bonds hold the whole space, at D = 64 and 128 they truncate.
Usage: check_small.py [out.json] [gpu|gpu1|gpu1t|gpu2|gpu3|gpu3d|gpu3s]"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, cupy as cp, edc_dmrg as ed, edc_dmrg_gpu as eg

lat = ed.kagome_cylinder(2, 3, tail=True); model = ed.Model(lat); N = lat.N; rng = np.random.default_rng(5); out = dict(lattice=lat.name, N=N)
_w = sys.argv[2] if len(sys.argv) > 2 else 'gpu'
if _w in ('gpu1t', 'gpu2'):
    import edc_dmrg_gpu2 as e2
if _w in ('gpu3', 'gpu3d'):
    import edc_dmrg_gpu3 as e3
if _w == 'gpu3s':
    import edc_dmrg_gpu3s as e3s
CLS = {'gpu': eg.DMRGGPU, 'gpu1a': eg.DMRGGPU1, 'gpu1': eg.DMRGGPU1b, 'gpu1t': e2.DMRGGPU1t if _w == 'gpu1t' else None, 'gpu2': e2.DMRGGPU2 if _w == 'gpu2' else None,
       'gpu3': e3.DMRGGPU3 if _w == 'gpu3' else None, 'gpu3d': e3.DMRGGPU3d if _w == 'gpu3d' else None, 'gpu3s': e3s.DMRGGPU3s if _w == 'gpu3s' else None}[_w]; out['version'] = CLS.version; print('back end', CLS.version)
eng = eg.DMRGGPU(model, N//2, 0); eng.host = True; eng.sweep(48, 1e-2, 1e-6); eng.sweep(48, 1e-2, 1e-6); worst = 0.0; _p = lambda a: a.ctypes.data
for r in range(eng.n):
    p = eng.plan[r]; n = p.size; v = rng.standard_normal(n); ref = np.zeros(n); U = np.empty(n); oU = np.empty(n)
    eng.lib.heff(p.ref, _p(eng.LH[r]), _p(eng.GL[r]), _p(eng.RH[r+1]), _p(eng.GR[r+1]), _p(eng.Q[r]), _p(eng.P[r+1]), _p(v), _p(ref), _p(U), _p(oU))
    got = cp.asnumpy(eg.Product(eng, r, {}).apply(cp.asarray(v))); d = float(np.abs(got-ref).max()); worst = max(worst, d)
    print(f'product, layer {r}: tensor {n}, device - host = {d:.1e}', flush=True)
    if r < eng.n-1:
        eng.move_right(r, 48, 0.0)
out['product_max_difference'] = worst; assert worst < 1e-10
ex = {nup: ed.exact_levels(lat.global_bonds(), N, nup, 1.0, tperm=lat.translation(), c=lat.c) for nup in (N//2, N//2+1)}
e0x, e1x = min(ex[N//2].values()), min(ex[N//2+1].values()); out['exact'] = dict(singlet=e0x, triplet=e1x, gap=e1x-e0x)
print(f'exact: singlet {e0x:.10f}, triplet {e1x:.10f}, gap {e1x-e0x:.10f}', flush=True); out['dmrg'] = []
for D in (64, 128, 512):
    lev = {}
    for nup in (N//2, N//2+1):
        for K in range(lat.c//2+1):
            lev[(nup, K)] = eg.run(model, D, nup=nup, K=K, cls=CLS, log=None)[0]
    e0 = min(lev[(N//2, K)] for K in range(lat.c//2+1)); e1 = min(lev[(N//2+1, K)] for K in range(lat.c//2+1))
    out['dmrg'].append(dict(D=D, singlet=e0, triplet=e1, gap=e1-e0, errors=dict(singlet=e0-e0x, triplet=e1-e1x, gap=e1-e0-(e1x-e0x)),
                            by_momentum={f'{nup-N//2},{K}': e for (nup, K), e in lev.items()},
                            momentum_errors={f'{nup-N//2},{K}': lev[(nup, K)]-ex[nup][K] for (nup, K) in lev}))
    print(f'D = {D:3d}: singlet {e0:.10f} ({e0-e0x:+.1e}), triplet {e1:.10f} ({e1-e1x:+.1e}), gap {e1-e0:.10f} ({e1-e0-(e1x-e0x):+.1e}); '
          f'largest error over the momenta {max(abs(v) for v in out["dmrg"][-1]["momentum_errors"].values()):.1e}', flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
