#!/usr/bin/env python3
"""v3 against v3g (grouped cuBLAS products) on a kagome YC(2c) cylinder, standard schedule, one GPU.

    yc8_g.py Lx c D nup K gpu3|gpu3g seed out.json [mem]

Without `mem`: timed run (energy, seconds, products of every sweep; seconds by part; pool peak = largest size of cupy's
memory pool seen at the end of a layer solve).  With `mem`: the same run with a cupy memory hook that records the largest
amount of device memory in use at once (the timing of such a run is not comparable)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, cupy as cp
import edc_dmrg as ed, edc_dmrg_gpu as eg, edc_dmrg_gpu3 as e3, edc_dmrg_gpu3g as e3g

Lx, c, D, nup, K = (int(x) for x in sys.argv[1:6]); which = sys.argv[6]; seed = int(sys.argv[7]); outf = sys.argv[8]; mem = len(sys.argv) > 9 and sys.argv[9] == 'mem'
cls = {'gpu3': e3.DMRGGPU3, 'gpu3g': e3g.DMRGGPU3g}[which]
lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat)


class Peak(cp.cuda.MemoryHook):
    name = 'in_use'

    def __init__(self):
        self.cur = 0; self.peak = 0

    def malloc_postprocess(self, **kw):
        self.cur += kw['mem_size']; self.peak = max(self.peak, self.cur)

    def free_postprocess(self, **kw):
        self.cur -= kw['mem_size']


print(f'{lat.name}, N = {lat.N}, nup = {nup}, K = {K}, D = {D}, {cls.version}, seed {seed}' + (', memory hook' if mem else ''), flush=True)
hook = Peak() if mem else None
if hook:
    hook.__enter__()
t0 = time.perf_counter(); E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=cls, seed=seed, log=print); total = time.perf_counter() - t0
if hook:
    hook.__exit__(None, None, None)
tot = {}
for r in rows:
    for k, v in r['gpu'].items():
        tot[k] = tot.get(k, 0.0) + v
parts = {k: v for k, v in tot.items() if not k.startswith(('mv_', 'env_'))}
parts['moves'] = sum(r['svd'] for r in rows); parts['environments'] = sum(r['env'] for r in rows)
out = dict(lattice=lat.name, N=lat.N, nup=nup, K=K, D=D, version=cls.version, seed=seed, E=E, total_seconds=total, products=eng.stats['products'], seconds_by_part=parts,
           pool_peak_GiB=eng.peak/2**30, in_use_peak_GiB=(hook.peak/2**30 if hook else None), timing_comparable=not mem, device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), sweeps=rows)
print('summary: ' + json.dumps({k: v for k, v in out.items() if k != 'sweeps'}), flush=True)
json.dump(out, open(outf, 'w'), indent=1)
