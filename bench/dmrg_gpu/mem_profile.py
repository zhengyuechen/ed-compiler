#!/usr/bin/env python3
"""Where the device memory of the GPU layer DMRG (version 3, fixed rule) goes, on short cylinders (needs a GPU).

    mem_profile.py Lx c D [nup] [sweeps at the full D] [out.json]        kagome YC(2c) cylinder with Lx rings, K = 0

A cupy memory hook counts what is allocated from the pool at each moment ("in use"; nvidia-smi shows what the pool
keeps, which is more).  Recorded per bond dimension of the schedule:
  resident   in use between two steps: the tensors and the environments of all layers (plus plan tables)
  solve      peak during a layer solve, and how much of it is above the resident level (work buffers of the product,
             Krylov vectors)
  move       peak during a move between layers (blocks, factors, new tensors, growth of the environment)
  grow       the growth of an environment alone (inside the move)
and at the end the bytes of each family of arrays, the largest tensor, the peak of the pool, and how many of the 40
reserved Krylov vectors the solves used (in the solve with the largest rise, and over the largest tensors).  Timing with the
hook active means nothing."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, cupy as cp, edc_dmrg as ed, edc_dmrg_gpu as eg, edc_dmrg_gpu3 as e3


class Peak(cp.cuda.MemoryHook):
    name = 'in_use'

    def __init__(self):
        self.cur = 0; self.peak = 0

    def malloc_postprocess(self, **kw):
        self.cur += kw['mem_size']; self.peak = max(self.peak, self.cur)

    def free_postprocess(self, **kw):
        self.cur -= kw['mem_size']


HOOK = Peak()


class Prof(e3.DMRGGPU3):
    def _phase(self, name, f, *a):
        h = HOOK; outer = h.peak; base = h.cur; h.peak = h.cur; out = f(*a)
        rec = self.memlog.setdefault(self.curD, {}).setdefault(name, dict(peak=0, above_start=0, resident_at_start=0))
        if h.peak - base >= rec['above_start']:
            rec['above_start'] = h.peak - base; rec['resident_at_start'] = base
        rec['peak'] = max(rec['peak'], h.peak); h.peak = max(outer, h.peak)
        return out

    def sweep(self, Dmax, alpha, tol=1e-6, noise=0.0):
        self.curD = Dmax
        if not hasattr(self, 'memlog'):
            self.memlog = {}; self.klog = {}
        return super().sweep(Dmax, alpha, tol, noise)

    def solve(self, r, tol=1e-6, maxit=40):                            # Krylov vectors used = products of this solve (one new vector per product)
        p0 = self.stats['products']; rec = self.memlog.setdefault(self.curD, {}).get('solve', {}); before = rec.get('above_start', -1)
        out = self._phase('solve', super().solve, r, tol, maxit); used = self.stats['products'] - p0; rec = self.memlog[self.curD]['solve']
        self.klog.setdefault(self.curD, []).append((int(self.plan[r].size), used))
        if rec['above_start'] != before:
            rec['krylov_at_peak'] = used; rec['tensor_at_peak'] = int(self.plan[r].size)
        return out

    def move_right(self, r, Dmax, alpha, cutoff=1e-12):
        return self._phase('move', super().move_right, r, Dmax, alpha, cutoff)

    def move_left(self, r, Dmax, alpha, cutoff=1e-12):
        return self._phase('move', super().move_left, r, Dmax, alpha, cutoff)

    def _grow_left(self, r):
        return self._phase('grow', super()._grow_left, r) if hasattr(self, 'memlog') else super()._grow_left(r)

    def _grow_right(self, r):
        return self._phase('grow', super()._grow_right, r) if hasattr(self, 'memlog') else super()._grow_right(r)


Lx, c, D = (int(v) for v in sys.argv[1:4]); lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat)
nup = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else lat.N//2; sweeps = int(sys.argv[5]) if len(sys.argv) > 5 else 3; outf = sys.argv[6] if len(sys.argv) > 6 else None
G = 2.0**30; nb = lambda lst: sum(int(a.nbytes) for a in lst if a is not None and hasattr(a, 'nbytes'))
with HOOK:
    E, eng, rows = eg.run(model, D, nup=nup, K=0, cls=Prof, sweeps=sweeps, log=None)
    eng._dev = None; resident = HOOK.cur
fam = dict(tensors=nb(eng.M), LH=nb(eng.LH), RH=nb(eng.RH), GL=nb(eng.GL), GR=nb(eng.GR))
nmax = max(int(p.size) for p in eng.plan); out = dict(lattice=lat.name, N=lat.N, layers=eng.n, D=D, nup=nup, sweeps_at_D=sweeps, E=E, largest_bond=max(eng.bond_sizes()),
                                                    bonds=eng.bond_sizes(), largest_tensor_numbers=nmax, resident_end_GiB=resident/G, families_GiB={k: v/G for k, v in fam.items()},
                                                    pool_peak_GiB=eng.peak/G, by_D={str(k): {ph: {kk: (vv if kk.endswith('at_peak') and not kk.startswith('resident') else vv/G) for kk, vv in rec.items()} for ph, rec in v.items()} for k, v in eng.memlog.items()},
                                                    krylov={str(k): v for k, v in eng.klog.items()})
m = eng.memlog[D]
print(f'{lat.name}, N = {lat.N}, {eng.n} layers, D = {D} (largest bond {out["largest_bond"]}), nup = {nup}, E = {E:.8f}')
print(f'  largest tensor {nmax:.3e} numbers = {8*nmax/G:.3f} GiB; 40 Krylov vectors of it {40*8*nmax/G:.2f} GiB')
print(f'  resident at the end {resident/G:.2f} GiB: ' + ', '.join(f'{k} {v/G:.2f}' for k, v in fam.items()) + f' (sum {sum(fam.values())/G:.2f})')
for ph in ('solve', 'move', 'grow'):
    print(f'  {ph:5s}: peak in use {m[ph]["peak"]/G:6.2f} GiB; largest rise above its start {m[ph]["above_start"]/G:6.2f} GiB (from {m[ph]["resident_at_start"]/G:.2f})')
kl = eng.klog[D]; big = [u for n, u in kl if n >= 0.8*nmax]; hist = np.bincount([u for n, u in kl], minlength=41)
print(f'  Krylov vectors (of 40 reserved) at D = {D}: in the solve with the largest rise {m["solve"].get("krylov_at_peak")}; over the {len(big)} solves of the largest tensors '
      f'(>= 0.8 of the largest): min {min(big)}, median {int(np.median(big))}, max {max(big)}; over all {len(kl)} solves: max {max(u for n, u in kl)}, '
      f'{int(hist[33:].sum())} solves used more than 32, {int(hist[40:].sum())} used all 40')
print(f'  peak in use over the run {HOOK.peak/G:.2f} GiB; peak kept by the pool {eng.peak/G:.2f} GiB', flush=True)
if outf:
    json.dump(out, open(outf, 'w'), indent=1)
