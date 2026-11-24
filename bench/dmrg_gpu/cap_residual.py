#!/usr/bin/env python3
"""Residual of the layer solves that use up the 40 Krylov vectors (needs a GPU).

    cap_residual.py Lx c D [nup] [K] [out.json]      kagome YC(2c) cylinder with Lx rings, version 3 with the record of edc_dmrg_gpu3r

Per sweep: solves, how many used all 40 vectors, how many of those ended above the tolerance, and the largest
residual / tolerance; then the energy, the whole-run time and the peak the pool kept."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import edc_dmrg as ed, edc_dmrg_gpu as eg, edc_dmrg_gpu3r as e3r

Lx, c, D = (int(v) for v in sys.argv[1:4]); lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat)
nup = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else lat.N//2; K = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 0
outf = sys.argv[6] if len(sys.argv) > 6 else None
t0 = time.perf_counter(); E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=e3r.DMRGGPU3r, log=None); total = time.perf_counter() - t0; tab = e3r.cap_table(eng)
print(f'{lat.name}, N = {lat.N}, D = {D}, nup = {nup}, K = {K}: E = {E:.10f}, whole run {total:.1f} s, pool peak {eng.peak/2**30:.1f} GiB')
print('  sweep     D  tolerance  solves  used all 40  of those above tolerance  largest residual/tolerance  (largest tensors: above tolerance / solves)  E after the sweep')
for t, r in zip(tab, rows):
    print(f'  {t["sweep"]:5d} {t["D"]:5d}  {t["tolerance"]:.1e}  {t["solves"]:6d}  {t["used_all"]:11d}  {t["above_tolerance"]:23d}  {t["largest_ratio"]:26.1f}  '
          f'{t["largest_tensors_above_tolerance"]:20d} / {t["largest_tensors"]:<3d}  {r["E"]:.10f}')
last = tab[-1]; print(f'  last sweep: {last["above_tolerance"]} of {last["solves"]} solves ended above the tolerance; the final solve (layer 0) ended at {last["last_solve_ratio"]:.2f} of its tolerance', flush=True)
if outf:
    json.dump(dict(lattice=lat.name, N=lat.N, D=D, nup=nup, K=K, E=E, total_seconds=total, pool_peak_GiB=eng.peak/2**30, per_sweep=tab, sweep_energies=[r['E'] for r in rows],
                   sweep_elapsed=[r['elapsed'] for r in rows], solves=eng.reslog), open(outf, 'w'))
