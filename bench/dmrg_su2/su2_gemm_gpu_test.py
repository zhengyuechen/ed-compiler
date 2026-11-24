#!/usr/bin/env python3
"""(j, k) SU(2) layer DMRG on the GPU (su2_gpu, version 1): its moves between layers run the CPU reference code of
su2_dmrg.  This switches the contractions of that code from numpy's einsum to chained BLAS products (su2_gemm.install)
before a chosen sweep and prints, per sweep, the energy and the time in solves, moves and environments.

    su2_gemm_gpu_test.py Lx c D S2 K FROM [out.json]     FROM = 0: einsum throughout; n >= 1: su2_gemm from sweep n on

Two runs with the same seed are bit-identical until the sweep where one of them switches, so that sweep starts from the
same state in both and its times and energy can be compared directly."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import edc_dmrg as ed, su2_dmrg as sd, su2_gpu as sg, su2_gemm

Lx, c, D, S2, K, FROM = (int(v) for v in sys.argv[1:7]); outf = sys.argv[7] if len(sys.argv) > 7 else None
lat = ed.kagome_cylinder(Lx, c); model = sd.SU2Model(lat); eng = sg.SU2GPU1(model, S2, K); print(f'{lat.name}, N = {lat.N}, 2S = {S2}, K = {K}, {D} multiplets, su2_gemm from sweep {FROM or "never"}', flush=True)
t0 = time.perf_counter(); rows = []
sched = [(max(min(D, 64), D//4), 1e-2, 3e-5, 1e-2), (max(min(D, 64), D//2), 1e-2, 3e-5, 1e-2)] + [(D, al, 3e-5 if al > 0 else 1e-6, 0.0) for al in sd.ALPHAS]
for sw, (Dw, al, tl, noise) in enumerate(sched):
    if FROM and sw + 1 == FROM:
        su2_gemm.install(sd)
    ts = time.perf_counter(); s0 = dict(eng.stats); E = eng.sweep(Dw, al, tl, noise); te = time.perf_counter()
    row = dict(sweep=sw+1, D=Dw, expansion=al, E=E, seconds=te-ts, elapsed=te-t0, solve=eng.stats['t_solve']-s0['t_solve'], move=eng.stats['t_move']-s0['t_move'], env=eng.stats['t_env']-s0['t_env'],
               products=eng.stats['products']-s0['products'], multiplets=max(eng.bond_multiplets()), kernel='gemm' if FROM and sw + 1 >= FROM else 'einsum'); rows.append(row)
    print(f'  sweep {sw+1:2d} D={Dw:5d} exp={al:.0e} [{row["kernel"]:6s}]: E = {E:.12f} | {te-ts:7.2f} s = solve {row["solve"]:6.2f} + move {row["move"]:6.2f} + env {row["env"]:5.2f} | elapsed {te-t0:7.1f} | products {row["products"]}', flush=True)
if outf:
    json.dump(dict(lattice=lat.name, D=D, S2=S2, K=K, gemm_from=FROM, sweeps=rows), open(outf, 'w'), indent=1)
