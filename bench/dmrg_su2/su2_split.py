"""(j, k) SU(2) layer DMRG, CPU reference: per sweep the wall time split into layer solves, moves, environments.
usage: su2_split.py REPO Lx c D S2 K out.json [profile] [gemm]
gemm: operator pairs applied as chained BLAS products (su2_gemm.py), the only change."""
import sys, os, json, time, cProfile, pstats, io
REPO = sys.argv[1]; sys.path.insert(0, os.path.join(REPO, "bench", "dmrg_su2")); sys.path.insert(0, REPO)
import numpy as np, edc_dmrg as ed, su2_dmrg as su
Lx, c, D, S2, K = (int(v) for v in sys.argv[2:7]); out = sys.argv[7]; prof = 'profile' in sys.argv[8:]
if 'gemm' in sys.argv[8:]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import su2_gemm; su2_gemm.install(su); print('kernel: chained BLAS products', flush=True)
lat = ed.kagome_cylinder(Lx, c); model = su.SU2Model(lat); REF = -26.64456
eng = su.SU2DMRG(model, S2, K, seed=0); alphas = list(su.ALPHAS); tol = 3e-5; rows = []; t0 = time.perf_counter()
sched = [(max(min(D, 64), D//4), 1e-2, tol, 1e-2), (max(min(D, 64), D//2), 1e-2, tol, 1e-2)] + [(D, al, tol if al > 0 else tol/30, 0.0) for al in alphas]
pr = cProfile.Profile() if prof else None
for sw, (Dw, al, tl, noise) in enumerate(sched):
    s0 = dict(eng.stats); ts = time.perf_counter()
    if pr and Dw == D and al > 0: pr.enable()
    E = eng.sweep(Dw, al, tl, noise)
    if pr: pr.disable()
    te = time.perf_counter(); st = eng.stats
    rows.append(dict(sweep=sw+1, D=Dw, expansion=al, E=E, err=E-REF, seconds=te-ts, elapsed=te-t0, products=st['products']-s0['products'],
                     solve=st['t_solve']-s0['t_solve'], move=st['t_move']-s0['t_move'], env=st['t_env']-s0['t_env'],
                     multiplets=max(eng.bond_multiplets()), states=max(eng.bond_states())))
    r = rows[-1]; print(f"sweep {sw+1:2d} D={Dw:4d} exp={al:.0e} err {r['err']:.2e} | {r['seconds']:7.1f} s = solve {r['solve']:6.1f} + move {r['move']:6.1f} + env {r['env']:6.1f} | "
                        f"products {r['products']:4d} ({r['solve']/max(1,r['products'])*1e3:.0f} ms each) | cum {r['elapsed']:7.1f} s | {r['multiplets']} multiplets = {r['states']} states", flush=True)
json.dump(rows, open(out, 'w'), indent=1)
if pr:
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14); print(s.getvalue()[:3500])
