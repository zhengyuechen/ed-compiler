#!/usr/bin/env python3
"""Long computation, both engines: set-up, one ground state, then NSTEP steps of the three-term recursion
v_{n+1} = 2 (H / s) v_n - v_{n-1}  (the work of a spectral function, a density of states or a time evolution by Chebyshev
polynomials: one product and two vector updates per step).  Same torus, same sector, same device.
Usage: rows_long.py cpu|gpu NSTEP LxxLy [LxxLy ...]"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc_lattice as el, edc_sub, edc_rows
device = sys.argv[1]; NSTEP = int(sys.argv[2])
if device == 'gpu':
    import cupy as xp
    sync = lambda: xp.cuda.Device().synchronize()
else:
    xp = np; sync = lambda: None

def recursion(matvec, dim, scale):
    v0 = xp.asarray(np.random.default_rng(5).standard_normal(dim)); v0 /= xp.linalg.norm(v0)
    v1 = xp.zeros(dim); w = xp.zeros(dim); matvec(v0, v1); v1 *= 1.0/scale; sync(); t0 = time.perf_counter()
    for n in range(NSTEP):
        matvec(v1, w); w *= 2.0/scale; w -= v0; v0, v1, w = v1, w, v0
    sync(); t = time.perf_counter()-t0
    return t, float(xp.linalg.norm(v1))

def run(tag, group, build, solve, matvec_of):
    t0 = time.perf_counter(); e = build(); sync(); t_setup = time.perf_counter()-t0
    t0 = time.perf_counter(); E0, products = solve(e); sync(); t_solve = time.perf_counter()-t0
    t_rec, nrm = recursion(matvec_of(e), int(e.dim), 1.05*abs(E0))
    print(json.dumps(dict(engine=tag, group=group, torus=torus, device=device, dim=int(e.dim), setup_seconds=t_setup, solve_seconds=t_solve,
                          products=int(products), E0=E0, steps=NSTEP, recursion_seconds=t_rec, per_step_ms=1e3*t_rec/NSTEP, final_norm=nrm)), flush=True)

for torus in sys.argv[3:]:
    Lx, Ly = [int(v) for v in torus.split('x')]
    cl = el.Cluster((Lx, 0), (0, Ly)); flip = 1 if (cl.N//2) % 2 == 0 else -1; nev = {(6, 5): 3}.get((Lx, Ly), 1)
    def sz_solve(e):
        r = e.ground_energy(tol=1e-12, nev=nev)
        return (float(r[0][-1]) if nev > 1 else float(r[0])), r[1]
    def rows_solve(e):
        en, products, ts, res = e.ground_energy(nev=1, tol=1e-12)
        return en[0], products
    for rep in range(int(os.environ.get('PASSES', 2))):                                        # second pass: warm caches
        run('rows', 'translations x D2', lambda: edc_rows.RowsEngine(Lx, Ly, device=device), rows_solve, lambda e: (lambda x, y: e.matvec(x, y)))
    for rep in range(int(os.environ.get('PASSES', 2))):
        run('sz', 'all symmetries', lambda: edc_sub.SubEngine(cl.j1j2(1.0), cl.N//2, cl, (0.0, 0.0), 'A1', flip, device=device), sz_solve,
            lambda e: (lambda x, y: e.matvec(x, y)))
