#!/usr/bin/env python3
"""Parameter scan as one family (edc_scan.reduced_basis_scan) against the engine's point-by-point scan (SubEngine.scan; the
standard is the safe one, every point from a random vector), same engine, same machine.  J1-J2 Heisenberg model on an Lx x Ly torus, sector K = (0,0), A1, flip +1,
J2/J1 from lo to hi in P points.
Usage: scan_family.py Lx Ly P [cpu|gpu] [lo hi] [tol]     (env SKIP_BASELINE=1: family only; BASELINES, FAMILY: start modes;
       KX, KY, IRREP, FLIP: sector)"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, edc, edc_lattice as el, edc_sub, edc_scan
Lx, Ly, P = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
lo, hi = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (0.0, 1.0); tol = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-10
cl = el.Cluster((Lx, 0), (0, Ly)); grid = np.linspace(lo, hi, P)


def model(j2):                                                   # J2 bonds are present at J2 = 0 as well (same structure)
    m = edc.SpinHalfModel(cl.N); m.keep_zero = True
    for s, (x, y) in enumerate(cl.coord):
        m.heisenberg(s, cl.site(x+1, y), 1.0); m.heisenberg(s, cl.site(x, y+1), 1.0)
        m.heisenberg(s, cl.site(x+1, y+1), float(j2)); m.heisenberg(s, cl.site(x+1, y-1), float(j2))
    return m


def sync():
    if device == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()


models = [model(j) for j in grid]
import math
kx, ky = int(os.environ.get('KX', 0)), int(os.environ.get('KY', 0)); irrep = os.environ.get('IRREP', 'A1'); flip = int(os.environ.get('FLIP', 1))
build = lambda: edc_sub.SubEngine(model(0.5), cl.N//2, cl, (2*math.pi*kx/Lx, 2*math.pi*ky/Ly), irrep, flip, device=device, stored=True)
out = dict(torus=f'{Lx}x{Ly}', device=device, points=P, range=[lo, hi], tol=tol, k=[kx, ky], irrep=irrep, flip=flip)
sync(); t = time.perf_counter(); e = build(); sync(); out.update(dim=int(e.dim), real=bool(e.real), setup_seconds=time.perf_counter()-t); del e
log = lambda s: print(s, file=sys.stderr, flush=True)
ref = None
if not os.environ.get('SKIP_BASELINE'):                          # the standard: every point from a random vector
    for w in ['random']+os.environ.get('BASELINES', 'mixed,guess').split(','):
        if not w:
            continue
        e = build(); e.retune(models[0]); sync(); t = time.perf_counter(); rec = e.scan(models, tol=1e-12, warm=False if w == 'random' else w); sync()
        sec = time.perf_counter()-t; E = np.array([float(np.atleast_1d(x['E'])[0]) for x in rec])
        if ref is None:
            ref = E; out['E_first_last'] = [float(E[0]), float(E[-1])]
        out[f'point_by_point_{w}'] = dict(seconds=sec, products=sum(int(x['iterations'])+int(x.get('replay_iterations', 0)) for x in rec),
                                          largest_difference_to_random_start=float(np.abs(E-ref).max()))
        del e
for st in os.environ.get('FAMILY', 'random,mixed').split(','):
    e = build(); sync(); t = time.perf_counter(); r = edc_scan.reduced_basis_scan(e, models, tol=tol, start=st, certify=bool(os.environ.get('CERTIFY')), log=log); sync()
    r.pop('_basis'); sec = time.perf_counter()-t; Ef = np.array(r['E'])
    out[f'family_{st}'] = dict(seconds=sec, products=r['products'], lanczos_products=r['lanczos_products'], replay_products=r['replay_products'],
                               term_products=r['term_products'], solves=r['solves'], basis=r['basis'], terms=r['terms'], retunes=r['retunes'],
                               largest_error_estimate=float(max(r['error_estimate'])), audit=[float(d) for _, d in r['audit']],
                               trained_J2=[float(grid[i]) for i in r['trained']])
    if ref is not None:
        out[f'family_{st}'].update(largest_difference_to_random_start=float(np.abs(Ef-ref).max()),
                                   speedup_vs_point_by_point_random=out['point_by_point_random']['seconds']/sec)
    del e
print(json.dumps(out))
