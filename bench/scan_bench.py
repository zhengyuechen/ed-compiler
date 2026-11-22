#!/usr/bin/env python3
"""J2 scan on one sector: cold (fresh random start, engine rebuilt per point) vs retune + warm start.
Usage: scan_bench.py <kind|square> <A1x,A1y,A2x,A2y> <k1/pi,k2/pi> <irrep> <flip> <device> <J2 list, comma separated>"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_lattices as L, edc_lattice as el, edc_sub
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[3].split(',')); ir = sys.argv[4]; flip = int(sys.argv[5]); device = sys.argv[6]
J2s = [float(v) for v in sys.argv[7].split(',')]
if kind == 'square':
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); make = lambda J2: cl.j1j2(1.0, J2)
else:
    cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); make = lambda J2: cl.heisenberg(1.0, J2)
t0 = time.perf_counter(); cold = []
for J2 in J2s:
    e = edc_sub.SubEngine(make(J2), cl.N//2, cl, k, ir, flip, device=device)
    E, it, _ = e.ground_energy(tol=1e-12); cold.append((E, it)); del e
tc = time.perf_counter()-t0
t0 = time.perf_counter()
e = edc_sub.SubEngine(make(J2s[0]), cl.N//2, cl, k, ir, flip, device=device)
warm = e.scan([make(J2) for J2 in J2s], tol=1e-12, warm=True)
tw = time.perf_counter()-t0
print(json.dumps(dict(N=cl.N, dim=e.dim, stored=getattr(e, '_st', None) is not None, J2=J2s, cold_seconds=tc, warm_seconds=tw,
                      cold_iterations=[c[1] for c in cold], warm_iterations=[w['iterations'] for w in warm],
                      max_dE=max(abs(c[0]-w['E']) for c, w in zip(cold, warm)))), flush=True)
