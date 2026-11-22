#!/usr/bin/env python3
"""Product time with the ordinary and the incremental representative search (edc_sub incremental=False/True), same
sector, same device; checks that the two products agree.
Usage: inc_bench.py <kind|square> <A1x,A1y,A2x,A2y> <J1[,J2]> <k1/pi,k2/pi> <irrep> <flip> <device> [label]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_lattice as el, edc_sub
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]; flip = int(sys.argv[6]); device = sys.argv[7]
label = sys.argv[8] if len(sys.argv) > 8 else kind
if kind == 'square':
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); model = cl.j1j2(*J)
else:
    cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
out = dict(label=label, N=cl.N, device=device)
ys = []
for inc in (False, True):
    e = edc_sub.SubEngine(model, cl.N//2, cl, k, ir, flip, device=device, incremental=inc)
    out.update(dim=e.dim, group_order=e.group_order, cosets=e.ncos, real=e.real)
    out['incremental' if inc else 'ordinary'] = e.time_matvec(3)
    xp = e.xp
    rng = np.random.default_rng(1)
    x = rng.standard_normal(e.dim) if e.real else rng.standard_normal(e.dim)+1j*rng.standard_normal(e.dim)
    y = e.matvec(xp.asarray(x), xp.zeros(e.dim, x.dtype))
    ys.append(y.get() if xp is not np else y)
    del e
out['speedup'] = out['ordinary']/out['incremental']
out['max_diff'] = float(np.max(np.abs(ys[0]-ys[1])))
print(json.dumps(out), flush=True)
