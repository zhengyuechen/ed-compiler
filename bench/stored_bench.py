#!/usr/bin/env python3
"""On-the-fly product vs stored connectivity (SubEngine.build_stored): build time, memory, product time, and a full
ground-state run with each.  Usage: stored_bench.py <kind|square> <A1x,A1y,A2x,A2y> <J1[,J2]> <k1/pi,k2/pi> <irrep> <flip>
<device> [label] [gs 0|1]"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc_lattices as L, edc_lattice as el, edc_sub
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]; flip = int(sys.argv[6]); device = sys.argv[7]
label = sys.argv[8] if len(sys.argv) > 8 else kind; gs = len(sys.argv) > 9 and sys.argv[9] == '1'
if kind == 'square':
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); model = cl.j1j2(*J)
else:
    cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); model = cl.heisenberg(*J)
e = edc_sub.SubEngine(model, cl.N//2, cl, k, ir, flip, device=device, stored=False)    # on the fly first, then build
out = dict(label=label, N=cl.N, device=device, dim=e.dim, group_order=e.group_order, real=e.real)
out['onthefly_matvec'] = e.time_matvec(3)
if gs:
    E, it, t = e.ground_energy(tol=1e-12); out.update(onthefly_E=E, onthefly_iterations=it, onthefly_lanczos=t)
ok = e.build_stored()
out['stored'] = ok
if ok:
    out.update(build_seconds=e.stored_build_seconds, nnz=e.stored_nnz, gib=e.stored_bytes/2**30, nnz_per_row=e.stored_nnz/e.dim)
    out['stored_matvec'] = e.time_matvec(3)
    out['speedup'] = out['onthefly_matvec']/out['stored_matvec']
    if gs:
        E, it, t = e.ground_energy(tol=1e-12); out.update(stored_E=E, stored_iterations=it, stored_lanczos=t)
print(json.dumps(out), flush=True)
