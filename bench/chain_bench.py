#!/usr/bin/env python3
"""Heisenberg chain (periodic, J = 1) at momentum zero with reflection (and optionally spin flip): the sectors of the
published single-node benchmarks of XDiag (arXiv:2505.02901 Table 1: U(1) + translations + reflection) and
lattice-symmetries (arXiv:2308.16712 Figs. 7, 9: + spin inversion).  Times the set-up and one product H x.
Usage: chain_bench.py N irrep(A1|A2) flip(0|1|-1) device(cpu|gpu) [reps]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc, edc_lattice as el, edc_sub
N, ir, flip, device = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 2
cl = el.Cluster((N, 0), (0, 1))
t0 = time.perf_counter()
e = edc_sub.SubEngine(edc.ring(N), N//2, cl, (0.0, 0.0), ir, flip, device=device, point_ops=['E', 'Mx'])
setup = time.perf_counter()-t0
mv = e.time_matvec(reps)
print(json.dumps(dict(code='edc_sub', N=N, irrep=ir, flip=flip, device=device, threads=os.environ.get('OMP_NUM_THREADS'),
                      dim=e.dim, group_order=e.group_order, w=e.w, cosets=e.ncos, real=e.real, setup_seconds=setup,
                      matvec_seconds=mv, seconds_per_state=mv/e.dim, device_gib=getattr(e, 'device_gib', None))), flush=True)
