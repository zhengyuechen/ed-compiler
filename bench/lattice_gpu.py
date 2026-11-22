#!/usr/bin/env python3
"""Heisenberg model on a triangular / honeycomb / kagome / square cluster, one sector on one GPU.
Usage: lattice_gpu.py <kind> <A1x,A1y,A2x,A2y> <J1[,J2,...]> <k1/pi,k2/pi> <irrep X0..> [nup] [flip]
Momenta are lattice-reduced (phase per step along a1, a2). Environment: EDC_TOL, EDC_ROTMIN, EDC_ENGINE (lean | sub), EDC_STORE (0 | 1, sub engine), EDC_DEVICES (e.g. 0,1,2,3: several GPUs)."""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_lattices as L, edc_gpu_lean as gl
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); nup = int(sys.argv[6]) if len(sys.argv) > 6 else cl.N//2
flip = int(sys.argv[7]) if len(sys.argv) > 7 else 0
rm = os.environ.get('EDC_ROTMIN'); rm = None if rm is None else bool(int(rm))
t0 = time.perf_counter()
if os.environ.get('EDC_ENGINE', 'lean') == 'sub':
    import edc_sub; e = edc_sub.SubEngine(cl.heisenberg(*J), nup, cl, k, ir, flip, device='gpu', store_states=({'0': False, '1': True}.get(os.environ.get('EDC_STORE'))),
                               devices=([int(v) for v in os.environ['EDC_DEVICES'].split(',')] if os.environ.get('EDC_DEVICES') else None)); e.rotmin = 'sub'; e.ops = None
else:
    e = gl.GPULean(cl.heisenberg(*J), nup, cl, k, ir, flip=flip, rotmin=rm)
setup = time.perf_counter()-t0
mv = e.time_matvec() if e.dim < 1.2e9 or getattr(e, 'devices', None) else None; E, it, t = e.ground_energy(tol=float(os.environ.get('EDC_TOL', '1e-12')))
print(json.dumps(dict(store=getattr(e, 'store', None), index_bits=getattr(e, 'index_bits', None), seconds_per_step=(t/it if it else None), kind=kind, cluster=a, N=cl.N, J=J, k_over_pi=[k[0]/math.pi, k[1]/math.pi], irrep=ir, chars={o: str(c) for o, c in cl.irrep_table(k)[ir].items()},
                      flip=flip, nup=nup, rotmin=e.rotmin, real=e.real, group_order=e.group_order, dim=e.dim, full_dim=e.full_dim,
                      setup_seconds=setup, device_gib=e.device_gib, matvec_seconds=mv, iterations=it, lanczos_seconds=t, E=E,
                      E_per_site=(E/cl.N if E is not None else None))), flush=True)
