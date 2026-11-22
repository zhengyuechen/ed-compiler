#!/usr/bin/env python3
"""J1-J2 square lattice, one symmetry sector on one GPU.
Usage: j1j2_gpu.py <a1x,a1y,a2x,a2y> <J2> <kx/pi,ky/pi> <irrep> [nup] [flip: 0, 1, -1]
Environment: EDC_TOL, EDC_MAXIT, EDC_SEED, EDC_ROTMIN (0/1), EDC_ENGINE (lean | sub), EDC_STORE (0 | 1, sub engine)"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cupy as cp, edc_lattice as el, edc_gpu_lean as gl
a = [int(v) for v in sys.argv[1].split(',')]; J2 = float(sys.argv[2])
k = tuple(float(v)*math.pi for v in sys.argv[3].split(',')); ir = sys.argv[4]
cl = el.Cluster((a[0], a[1]), (a[2], a[3])); nup = int(sys.argv[5]) if len(sys.argv) > 5 else cl.N//2
flip = int(sys.argv[6]) if len(sys.argv) > 6 else 0
t0 = time.perf_counter(); rm = os.environ.get('EDC_ROTMIN'); rm = None if rm is None else bool(int(rm))
if os.environ.get('EDC_ENGINE', 'lean') == 'sub':
    import edc_sub; e = edc_sub.SubEngine(cl.j1j2(1.0, J2), nup, cl, k, ir, flip, device='gpu', store_states=({'0': False, '1': True}.get(os.environ.get('EDC_STORE')))); e.rotmin = 'sub'; e.ops = None
else:
    e = gl.GPULean(cl.j1j2(1.0, J2), nup, cl, k, ir, flip=flip, rotmin=rm)
setup = time.perf_counter()-t0
mv = e.time_matvec() if e.dim < 1.2e9 else None; tol = float(os.environ.get('EDC_TOL', '1e-12')); E, it, t = e.ground_energy(tol=tol, maxit=int(os.environ.get('EDC_MAXIT', '1000')),
                                seed=int(os.environ.get('EDC_SEED', '42')))
print(json.dumps(dict(store=getattr(e, 'store', None), index_bits=getattr(e, 'index_bits', None), seconds_per_step=(t/it if it else None), cluster=a, N=cl.N, J2=J2, k_over_pi=[k[0]/math.pi, k[1]/math.pi], irrep=ir, flip=flip, rotmin=e.rotmin, ops=e.ops, nup=nup,
                      Sz=nup-cl.N/2, real=e.real, group_order=e.group_order, dim=e.dim, full_dim=e.full_dim, setup_seconds=setup,
                      device_gib=e.device_gib, matvec_seconds=mv, iterations=it, lanczos_seconds=t, E=E, tol=tol,
                      E_per_site=(E/cl.N if E is not None else None), seed=int(os.environ.get('EDC_SEED', '42')))), flush=True)
