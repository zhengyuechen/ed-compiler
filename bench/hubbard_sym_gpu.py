#!/usr/bin/env python3
"""Hubbard model on one GPU (or CPU), one sector.
  hubbard_sym_gpu.py sym   <a1x,a1y,a2x,a2y> <U> <nup> <ndn> <kx/pi,ky/pi> <irrep> <flip> [cpu|gpu]
  hubbard_sym_gpu.py plain <Lx> <Ly> <U> <nup> <ndn>          (no lattice symmetry, species-product GPU kernel)
Environment: EDC_TOL (default 1e-12)."""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
tol = float(os.environ.get('EDC_TOL', '1e-12'))
if sys.argv[1] == 'sym':
    import edc_lattice as el, edc_hubbard_sym as hs
    a = [int(v) for v in sys.argv[2].split(',')]; U = float(sys.argv[3]); nu, nd = int(sys.argv[4]), int(sys.argv[5])
    k = tuple(float(v)*math.pi for v in sys.argv[6].split(',')); ir = sys.argv[7]; fl = int(sys.argv[8])
    dev = sys.argv[9] if len(sys.argv) > 9 else 'gpu'
    cl = el.Cluster((a[0], a[1]), (a[2], a[3])); bonds = hs.hubbard_bonds(cl, 1.0)
    t0 = time.perf_counter(); e = hs.HubbardSym(cl, bonds, U, nu, nd, k, ir, fl, device=dev)
    mv = e.time_matvec() if e.dim < 4e8 else None          # large sectors: per-step time from the Lanczos run
    E, it, t = e.ground_energy(tol=tol)
    out = dict(mode='sym', device=dev, cluster=a, N=cl.N, U=U, nup=nu, ndn=nd, k_over_pi=[k[0]/math.pi, k[1]/math.pi], irrep=ir,
               flip=fl, group_order=e.group_order, real=e.real, dim=e.dim, up_orbits=e.nq, setup_seconds=e.setup_seconds,
               device_gib=e.device_gib, matvec_seconds=mv, iterations=it, lanczos_seconds=t, seconds_per_step=t/it, E=E, E_per_site=E/cl.N, tol=tol)
else:
    import edc_hubbard as eh, edc_hubbard_gpu as hg
    Lx, Ly = int(sys.argv[2]), int(sys.argv[3]); U = float(sys.argv[4]); nu, nd = int(sys.argv[5]), int(sys.argv[6])
    m = eh.square(Lx, Ly, 1.0, U)
    e = hg.HubbardGPU(m, nu, nd); mv = e.time_matvec(); E, it, t = e.ground_energy(tol=tol)
    out = dict(mode='plain', Lx=Lx, Ly=Ly, N=Lx*Ly, U=U, nup=nu, ndn=nd, dim=e.dim, setup_seconds=e.setup_seconds,
               matvec_seconds=mv, iterations=it, lanczos_seconds=t, E=E, E_per_site=E/(Lx*Ly), tol=tol)
print(json.dumps(out), flush=True)
