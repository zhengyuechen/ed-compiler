#!/usr/bin/env python3
"""Heisenberg sector with the sublattice-table engine on the CPU (OpenMP).
Usage: lattice_cpu.py <kind> <A1x,A1y,A2x,A2y> <J1[,J2]> <k1/pi,k2/pi> <irrep> [nup] [flip]"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_lattices as L, edc_sub
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
k = tuple(float(v)*math.pi for v in sys.argv[4].split(',')); ir = sys.argv[5]
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); nup = int(sys.argv[6]) if len(sys.argv) > 6 else cl.N//2
flip = int(sys.argv[7]) if len(sys.argv) > 7 else 0
e = edc_sub.SubEngine(cl.heisenberg(*J), nup, cl, k, ir, flip, device='cpu')
mv = e.time_matvec(); E, it, t = e.ground_energy(tol=float(os.environ.get('EDC_TOL', '1e-12')))
print(json.dumps(dict(code='edc_sub', kind=kind, cluster=a, N=cl.N, J=J, irrep=ir, flip=flip, threads=os.environ.get('OMP_NUM_THREADS'),
                      dim=e.dim, index_bits=e.index_bits, w=e.w, K=e.nK, cosets=e.ncos, setup_seconds=e.setup_seconds,
                      matvec_seconds=mv, iterations=it, lanczos_seconds=t, seconds_per_step=t/it, E=E, E_per_site=E/cl.N)), flush=True)
