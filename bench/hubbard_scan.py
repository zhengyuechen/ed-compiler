#!/usr/bin/env python3
"""Every symmetry sector of the Hubbard model on a cluster (edc_hubbard_sym.sectors: complete), lowest level of each.
Usage: hubbard_scan.py <a1x,a1y,a2x,a2y> <U> <nup> <ndn> <out.jsonl> [gpu|cpu]
One JSON line per sector, then a summary line with the ground state. Environment: EDC_TOL (default 1e-12)."""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_lattice as el, edc_hubbard_sym as hs
a = [int(v) for v in sys.argv[1].split(',')]; U = float(sys.argv[2]); nu, nd = int(sys.argv[3]), int(sys.argv[4])
out = sys.argv[5]; dev = sys.argv[6] if len(sys.argv) > 6 else 'gpu'; tol = float(os.environ.get('EDC_TOL', '1e-12'))
cl = el.Cluster((a[0], a[1]), (a[2], a[3])); bonds = hs.hubbard_bonds(cl, 1.0)
best, total = None, 0
with open(out, 'w') as f:
    for k, ir, fl, pops, mult in hs.sectors(cl, nu, nd):
        t0 = time.perf_counter()
        e = hs.HubbardSym(cl, bonds, U, nu, nd, k, ir, fl, device=dev, point_ops=pops)
        E, it, t = e.ground_energy(tol=tol) if e.dim else (None, 0, 0.0)
        total += mult*e.dim
        r = dict(cluster=a, N=cl.N, U=U, nup=nu, ndn=nd, k_over_pi=[k[0]/math.pi, k[1]/math.pi], irrep=ir, flip=fl,
                 point_ops=pops, multiplicity=mult, dim=e.dim, real=e.real, setup_seconds=e.setup_seconds, iterations=it,
                 lanczos_seconds=t, seconds_per_step=(t/it if it else None), E=E, wall=time.perf_counter()-t0)
        f.write(json.dumps(r)+'\n'); f.flush()
        if E is not None and (best is None or E < best['E']):
            best = r
        del e
        if dev == 'gpu':
            import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
    f.write(json.dumps(dict(summary=True, U=U, nup=nu, ndn=nd, full_dim=math.comb(cl.N, nu)*math.comb(cl.N, nd),
                            sum_mult_dim=total, ground_state=best))+'\n')
