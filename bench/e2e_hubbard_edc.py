#!/usr/bin/env python3
"""Hubbard ground-state energy with edc_hubbard.  Usage: e2e_hubbard_edc.py chain:L:U:nup:ndn <threads> <kron|flat>"""
import sys, os, json, time, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_hubbard as eh
p = sys.argv[1].split(':'); nt = int(sys.argv[2]); kern = sys.argv[3]
U, nup, ndn = float(p[2]), int(p[3]), int(p[4])
m = eh.chain(int(p[1]), 1.0, U) if p[0] == 'chain' else eh.square(*map(int, p[1].split('x')), 1.0, U)
t0 = time.perf_counter(); e = eh.HubbardEngine(m, nup, ndn, threads=nt); setup = time.perf_counter()-t0
E, it, t = e.ground_energy(kernel=kern)
print(json.dumps(dict(code='edc-'+kern, model=sys.argv[1], threads=nt, dim=e.dim, setup_seconds=setup, iterations=it,
                      lanczos_seconds=t, seconds_per_iteration=t/it, E0=E,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform == 'darwin' else 2**20))))
