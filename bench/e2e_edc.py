#!/usr/bin/env python3
"""End-to-end ground-state energy with edc.  Usage: e2e_edc.py <model> <threads> [stored]"""
import sys, os, json, time, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edc, models
name, nt = sys.argv[1], int(sys.argv[2]); stored = len(sys.argv) > 3 and sys.argv[3] == 'stored'
m, g = models.build(name)
t0 = time.perf_counter()
e = edc.Engine(m, m.n//2, group=g, threads=nt)
setup = time.perf_counter()-t0
sb = e.build_stored() if stored else 0.0
E, it, t = e.ground_energy(tol=1e-12, stored=stored)
print(json.dumps(dict(code='edc', model=name, threads=nt, stored=stored, dim=e.dim, setup_seconds=setup,
                      compile_seconds=e.compile_seconds, stored_build_seconds=sb, iterations=it, lanczos_seconds=t,
                      seconds_per_iteration=t/it, E0=E, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform == 'darwin' else 2**20))))
