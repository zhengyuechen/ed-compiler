#!/usr/bin/env python3
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cupy as cp, edc_hubbard as eh, edc_hubbard_gpu as hg
p = sys.argv[1].split(':'); U, nup, ndn = float(p[2]), int(p[3]), int(p[4])
m = eh.chain(int(p[1]), 1.0, U) if p[0] == 'chain' else eh.square(*map(int, p[1].split('x')), 1.0, U)
e = hg.HubbardGPU(m, nup, ndn); mv = e.time_matvec(); E, it, t = e.ground_energy()
print(json.dumps(dict(code='edc-gpu', model=sys.argv[1], threads=0, dim=e.dim, setup_seconds=e.setup_seconds, matvec_seconds=mv,
                      iterations=it, lanczos_seconds=t, seconds_per_iteration=t/it, E0=E)), flush=True)
