#!/usr/bin/env python3
"""Turn end-to-end JSON lines into a comparison table (time to ground-state energy)."""
import sys, json
from collections import defaultdict
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
by = defaultdict(dict)
for r in rows:
    tag = r['code'] + ('-stored' if r.get('stored') else '')
    by[r['model']][(tag, int(r['threads']))] = r
print('| model | dim | code | threads | setup s | iterations | Lanczos s | s / iteration | total s | E0 |')
print('|---|---|---|---|---|---|---|---|---|---|')
for model, d in by.items():
    for (tag, nt), r in sorted(d.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        setup = r['setup_seconds'] + r.get('stored_build_seconds', 0.0)
        print(f"| {model} | {r['dim']:,} | {tag} | {nt} | {setup:.2f} | {r['iterations']} | {r['lanczos_seconds']:.2f} | "
              f"{r['seconds_per_iteration']*1e3:.1f} ms | {setup + r['lanczos_seconds']:.2f} | {r['E0']:.10f} |")
    # speed-up summary at equal threads
    for nt in (8, 1):
        a, b = d.get(('edc', nt)), d.get(('xdiag', nt))
        if a and b:
            ta = a['setup_seconds'] + a['lanczos_seconds']; tb = b['setup_seconds'] + b['lanczos_seconds']
            print(f"| {model} | | **edc vs XDiag, {nt} thr** | | | | | {b['seconds_per_iteration']/a['seconds_per_iteration']:.1f}× per iteration | "
                  f"{tb/ta:.1f}× total | ΔE {abs(a['E0']-b['E0']):.1e} |")
