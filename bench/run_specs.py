#!/usr/bin/env python3
"""Run several j1j2_gpu.py specs one after another, one output file each.
Usage: run_specs.py <outdir> "<a> <J2> <k> <irrep> <nup> <flip>" ..."""
import sys, os, subprocess
here = os.path.dirname(os.path.abspath(__file__)); out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    a, J2, k, ir, nup, flip = spec.split()
    tag = f"N40_J2-{J2}_k-{k.replace(',', '_')}_{ir}_nup{nup}_flip{flip}".replace('+', 'p').replace('-1', 'm1')
    with open(os.path.join(out, tag+'.json'), 'w') as f:
        r = subprocess.run([sys.executable, os.path.join(here, 'j1j2_gpu.py'), a, J2, k, ir, nup, flip], stdout=f)
    print(tag, 'exit', r.returncode, flush=True)
