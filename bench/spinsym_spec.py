#!/usr/bin/env python3
"""Write a spin-1/2 Heisenberg sector description for bench/spinsym_xdiag.jl (plain text, 1-based sites):
  line 1: N nup;  'B i j J' per bond;  'P c p_1 ... p_N' per group element (real character c), site i -> p_i.
Usage: spinsym_spec.py <kind> <A1x,A1y,A2x,A2y> <J1[,J2]> <out>      (Gamma point, trivial irrep, S_z = 0)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_lattices as L
kind = sys.argv[1]; a = [int(v) for v in sys.argv[2].split(',')]; J = [float(v) for v in sys.argv[3].split(',')]
cl = L.LatticeCluster(kind, (a[0], a[1]), (a[2], a[3])); m = cl.heisenberg(*J)
perms, chars, ops = cl.group((0.0, 0.0), 'X0')
with open(sys.argv[4], 'w') as f:
    f.write(f'{cl.N} {cl.N//2}\n')
    for (i, j), (jxy, jz) in sorted(m.bonds.items()):
        assert jxy == jz
        f.write(f'B {i+1} {j+1} {jxy}\n')
    for p, c in zip(perms, chars):
        f.write('P ' + str(complex(c).real) + ' ' + ' '.join(str(x+1) for x in p) + '\n')
print(cl.N, len(perms), ops)
