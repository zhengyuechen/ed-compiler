#!/usr/bin/env python3
"""Write a Hubbard sector description for bench/hubsym_xdiag.jl (plain text, 1-based sites):
  line 1: L nup ndn U
  'B i j' per bond (hopping t = 1), 'P c p_1 ... p_L' per group element (character c, real), site i -> p_i.
Usage: hubsym_spec.py <a1x,a1y,a2x,a2y> <U> <nup> <ndn> <out>     (Gamma point, A1 of the full point group)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edc_lattice as el, edc_hubbard_sym as hs
a = [int(v) for v in sys.argv[1].split(',')]; U = float(sys.argv[2]); nu, nd = int(sys.argv[3]), int(sys.argv[4])
cl = el.Cluster((a[0], a[1]), (a[2], a[3]))
perms, chars, ops = cl.group((0.0, 0.0), 'A1')
with open(sys.argv[5], 'w') as f:
    f.write(f'{cl.N} {nu} {nd} {U}\n')
    for (i, j), t in sorted(hs.hubbard_bonds(cl, 1.0).items()):
        assert t == 1.0
        f.write(f'B {i+1} {j+1}\n')
    for p, c in zip(perms, chars):
        assert abs(c.imag if isinstance(c, complex) else 0) < 1e-12
        f.write('P ' + str(float(complex(c).real)) + ' ' + ' '.join(str(x+1) for x in p) + '\n')
print(cl.N, len(perms), ops)
