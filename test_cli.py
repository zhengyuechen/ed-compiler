#!/usr/bin/env python3
"""Model-language front end: the CPU examples give the known values and complete scans cover the whole space."""
import edc_cli as cli

def run(path):
    return cli.run(cli.parse(open(path).read()))

r = run('examples/square4x4_j1j2.edc')
assert abs(r['E']-(-8.457923351393)) < 1e-9, r['E']
r = run('examples/kagome12_all.edc')
assert r['sum_mult_dim'] == r['full_dim'] == 924 and abs(r['ground_state']['E']-(-5.444875216)) < 1e-9
r = run('examples/hubbard_triangular12.edc')
assert r['sum_mult_dim'] == r['full_dim'] == 48400
r = run('examples/kagome12_structure.edc')
assert abs(r['E']-(-5.444875216)) < 1e-9 and r['residual'] < 1e-12 and r['S(Qr=1,0)']['star'] == 6
print('cli examples: square 4x4 J1-J2, kagome 12 (Lauchli et al. 2011), triangular Hubbard 12 (complete scan), '
      'kagome 12 structure factor: ok')
