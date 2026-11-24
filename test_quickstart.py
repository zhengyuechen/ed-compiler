#!/usr/bin/env python3
"""The quick start of README.md: its model is examples/kagome12.edc, its energy is the published one and that of a dense
diagonalisation written without the engine, and pyproject.toml installs every edc module."""
import glob, itertools, json, re
import numpy as np
import edc_cli as cli

REF, TOL = -5.444875216, 1e-9         # Läuchli, Sudan, Sørensen, PRB 83, 212401 (2011), Table 1, N = 12; one unit of its last digit


def kagome12_bonds():
    """The 24 nearest-neighbour bonds of the 2 x 2 kagome torus from its geometry: sites A, B = A + a1/2, C = A + a2/2 of a cell."""
    site = lambda x, y, s: 3*((x % 2)+2*(y % 2))+s
    bonds = []
    for x, y in itertools.product(range(2), repeat=2):
        A, B, C = (site(x, y, s) for s in range(3))               # the triangle of the cell, then the triangle that points down
        bonds += [(A, B), (A, C), (B, C), (B, site(x+1, y, 0)), (C, site(x, y+1, 0)), (B, site(x+1, y-1, 2))]
    return bonds


def dense_ground_energy(bonds, n, nup):
    """Lowest eigenvalue of the sum of S_i . S_j over bonds among the states with nup up spins: a dense matrix, no symmetry."""
    st = np.array(sorted(sum(1 << i for i in c) for c in itertools.combinations(range(n), nup))); rows = np.arange(len(st))
    H = np.zeros((len(st), len(st)))
    for i, j in bonds:
        bi, bj = (st >> i) & 1, (st >> j) & 1
        H[rows, rows] += np.where(bi == bj, 0.25, -0.25)
        m = np.flatnonzero(bi != bj); H[m, np.searchsorted(st, st[m] ^ ((1 << i) | (1 << j)))] += 0.5
    return float(np.linalg.eigvalsh(H)[0])


statements = lambda text: [line.split('#', 1)[0].split() for line in text.splitlines() if line.split('#', 1)[0].strip()]
readme = open('README.md').read()
model = re.search(r"cat > kagome12\.edc <<'EOF'\n(.*?)\nEOF\n", readme, re.S).group(1)
assert statements(model) == statements(open('examples/kagome12.edc').read()), 'README quick start and examples/kagome12.edc differ'

bonds = kagome12_bonds()
assert len({frozenset(b) for b in bonds}) == 24 and all(sum(s in b for b in bonds) == 4 for s in range(12))
dense = dense_ground_energy(bonds, 12, 6)
spec = cli.parse(model); kind, cl = cli.cluster(spec)
assert cl.N == 12 and len(cli.spin_model(spec, kind, cl).bonds) == 24
r = cli.run(spec)
assert r['dim'] == 10 and abs(r['E']-REF) < TOL and abs(r['E']-dense) < 1e-10, (r['E'], REF, dense)
shown = json.loads(re.search(r'```json\n(\{"lattice": "kagome".*?\})\n```', readme).group(1))
assert {k: v for k, v in shown.items() if k not in ('E', 'seconds')} == {k: v for k, v in r.items() if k not in ('E', 'seconds')} and abs(shown['E']-r['E']) < 1e-10
every = cli.run(cli.parse(model.replace('sector Sz=0 k=0,0 irrep=X2 flip=+1', 'sector Sz=0 all')))
assert every['sum_mult_dim'] == every['full_dim'] == 924 and abs(every['ground_state']['E']-r['E']) < 1e-10

listed = re.findall(r'"(edc\w*)"', re.search(r'py-modules = \[(.*?)\]', open('pyproject.toml').read(), re.S).group(1))
assert sorted(listed) == sorted(p[:-3] for p in glob.glob('edc*.py')), 'pyproject.toml: py-modules and the edc*.py files differ'
print(f'quick start: kagome 12 E = {r["E"]:.12f}, published {REF} (within {abs(r["E"]-REF):.1e}), dense {dense:.12f} '
      f'(within {abs(r["E"]-dense):.1e}), complete scan 924 states, {len(listed)} modules installed: ok')
