#!/usr/bin/env python3
"""Checks of bench/thermo_run.py + bench/thermo_post.py against references that share no code with the engines:
  1. kagome 12, J1-J2: every eigenvalue of every S_z sector (blocks diagonalised exactly, repeated by their multiplicity)
     against a dense diagonalisation of the full 2^12 Hamiltonian;
  2. kagome 18, J1-J2: Tr H^p (p = 1..4) of every S_z sector from the block spectra against traces of a sparse matrix
     built in the plain S_z basis;
  3. kagome 18: finite-temperature Lanczos (few steps, few vectors) against the exact block spectra.
Usage: test_thermo.py [cpu|gpu]"""
import sys, os, json, math, subprocess, tempfile, itertools
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edc_lattices as L

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix='thermo_test_')


def run(kind, vec, J, out, *extra):
    if os.path.exists(out):
        os.remove(out)
    subprocess.run([sys.executable, os.path.join(here, 'bench', 'thermo_run.py'), kind, vec, J, '--out', out, '--device', device, *extra],
                   check=True, stderr=subprocess.DEVNULL)
    return [json.loads(l) for l in open(out) if l.strip()]


def post(out_json, files, *extra):
    subprocess.run([sys.executable, os.path.join(here, 'bench', 'thermo_post.py'), out_json, *files, *extra], check=True,
                   stdout=subprocess.DEVNULL)
    return json.load(open(out_json))


def sector_matrix(model, nup):
    """Sparse H in the plain basis of all states with nup up spins (bit i = 1: spin i up)."""
    n = model.n
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(n), nup)]
    index = {s: a for a, s in enumerate(states)}
    rows, cols, vals = [], [], []
    for a, s in enumerate(states):
        d = 0.0
        for (i, j), (jxy, jz) in model.bonds.items():
            bi, bj = (s >> i) & 1, (s >> j) & 1
            d += jz*(0.25 if bi == bj else -0.25)
            if bi != bj and jxy:
                rows.append(index[s ^ (1 << i) ^ (1 << j)]); cols.append(a); vals.append(0.5*jxy)
        rows.append(a); cols.append(a); vals.append(d)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(states), len(states)))


worst = 0.0
# 1. kagome 12: complete spectrum
cl = L.LatticeCluster('kagome', (2, 0), (0, 2)); Jv = (1.0, 0.3); model = cl.heisenberg(*Jv)
recs = run('kagome', '2,0,0,2', '1,0.3', os.path.join(tmp, 'k12.jsonl'), '--dense', '1000000')
for nup in range(6, 13):
    ref = np.linalg.eigvalsh(sector_matrix(model, nup).toarray())
    got = np.sort(np.concatenate([np.repeat(r['eigs'], r['mult']) for r in recs if r['nup'] == nup and r['dim']]))
    assert len(got) == len(ref) == math.comb(12, nup), (nup, len(got), len(ref))
    d = float(np.abs(got-ref).max()); worst = max(worst, d)
    assert d < 1e-10, (nup, d)
print(f'kagome 12, J1-J2 = {Jv}: all 2^12 levels, largest difference {worst:.1e}')

# 2. kagome 18: traces of powers
cl = L.LatticeCluster('kagome', (2, -2), (-2, -1)); Jv = (1.0, 0.2); model = cl.heisenberg(*Jv)
f18 = os.path.join(tmp, 'k18.jsonl')
recs = run('kagome', '2,-2,-2,-1', '1,0.2', f18, '--dense', '1000000')
wt = 0.0
for nup in range(9, 19):
    H = sector_matrix(model, nup); H2 = H @ H
    ref = [H.diagonal().sum(), H.multiply(H).sum(), H2.multiply(H).sum(), H2.multiply(H2).sum()]
    ev = [(np.array(r['eigs']), r['mult']) for r in recs if r['nup'] == nup and r['dim']]
    assert sum(m*len(e) for e, m in ev) == math.comb(18, nup)
    for p in range(1, 5):
        got = sum(m*(e**p).sum() for e, m in ev)
        d = abs(got-ref[p-1])/max(1.0, abs(ref[p-1])); wt = max(wt, d)
        assert d < 1e-10, (nup, p, got, ref[p-1])
print(f'kagome 18, J1-J2 = {Jv}: Tr H^p, p = 1..4, every S_z sector, largest relative difference {wt:.1e}')

# 3. kagome 18: FTLM against the exact spectra
exact = post(os.path.join(tmp, 'k18_exact.json'), [f18], '--T', '0.05:10:60', '--h', '0,1.0')
f18l = os.path.join(tmp, 'k18_ftlm.jsonl')
run('kagome', '2,-2,-2,-1', '1,0.2', f18l, '--dense', '150', '--NL', '80', '--R', '0:8')
ftlm = post(os.path.join(tmp, 'k18_ftlm.json'), [f18l], '--T', '0.05:10:60', '--h', '0,1.0')
T = np.array(exact['T'])
assert abs(ftlm['ln_total_weight_per_site']-math.log(2)) < 1e-12 and abs(exact['ln_total_weight_per_site']-math.log(2)) < 1e-12
vx = 3/16*sum(jz*jz for (_, jz) in model.bonds.values())/18
assert abs(exact['var_H_infinite_T_per_site']-vx) < 1e-10, (exact['var_H_infinite_T_per_site'], vx)
for h in ('0.0', '1.0'):
    for q, tol in (('C', 0.05), ('S', 0.02), ('chi', 0.02), ('m', 0.02), ('E', 0.01)):
        d = np.abs(np.array(ftlm['fields'][h][q])-np.array(exact['fields'][h][q]))
        hi_ = float(d[T >= 0.3].max()); lo_ = float(d.max())
        print(f'  h = {h} {q:3s}: FTLM - exact, largest for T >= 0.3: {hi_:.1e}; for T >= 0.05: {lo_:.1e}; '
              f'largest half difference {max(ftlm["fields"][h][q+"_half_diff"]):.1e}')
        assert hi_ < tol, (h, q, hi_)
print(f'kagome 18: var(H)/N at infinite T {exact["var_H_infinite_T_per_site"]:.10f} = 3/16 sum J^2 / N = {vx:.10f}')
print('thermo: all checks passed', device)
