#!/usr/bin/env python3
"""Checks of bench/levels_run.py:
  1. kagome 12, J1-J2 and XXZ: the lowest levels of every block, repeated by multiplicity, against a dense
     diagonalisation of each S_z sector built in the plain basis (no symmetry code);
  2. kagome 18: the family scan along J2 and along delta against one direct solve per point;
  3. kagome 18: the five lowest levels of every block against the complete block spectra (thermo_run.py --dense).
Usage: test_levels.py [cpu|gpu]"""
import sys, os, json, math, subprocess, tempfile, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edc, edc_lattices as L

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix='levels_test_')


def run(vec, out, *extra):
    if os.path.exists(out):
        os.remove(out)
    subprocess.run([sys.executable, os.path.join(here, 'bench', 'levels_run.py'), 'kagome', vec, '--out', out, '--device', device, *extra],
                   check=True, stderr=subprocess.DEVNULL)
    return [json.loads(l) for l in open(out) if l.strip()]


def sector_dense(cl, Js, delta, nup):
    n = cl.N
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(n), nup)]
    index = {s: a for a, s in enumerate(states)}
    H = np.zeros((len(states), len(states)))
    for sh, Jv in enumerate(Js, start=1):
        if not Jv:
            continue
        for i, j in cl.bonds(sh):
            for a, s in enumerate(states):
                bi, bj = (s >> i) & 1, (s >> j) & 1
                H[a, a] += Jv*delta*(0.25 if bi == bj else -0.25)
                if bi != bj:
                    H[index[s ^ (1 << i) ^ (1 << j)], a] += 0.5*Jv
    return np.linalg.eigvalsh(H)


# 1. kagome 12: lowest levels of every block against the plain-basis spectrum
cl = L.LatticeCluster('kagome', (2, 0), (0, 2))
for Js, delta in (((1.0, 0.3), 1.0), ((1.0,), 0.6), ((1.0, -0.2), 1.7)):
    recs = run('2,0,0,2', os.path.join(tmp, 'k12.jsonl'), '--J', ','.join(str(v) for v in Js), '--delta', str(delta), '--nev', '3')
    worst = 0.0
    for nup in range(6, 13):
        ref = sector_dense(cl, Js, delta, nup)
        got = [x for r in recs if r['nup'] == nup and r['dim'] for x in r['E']]
        assert sum(r['mult']*r['dim'] for r in recs if r['nup'] == nup) == math.comb(12, nup)
        assert abs(min(got)-ref[0]) < 1e-9, (Js, delta, nup, min(got), ref[0])
        for x in got:                                                 # every reported level is a level of the sector
            d = float(np.abs(ref-x).min()); worst = max(worst, d)
            assert d < 1e-8, (Js, delta, nup, x, d)
    print(f'kagome 12, J = {Js}, delta = {delta}: lowest levels of every block, largest distance to an exact level {worst:.1e}')

# 2. kagome 18: family scans against direct solves
vec = '2,-2,-2,-1'
for scan, fixed, direct in (('J2=-0.3:0.3:7', [], lambda v: ['--J', f'1,{v}']),
                            ('delta=0:1.5:7', ['--J', '1'], lambda v: ['--J', '1', '--delta', str(v)])):
    fam = run(vec, os.path.join(tmp, 'k18_scan.jsonl'), '--scan', scan, '--nup', '9:11', *fixed)
    values = fam[0]['scan']['values']
    worst = 0.0
    for p, v in enumerate(values):
        if scan.startswith('J2') and abs(v) < 1e-12:
            ref = run(vec, os.path.join(tmp, 'k18_pt.jsonl'), '--J', '1', '--nup', '9:11')
        else:
            ref = run(vec, os.path.join(tmp, 'k18_pt.jsonl'), *direct(v), '--nup', '9:11')
        refd = {(r['nup'], r['block']): r for r in ref}
        for r in fam:
            if not r['dim']:
                continue
            d = abs(r['E'][p]-refd[(r['nup'], r['block'])]['E'][0]); worst = max(worst, d)
            assert d < 2e-7, (scan, v, r['nup'], r['block'], r['E'][p], refd[(r['nup'], r['block'])]['E'][0])
    print(f'kagome 18, scan {scan}, S_z = 0, 1, 2, every block: family against direct solves, largest difference {worst:.1e}; '
          f'methods {sorted({r.get("method") for r in fam if r["dim"]})}')
# 3. kagome 18: several levels per block (Lanczos, blocks above 400 states) against the complete block spectra
dense_out = os.path.join(tmp, 'k18_dense.jsonl')
subprocess.run([sys.executable, os.path.join(here, 'bench', 'thermo_run.py'), 'kagome', vec, '1,0.2', '--out', dense_out, '--device', device,
                '--dense', '1000000', '--nup', '9:11'], check=True, stderr=subprocess.DEVNULL)
dense = {(r['nup'], r['block']): np.array(r['eigs']) for r in (json.loads(l) for l in open(dense_out)) if r['dim']}
for extra in ([], ['--mirror']):
    lev = run(vec, os.path.join(tmp, 'k18_lev.jsonl'), '--J', '1,0.2', '--nev', '5', '--nup', '9:11', *extra)
    worst, n = 0.0, 0
    for r in lev:
        if not r['dim']:
            continue
        if extra:                                                     # other blocks: compare with the whole sector spectrum
            ref = np.sort(np.concatenate([e for (nu, _), e in dense.items() if nu == r['nup']]))
            for x in r['E']:
                d = float(np.abs(ref-x).min()); worst = max(worst, d); assert d < 1e-8, (r['nup'], r['block'], x, d)
            continue
        ref = dense[(r['nup'], r['block'])]
        distinct = [ref[0]]
        for x in ref[1:]:
            if abs(x-distinct[-1]) > 1e-8:
                distinct.append(x)
        want = distinct[:min(5, len(distinct))]
        assert r.get('converged', True) and len(r['E']) == len(want), (r['nup'], r['block'], r['E'], want)
        d = float(np.abs(np.array(r['E'])-np.array(want)).max()); worst = max(worst, d); n += 1
        assert d < 1e-8, (r['nup'], r['block'], r['E'], want)
    print(f"kagome 18, five lowest levels of every block{' (mirror blocks: each is a level of the sector)' if extra else f' ({n} blocks) against the complete block spectra'}: largest difference {worst:.1e}")
print('levels: all checks passed', device)
