#!/usr/bin/env python3
"""Zero-temperature analysis of levels_run.py (and thermo_run.py) output.

  plateaus <out.json> <scan.jsonl | levels.jsonl | thermo part files ...>
      E0(M) = lowest level over all blocks of the sector with N_up - N/2 = M, at every scan point (one point for a levels
      or thermo file).  Magnetisation process: M(h) minimises E0(M) - h M, i.e. the stable M are the vertices of the lower
      convex hull of E0(M) (E0(-M) = E0(M)).  For a stable M the field range is [h_lo, h_hi] with
      h_hi = (E0(M') - E0(M))/(M' - M) to the next vertex M' and h_lo the same to the previous one; width = h_hi - h_lo.
      Reported for every M, and separately for m = 2M/N = 1/9, 1/3, 5/9, 7/9 where N allows them.
  spectrum <out.json> <levels.jsonl> [<levels_mirror.jsonl>] --nup n [--count 40]
      The lowest levels of one S_z sector with momentum, multiplicity and point-group label (complete up to the energy
      printed: every block contributes its lowest few levels only).  With the mirror file the
      levels at momenta whose little group is not abelian get the full label (A1, A2, B1, B2, E1, E2 at Gamma for C6v;
      A1, A2, E at K for C3v) by matching energies between the rotation blocks and the mirror blocks.

Momentum names for kagome clusters: G = (0, 0); K = the two zone corners; M = the three edge midpoints; other stars are
named by their size and k/pi."""
import sys, json, math, cmath, argparse
import numpy as np


def read(files):
    recs = []
    for f in files:
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return recs


def kname(r):
    kx, ky = (round(v, 4) % 2 for v in r['k_over_pi'])
    if (kx, ky) == (0, 0):
        return 'G'
    if r['mult'] == 2:
        return 'K'
    if r['mult'] == 3:
        return 'M'
    return f'Q{r["mult"]}({kx:g},{ky:g})'


def rot_label(r):
    """Label of a rotation block from the character of the smallest rotation: its phase as a multiple of 2 pi / order."""
    ch = r.get('chars', {})
    rots = sorted(o for o in ch if o.startswith('R'))
    mirs = sorted(o for o in ch if o.startswith('S'))
    if rots:
        c = complex(*ch[rots[0]]); n = len(rots)+1
        q = round(cmath.phase(c)/(2*math.pi/n)) % n
        lab = f'C{n}:{q}'
        if mirs:
            lab += ' s' + ('+' if ch[mirs[0]][0] > 0 else '-')
        return lab
    if mirs:
        return 's+' if ch[mirs[0]][0] > 0 else 's-'
    return '1'


def e0_table(recs):
    """{nup: [E0 at every point]}, {nup: [label of the lowest block at every point]}, values, N."""
    N = recs[0]['N']
    by = {}
    for r in recs:
        if not r['dim']:
            continue
        if 'alphas' in r:                                               # thermo_run record: lowest Ritz value of the chains
            from scipy.linalg import eigh_tridiagonal
            E = [min(eigh_tridiagonal(np.array(a), np.array(b[:-1]), eigvals_only=True, select='i', select_range=(0, 0))[0]
                     for a, b in zip(r['alphas'], r['betas']))]
        elif 'eigs' in r:
            E = [min(r['eigs'])]
        elif 'scan' in r:
            E = r['E']
        else:
            E = [r['E'][0]]
        by.setdefault(r['nup'], []).append((np.array(E, float), f"{kname(r)} {rot_label(r) if 'chars' in r else r['irrep']}" + (f" f{r['flip']:+d}" if r.get('flip') else '')))
    values = next((r['scan']['values'] for r in recs if 'scan' in r), [None])
    E0, lab = {}, {}
    for nup, lst in by.items():
        A = np.array([e for e, _ in lst])
        i = A.argmin(axis=0)
        E0[nup] = A.min(axis=0); lab[nup] = [lst[j][1] for j in i]
    return E0, lab, values, N


def hull(Ms, Es):
    """Indices of the vertices of the lower convex hull of the points (M, E), M increasing."""
    h = []
    for i in range(len(Ms)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (Es[b]-Es[a])*(Ms[i]-Ms[a]) >= (Es[i]-Es[a])*(Ms[b]-Ms[a])-1e-12:
                h.pop()
            else:
                break
        h.append(i)
    return h


def process(E0, N, p):
    """Field ranges of every stable M at scan point p."""
    nups = sorted(E0)
    Ms = [n-N/2 for n in nups]; Es = [float(E0[n][p]) for n in nups]
    # mirror image for the lowest sector: M = 0 has h_lo = -h_hi; M = 1/2 has h_lo = 0
    Mx = ([-Ms[1]] if Ms[0] == 0 else [-Ms[0]])+Ms; Ex = ([Es[1]] if Ms[0] == 0 else [Es[0]])+Es
    hv = hull(Mx, Ex)
    out = {}
    for a, i in enumerate(hv):
        if i == 0:
            continue
        M = Mx[i]
        lo = (Ex[i]-Ex[hv[a-1]])/(Mx[i]-Mx[hv[a-1]])
        hi = (Ex[hv[a+1]]-Ex[i])/(Mx[hv[a+1]]-Mx[i]) if a+1 < len(hv) else float('inf')
        if M == 0:
            lo = 0.0
        out[M] = (max(lo, 0.0), hi)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('plateaus', 'spectrum')); ap.add_argument('out'); ap.add_argument('files', nargs='+')
    ap.add_argument('--nup', type=int); ap.add_argument('--count', type=int, default=40)
    args = ap.parse_args()
    recs = read(args.files if args.cmd == 'plateaus' else args.files[:1])
    if args.cmd == 'plateaus':
        E0, lab, values, N = e0_table(recs)
        P = len(values)
        res = dict(N=N, J=recs[0].get('J'), delta=recs[0].get('delta'), scan=next((r['scan']['name'] for r in recs if 'scan' in r), None),
                   values=values, M=[n-N/2 for n in sorted(E0)], E0={str(n-N/2): E0[n].tolist() for n in sorted(E0)},
                   lowest_block={str(n-N/2): lab[n] for n in sorted(E0)}, ranges=[], plateaus={})
        fr = {'1/9': N/18, '1/3': N/6, '5/9': 5*N/18, '7/9': 7*N/18}
        fr = {k: v for k, v in fr.items() if abs(v*2-round(v*2)) < 1e-9 and (v*2) % 2 == N % 2 and int(round(v+N/2)) in E0}
        for p in range(P):
            pr = process(E0, N, p)
            res['ranges'].append({str(M): list(v) for M, v in pr.items()})
            for name, M in fr.items():
                lo, hi = pr.get(M, (None, None))
                res['plateaus'].setdefault(name, []).append(dict(M=M, h_lo=lo, h_hi=hi, width=(hi-lo) if lo is not None else 0.0,
                                                               block=lab[int(round(M+N/2))][p]))
        json.dump(res, open(args.out, 'w'))
        print(f"N = {N}, {P} point(s){'' if values[0] is None else ' of '+res['scan']}; plateau fractions on this cluster: {list(fr)}")
        show = range(P) if P <= 8 else sorted({0, P//4, P//2, 3*P//4, P-1})
        for p in show:
            tag = '' if values[p] is None else f"{res['scan']} = {values[p]:+.3f}: "
            print('  '+tag+'; '.join(f"{n} [{d['h_lo']:.4f}, {d['h_hi']:.4f}] width {d['width']:.4f} ({d['block']})" if d['h_lo'] is not None
                                     else f"{n} not stable" for n, d in ((n, res['plateaus'][n][p]) for n in fr)))
    else:
        rot = [r for r in recs if r['nup'] == args.nup and r['dim']]
        mir = [r for r in read(args.files[1:2]) if r['nup'] == args.nup and r['dim']] if len(args.files) > 1 else []
        N = rot[0]['N']
        levels = []
        for r in rot:
            for E in r['E']:
                levels.append(dict(E=E, k=kname(r), mult=r['mult'], label=rot_label(r), dim=r['dim']))
        # mirror parities: levels of the mirror blocks at the same momentum with the same energy
        mlev = {}
        for r in mir:
            ch = r['chars']; mirs = sorted(o for o in ch if o.startswith('S'))
            par = ('+' if ch[mirs[0]][0] > 0 else '-') if mirs else ''
            r3 = ('+' if ch['R3'][0] > 0 else '-') if 'R3' in ch else ''
            for E in r['E']:
                mlev.setdefault(kname(r), []).append((E, par, r3))
        for l in levels:
            if l['label'].startswith('C') and ' s' not in l['label'] and l['k'] in mlev:
                n, q = (int(x) for x in l['label'][1:].split(':'))
                pars = sorted({par for E, par, r3 in mlev[l['k']] if abs(E-l['E']) < 1e-7*max(1.0, abs(l['E']))})
                real = q in (0, n//2) and n % 2 == 0 or q == 0
                if real and len(pars) == 1:
                    l['full'] = {(6, 0): 'A', (6, 3): 'B', (3, 0): 'A', (2, 0): 'A', (2, 1): 'B'}.get((n, q), '?')+('1' if pars[0] == '+' else '2')
                elif real and len(pars) == 2:
                    l['full'] = {(6, 0): 'A', (6, 3): 'B', (3, 0): 'A'}.get((n, q), '?')+'1+2 (degenerate or unresolved)'
                elif not real:
                    l['full'] = {(6, 1): 'E1', (6, 5): 'E1', (6, 2): 'E2', (6, 4): 'E2', (3, 1): 'E', (3, 2): 'E'}.get((n, q), '?')
        levels.sort(key=lambda l: l['E'])
        E0 = levels[0]['E']
        # a two-dimensional irrep appears in two rotation blocks with the same energy: list it once with its dimension
        merged = []
        for l in levels:
            lab = l.get('full', l['label'])
            m = next((x for x in merged if x['k'] == l['k'] and abs(x['E']-l['E']) < 1e-7*max(1.0, abs(l['E'])) and x['irrep'] == lab
                      and lab in ('E', 'E1', 'E2') and x['states'] == l['mult']), None)
            if m:
                m['states'] += l['mult']
            else:
                merged.append(dict(E=l['E'], dE=l['E']-E0, k=l['k'], irrep=lab, states=l['mult']))
        cum = 0
        for m in merged:
            cum += m['states']; m['cumulative'] = cum
        # the list holds every level of the sector below the highest level of the block that reaches least far up
        nev = max(len(r['E']) for r in rot)
        complete = min(max(r['E']) for r in rot if len(r['E']) == nev)-E0
        json.dump(dict(N=N, nup=args.nup, M=args.nup-N/2, J=rot[0].get('J'), delta=rot[0].get('delta'), E0=E0, blocks=len(rot),
                       complete_below=complete, levels=merged[:4*args.count]), open(args.out, 'w'))
        print(f'N = {N}, N_up = {args.nup} (M = {args.nup-N/2:g}), J = {rot[0].get("J")}, delta = {rot[0].get("delta")}: E0 = {E0:.9f}, E0/N = {E0/N:.8f}')
        print(f'   {len(rot)} blocks, {nev} levels each: complete below E - E0 = {complete:.4f}')
        print('   E - E0     momentum         irrep   states  cumulative')
        for m in merged[:args.count]:
            print(f"  {m['dE']:9.6f}  {m['k']:16s} {m['irrep']:7s} {m['states']:4d}   {m['cumulative']:5d}")
