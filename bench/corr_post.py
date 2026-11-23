#!/usr/bin/env python3
"""Tables from the files of bench/corr_full.py and bench/corr_sym.py.

    corr_post.py <directory> <out.json> [--top 4]

Files of the same cluster, N_up and J2 are sorted by energy; kinds with the same energy (within 1e-7) form one level,
whose matrices are the averages over its states (weights = number of states of each kind).  For every level and
operator set: the mean eigenvalue of the connected matrix (the variance of one operator), the leading patterns with
their momentum and irrep, and the largest eigenvalue at each momentum star."""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import edc_lattices as L
import corr_tools as ct
import corr_sym as cs

STATES = {'G:A1': 1, 'G:A2': 1, 'G:B1': 1, 'G:B2': 1, 'G:E1': 2, 'G:E2': 2, 'K:A1': 2, 'K:A2': 2, 'K:E': 4,
          'M:A1': 3, 'M:A2': 3, 'M:B1': 3, 'M:B2': 3, 'Q:e': 6, 'Q:o': 6}
SETS = (('z', 'Gz', 'mz'), ('zz', 'Gzz', 'ezz'), ('b', 'Gb', 'eb'), ('x', 'Gx', None))
_cache = {}


def context(A):
    key = tuple(A)
    if key not in _cache:
        cl = L.LatticeCluster('kagome', (A[0], A[1]), (A[2], A[3]))
        geo = ct.geometry(cl); G = ct.space_group(cl)
        _cache[key] = (cl, geo, {k: ct.OpSpace(cl, k, geo, G) for k in ('z', 'zz', 'b', 'x')})
    return _cache[key]


def load(directory):
    """Levels: list of dict(cluster, N, nup, J2, E, kinds, states, files, source ('full'|'sym'), data)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        z = np.load(f)
        meta = json.loads(str(z['meta']))
        recs.append((meta, z, os.path.basename(f)))
    levels = []
    groups = {}
    for meta, z, f in recs:
        src = 'sym' if 'Gb0' in z.files else 'full'
        groups.setdefault((tuple(meta['cluster']), meta['nup'], meta['J2'], src), []).append((meta, z, f))
    for (A, nup, J2, src), lst in sorted(groups.items()):
        lst.sort(key=lambda r: r[0]['E'])
        cur = []
        for r in lst+[None]:
            if r is not None and (not cur or abs(r[0]['E']-cur[0][0]['E']) < 1e-7):
                cur.append(r); continue
            w = [STATES.get(m['sector'], 1) for m, _, _ in cur]
            data = {k: sum(wi*z[k] for wi, (_, z, _) in zip(w, cur))/sum(w) for k in cur[0][1].files if k != 'meta'}
            levels.append(dict(cluster=list(A), N=cur[0][0]['N'], nup=nup, J2=J2, E=cur[0][0]['E'], source=src,
                               kinds=[m['sector'] for m, _, _ in cur], states=sum(w), files=[f for _, _, f in cur],
                               residual=max(m['residual'] for m, _, _ in cur), data=data))
            cur = [r] if r is not None else []
    return levels


def connected(level):
    """{set: connected matrix}; for a file of corr_sym the sets 'b' and 'x' hold the zero-momentum block only."""
    cl, geo, sp = context(level['cluster'])
    d, out = level['data'], {}
    if level['source'] == 'full':
        for kind, Gk, ek in SETS:
            out[kind] = d[Gk]-(np.outer(d[ek], d[ek]) if ek else 0)
    else:
        for kind, Gk, ek in SETS[:2]:
            out[kind] = d[Gk]-np.outer(d[ek], d[ek])
        # zero-momentum blocks; the average over the point group makes them level averages when the lowest state
        # of the sector belongs to a two-dimensional irrep (it changes nothing for a fully symmetric state)
        Nc = cl.Nc
        eb = float(np.mean(d['eb0']))
        out['b'] = sp['b'].average(cs.embed(sp['b'], d['Gb0']-Nc*eb*eb, np.rint(d['bond_type']).astype(int)))
        out['x'] = sp['x'].average(cs.embed(sp['x'], d['Gx0'], np.rint(d['tri_kind']).astype(int)))
    return out


def describe(level, top=4):
    cl, geo, sp = context(level['cluster'])
    C = connected(level)
    d = level['data']
    out = dict(cluster=level['cluster'], N=level['N'], nup=level['nup'], m=(2*level['nup']-level['N'])/level['N'], J2=level['J2'],
               E=level['E'], kinds=level['kinds'], states=level['states'], source=level['source'], residual=level['residual'], sets={})
    for kind in ('z', 'zz', 'b', 'x'):
        sm = ct.summary(sp[kind], C[kind])
        partial = level['source'] == 'sym' and kind in ('b', 'x')
        if partial:                                   # mean eigenvalue from the exact one-operator variances
            if kind == 'b':
                eb = float(np.mean(d['eb0'])); mean = 3/16-eb/2-eb**2
            else:
                eb = float(np.mean(d['eb0'])); mean = 3/32-3*eb/8
        else:
            mean = sm['mean']
        best = {}
        for mo in sm['modes']:
            if mo['star'] not in best:
                best[mo['star']] = dict(irrep=mo['irrep'], value=mo['values'][0], ratio=mo['values'][0]/mean)
        out['sets'][kind] = dict(n=sm['n'], mean=mean, partial=partial, best=best,
                                 top=[dict(star=mo['star'], irrep=mo['irrep'], states=mo['dim'], value=mo['values'][0],
                                           ratio=mo['values'][0]/mean, pattern=mo['pattern']) for mo in sm['modes'][:top]],
                                 all={f"{mo['star']}:{mo['irrep']}": mo['values'] for mo in sm['modes']})
        if not partial and kind != 'x':
            rows = ct.class_table(sp[kind], C[kind])
            for r in rows:
                r['ratio'] = r['value']/mean
            out['sets'][kind]['classes'] = rows
    if level['source'] == 'full':
        out['bond_energy'] = float(np.mean(d['eb']))
    else:
        out['bond_energy'] = float(np.mean(d['eb0']))
    return out


def line(desc, kind, top=3):
    s = desc['sets'][kind]
    t = '; '.join(f"{x['star']} {x['irrep']} {x['value']:.4f} ({x['ratio']:.2f})" for x in s['top'][:top])
    return f"{kind:>2}  mean {s['mean']:.4f} | {t}"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('directory'); ap.add_argument('out'); ap.add_argument('--top', type=int, default=4)
    a = ap.parse_args()
    res = []
    for lev in load(a.directory):
        dsc = describe(lev, a.top)
        res.append(dsc)
        print(f"N = {dsc['N']}, N_up = {dsc['nup']} (m = {dsc['m']:.3f}), J2 = {dsc['J2']}: E = {dsc['E']:.9f}, "
              f"{'+'.join(dsc['kinds'])} ({dsc['states']} states), residual {dsc['residual']:.0e}")
        for kind in ('z', 'zz', 'b', 'x'):
            print('    '+line(dsc, kind)+('   [zero momentum only]' if dsc['sets'][kind]['partial'] else ''))
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out)
