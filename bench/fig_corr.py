#!/usr/bin/env python3
"""Figures of the correlation test (data: bench/corr_full.py, bench/corr_sym.py; analysis: bench/corr_tools.py).

    fig_corr.py modes   <out.png> <directory> [--sets=z,b,x] <spec>=<label> [<spec>=<label> ...]
    fig_corr.py pattern <out.png> <directory> <spec>@<set>[@<kind>[@<symmetry class>]] [...]

<spec> = cluster/N_up/J2/level, e.g. 3,0,0,3/15/0/G:A1 (the level that contains this kind) or .../0 (the lowest level).
modes: one panel per operator set (z: S^z; zz: S^z S^z on bonds; b: S.S on bonds; x: chirality): the largest eigenvalue
  of the connected matrix in every kind of pattern, divided by the mean eigenvalue (1 = operators uncorrelated), one
  marker per state.  For files of corr_sym.py the sets b and x exist at zero momentum only.
pattern: the leading pattern of a set (or of one kind) drawn on the cluster (bonds: colour = sign, width = amplitude;
  sites: circles), as the member of its eigenspace with the given symmetry class (corr_tools.symmetry_classes), by
  default the largest class that contains one."""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corr_tools as ct, corr_post as cp

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                     'legend.fontsize': 7, 'legend.frameon': False, 'savefig.dpi': 300, 'axes.linewidth': 0.6})
COL = ['#111111', '#c0392b', '#2471a3', '#1e8449', '#b9770e', '#7d3c98', '#7f8c8d', '#d35400']
MRK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
KIND_TEX = {'G': r'$\Gamma$', 'K': 'K', 'M': 'M', 'Q': 'Q'}


def pick(levels, spec):
    A, nup, J2, which = spec.split('/')
    A = [int(x) for x in A.split(',')]
    cand = [l for l in levels if l['cluster'] == A and l['nup'] == int(nup) and abs(l['J2']-float(J2)) < 1e-12]
    cand.sort(key=lambda l: l['E'])
    if which.isdigit():
        return cand[int(which)]
    return next(l for l in cand if which in l['kinds'] or which == l['source'])


def tex(kind):
    s, ir = kind.split(':')
    if len(ir) == 2 and ir[1].isdigit():
        ir = f'{ir[0]}$_{ir[1]}$'
    return f'{KIND_TEX[s]} {ir}'


SET_NAME = {'z': 'z', 'zz': 'zz', 'b': 'b', 'x': 'x'}


def fig_modes(out, directory, specs, labels, sets=('z', 'b', 'x')):
    levels = cp.load(directory)
    descs = [cp.describe(pick(levels, s), top=1) for s in specs]
    order = {'G': 0, 'K': 1, 'M': 2, 'Q': 3}
    cols = []
    for kind in sets:
        kinds = []
        for d in descs:
            sd = d['sets'][kind]
            for kd in sd['all']:
                if kd not in kinds and not (sd['partial'] and not kd.startswith('G')):
                    kinds.append(kd)
        kinds.sort(key=lambda kd: (order[kd.split(':')[0]], kd))
        cols.append(kinds)
    fig, axs = plt.subplots(1, len(sets), figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[len(c)+1.5 for c in cols], wspace=0.3))
    axs = np.atleast_1d(axs)
    for ax, kind, kinds, letter in zip(axs, sets, cols, 'abcdef'):
        for di, d in enumerate(descs):
            sd = d['sets'][kind]
            xs = [i+(di-(len(descs)-1)/2)*0.11 for i, kd in enumerate(kinds) if kd in sd['all']]
            ys = [sd['all'][kd][0]/sd['mean'] for kd in kinds if kd in sd['all']]
            ax.plot(xs, ys, MRK[di % len(MRK)], color=COL[di % len(COL)], ms=3.6, mew=0, label=labels[di] if ax is axs[0] else None)
        ax.axhline(1.0, color='#999999', lw=0.6, zorder=0)
        ax.set_xticks(range(len(kinds))); ax.set_xticklabels([tex(k) for k in kinds], rotation=60)
        ax.set_xlim(-0.6, len(kinds)-0.4)
        ax.set_ylim(0, None)
        ax.tick_params(length=2.5, width=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=9, ha='right', va='bottom')
    axs[0].set_ylabel('largest eigenvalue / mean eigenvalue')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.13),
               handletextpad=0.2, columnspacing=1.2)
    fig.savefig(out, bbox_inches='tight')


def canonical(space, C, kind, label=None):
    """The pattern of largest variance inside one kind that is invariant under a subgroup: the symmetry class `label`
    of corr_tools.symmetry_classes, or else the largest one of them that leaves a pattern of that variance."""
    P = ct.kind_projectors(space)[kind]
    Cp = P @ C @ P
    top = np.linalg.eigvalsh(Cp)[-1]
    classes = ct.symmetry_classes(space)
    if label is not None:
        classes = [c for c in classes if c[0] == label]
    for lab, H in sorted(classes, key=lambda c: -len(c[1])):
        PH = sum(space.rep(g) for g in H)/len(H)
        w2, U2 = np.linalg.eigh(PH @ Cp @ PH)
        if w2[-1] > top*(1-1e-8):
            v = U2[:, -1]
            return v*np.sign(v[np.argmax(np.abs(v))]), lab, top
    raise RuntimeError(f'no pattern of kind {kind} with the symmetry {label}')


def draw(ax, cl, geo, kind_set, v, scale=None, radius=2.35):
    """Pattern v with its periodic copies inside a disc around a hexagon centre (the centre of the point group); the
    grey hexagon is the Wigner-Seitz cell of the cluster, one period of the pattern."""
    v = np.asarray(v, float)
    scale = scale or np.max(np.abs(v))
    C = np.array(cl.cart, float)
    A = [C @ cl.M[:, c].astype(float) for c in range(2)]
    shifts = [n1*A[0]+n2*A[1] for n1 in range(-3, 4) for n2 in range(-3, 4)]
    red, blue = '#c0392b', '#2471a3'
    def copies(p):
        return [p+sft for sft in shifts if np.linalg.norm(p.mean(axis=0)+sft) < radius]
    for xy in geo['bond_xy']:
        for q in copies(xy):
            ax.plot(q[:, 0], q[:, 1], '-', color='#dcdcdc', lw=0.5, zorder=1, solid_capstyle='round')
    if kind_set in ('b', 'zz'):
        for xy, c in zip(geo['bond_xy'], v):
            if abs(c) > 0.04*scale:
                for q in copies(xy):
                    ax.plot(q[:, 0], q[:, 1], '-', color=red if c > 0 else blue, lw=0.3+2.6*abs(c)/scale, zorder=2, solid_capstyle='round')
    elif kind_set == 'z':
        done = set()
        for t, P in zip(geo['tris'], geo['tri_xy']):
            for sidx, p in zip(t, P):
                if sidx in done:
                    continue
                done.add(sidx)
                for q in copies(p[None, :]):
                    ax.plot(q[:, 0], q[:, 1], 'o', color=red if v[sidx] > 0 else blue, ms=0.8+4.6*abs(v[sidx])/scale, mew=0, zorder=3)
    else:
        for P, c in zip(geo['tri_xy'], v):
            for q in copies(P.mean(axis=0)[None, :]):
                ax.plot(q[:, 0], q[:, 1], 'o', color=red if c > 0 else blue, ms=0.8+4.6*abs(c)/scale, mew=0, zorder=3)
    # Wigner-Seitz cell of the cluster's superlattice (a hexagon when the two periods have equal length at 60 degrees)
    near = sorted((sft for sft in shifts if 1e-9 < np.linalg.norm(sft) < 1.01*min(np.linalg.norm(A[0]), np.linalg.norm(A[1]))),
                  key=lambda sft: math.atan2(sft[1], sft[0]))
    if len(near) == 6:
        cor = []
        for i in range(6):
            p, q = near[i], near[(i+1) % 6]
            M2 = np.array([p, q]); rhs = 0.5*np.array([p @ p, q @ q])
            cor.append(np.linalg.solve(M2, rhs))
        cor = np.array(cor+[cor[0]])
        ax.plot(cor[:, 0], cor[:, 1], '-', color='#888888', lw=0.6, dashes=(3, 2), zorder=4)
    ax.set_xlim(-radius-0.1, radius+0.1); ax.set_ylim(-radius-0.1, radius+0.1)
    ax.set_aspect('equal'); ax.axis('off')


def fig_pattern(out, directory, items):
    levels = cp.load(directory)
    fig, axs = plt.subplots(1, len(items), figsize=(2.3*len(items), 2.4))
    axs = np.atleast_1d(axs)
    info = []
    for ax, it, letter in zip(axs, items, 'abcdefgh'):
        parts = it.split('@')
        spec, kind_set = parts[0], parts[1]
        lev = pick(levels, spec)
        cl, geo, sp = cp.context(lev['cluster'])
        C = cp.connected(lev)[kind_set]
        sm = ct.summary(sp[kind_set], C)
        kind = parts[2] if len(parts) > 2 and parts[2] else f"{sm['modes'][0]['star']}:{sm['modes'][0]['irrep']}"
        v, order, val = canonical(sp[kind_set], C, kind, parts[3] if len(parts) > 3 else None)
        draw(ax, cl, geo, kind_set, v)
        ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontweight='bold', fontsize=9, ha='left', va='top')
        info.append(dict(item=it, kind=kind, value=float(val), ratio=float(val/sm['mean']), symmetry=order))
    fig.savefig(out, bbox_inches='tight')
    return info


if __name__ == '__main__':
    cmd, out, directory = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd == 'modes':
        specs, labels, sets = [], [], ('z', 'b', 'x')
        for a in sys.argv[4:]:
            if a.startswith('--sets='):
                sets = tuple(a[7:].split(',')); continue
            s, _, lab = a.partition('=')
            specs.append(s); labels.append(lab or s)
        fig_modes(out, directory, specs, labels, sets)
    else:
        for r in fig_pattern(out, directory, sys.argv[4:]):
            print(r)
    print('wrote', out)
