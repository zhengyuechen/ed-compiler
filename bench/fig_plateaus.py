#!/usr/bin/env python3
"""Figures for the zero-temperature magnetisation process of the kagome clusters (levels_post.py output).

  fig_plateaus.py steps <out.png> <dir>
      a: magnetisation staircase at J2 = 0 for 27 and 36 sites; b: width of the steps at m = 1/9, 1/3, 5/9, 7/9 against J2
      (27 sites: lines; 36 sites: symbols), and for 1/9 the mean width of the two neighbouring steps; c: the same against the
      XXZ ratio on 27 sites (if the scan exists).
  fig_plateaus.py spectrum <out.png> <spectrum.json> [<spectrum.json> ...]
      lowest levels of one S_z sector, E - E0 against momentum, one panel per file; symbol = point-group label.
  fig_plateaus.py flow <out.png> <dir> [ymax=0.08] [jmax=0.1]
      the m = 1/9 sector against J2: a, b: the lowest level of each kind (Gamma A, Gamma E2, K A, K E, any other
      momentum) above the ground level of the sector, 27 sites (N_up = 15) and 36 sites (N_up = 20); c: the number of
      states within cut (default 0.035) of the ground level."""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                     'legend.fontsize': 6.5, 'legend.frameon': False, 'axes.linewidth': 0.6, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
COL = {'1/9': '#b5482a', '1/3': '#1f4e9c', '5/9': '#2a8a5a', '7/9': '#7a3fa0'}


def staircase(ax, d, p, **kw):
    N = d['N']; r = d['ranges'][p]
    Ms = sorted(float(k) for k in r)
    h, m = [0.0], [2*Ms[0]/N if r[str(Ms[0])][0] > 0 else 2*Ms[0]/N]
    for M in Ms:
        lo, hi = r[str(M)]
        h += [lo, min(hi, 3.6)]; m += [2*M/N, 2*M/N]
    ax.plot(h[1:], m[1:], **kw)


def neighbours(d, p, M):
    """Mean width of the stable steps just below and just above M at scan point p (None if M is not stable)."""
    r = d['ranges'][p]
    Ms = sorted(float(k) for k in r)
    if M not in Ms:
        return None
    i = Ms.index(M)
    # a neighbour counts only if the data hold the stable sectors on both of its sides (or it is the lowest sector)
    ok = [j for j in (i-1, i+1) if 0 <= j < len(Ms) and j+1 < len(Ms) and (j >= 1 or Ms[j] in (0.0, 0.5)) and np.isfinite(r[str(Ms[j])][1])]
    w = [r[str(Ms[j])][1]-r[str(Ms[j])][0] for j in ok]
    return float(np.mean(w)) if w else None


cmd, out = sys.argv[1], sys.argv[2]
if cmd == 'steps':
    D = sys.argv[3]
    sJ = json.load(open(os.path.join(D, 'n27_scan_J2_plateaus.json')))
    sd = json.load(open(os.path.join(D, 'n27_scan_delta_plateaus.json'))) if os.path.exists(os.path.join(D, 'n27_scan_delta_plateaus.json')) else None
    k36 = sorted((json.load(open(f)) for f in glob.glob(os.path.join(D, 'k36_*_plateaus.json'))), key=lambda d: (d['J']+[0.0])[1])
    fig, axs = plt.subplots(1, 3 if sd else 2, figsize=(7.4 if sd else 5.2, 2.5))
    ax = axs[0]
    i0 = int(np.argmin(np.abs(np.array(sJ['values']))))
    staircase(ax, sJ, i0, color='#8a8a8a', lw=0.9, label='27 sites')
    for d in k36:
        if len(d['J']) == 1 or d['J'][1] == 0:
            staircase(ax, d, 0, color='#1f4e9c', lw=0.9, label='36 sites')
    for f in (1/9, 1/3, 5/9, 7/9):
        ax.axhline(f, color='k', lw=0.3, ls=':')
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 1.02); ax.set_xlabel('$h/J_1$'); ax.set_ylabel('$M/M_{\\rm sat}$'); ax.legend(loc='upper left')
    ax = axs[1]
    v = np.array(sJ['values'])
    for name, c in COL.items():
        ax.plot(v, [p['width'] for p in sJ['plateaus'][name]], color=c, lw=0.9, label=f'$m={name}$')
        xs = [(d['J']+[0.0])[1] for d in k36 if name in d['plateaus']]; ys = [d['plateaus'][name][0]['width'] for d in k36 if name in d['plateaus']]
        ax.plot(xs, ys, 'o', color=c, ms=3.2, mfc='white', mew=0.8)
    nb = [neighbours(sJ, p, sJ['plateaus']['1/9'][p]['M']) for p in range(len(v))]
    ax.plot(v, [np.nan if x is None else x for x in nb], color=COL['1/9'], lw=0.7, ls='--')
    xs = [(d['J']+[0.0])[1] for d in k36]; ys = [neighbours(d, 0, d['plateaus']['1/9'][0]['M']) for d in k36]
    ax.plot(xs, [np.nan if y is None else y for y in ys], 's', color=COL['1/9'], ms=2.6, mfc='white', mew=0.6)
    ax.set_xlabel('$J_2/J_1$'); ax.set_ylabel('step width $/J_1$'); ax.set_ylim(0, 1.15); ax.legend(loc='upper center', ncol=2, columnspacing=1.0, handlelength=1.2)
    if sd:
        ax = axs[2]; v = np.array(sd['values']); hs = 1+2*v                 # saturation field in units of Jxy
        for name, c in COL.items():
            ax.plot(v, np.array([p['width'] for p in sd['plateaus'][name]])/hs, color=c, lw=0.9)
        nb = [neighbours(sd, p, sd['plateaus']['1/9'][p]['M']) for p in range(len(v))]
        ax.plot(v, np.array([np.nan if x is None else x for x in nb])/hs, color=COL['1/9'], lw=0.7, ls='--')
        ax.set_xlabel('$J_z/J_{xy}$'); ax.set_ylabel('step width $/h_{\\rm sat}$'); ax.set_ylim(0, None)
    for a_, l in zip(np.atleast_1d(axs), 'abc'):
        a_.text(-0.22, 1.03, l, transform=a_.transAxes, fontweight='bold', fontsize=9)
    fig.tight_layout(); fig.savefig(out); print('written', out)
elif cmd == 'flow':
    D = sys.argv[3]
    YMAX = next((float(f[5:]) for f in sys.argv[4:] if f.startswith('ymax=')), 0.08)
    JMAX = next((float(f[5:]) for f in sys.argv[4:] if f.startswith('jmax=')), 0.1)
    CUT = next((float(f[4:]) for f in sys.argv[4:] if f.startswith('cut=')), 0.035)
    KIND = {('G', 'A'): ('o', '#1f4e9c', True, '$\\Gamma$ A'), ('G', 'E2'): ('v', '#1f4e9c', False, '$\\Gamma$ E$_2$'),
            ('K', 'A'): ('o', '#b5482a', True, 'K A'), ('K', 'E'): ('D', '#b5482a', False, 'K E'),
            ('x', ''): ('x', '#8a8a8a', True, 'other momenta')}

    def kind(l):
        k, ir = l['k'], l['irrep']
        if k not in ('G', 'K'):
            return ('x', '')
        if ir[:1] in 'ABE':
            return (k, ir[0] if ir[0] in 'AB' else ir.split()[0])
        n, q = (int(x) for x in ir.split()[0][1:].split(':'))
        return (k, {0: 'A', n//2 if n % 2 == 0 else -1: 'B'}.get(q, ('E1' if q in (1, 5) else 'E2') if n == 6 else 'E'))

    def points(pat):
        pts = {}
        for f in glob.glob(os.path.join(D, pat)):
            d = json.load(open(f)); J2 = round((d['J']+[0.0])[1], 6)
            if abs(J2) > JMAX+1e-9 or (J2 in pts and pts[J2].get('complete_below', 9) >= d.get('complete_below', 9)):
                continue
            pts[J2] = d
        return sorted(pts.items())

    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    sets = [points('n27_*_nup15_spectrum.json'), points('k36_*_nup20*_spectrum.json')]
    for ax, pts in zip(axs, sets):
        for kd, (mk, c, filled, lab) in KIND.items():
            xs, ys = [], []
            for J2, d in pts:
                e = next((l['dE'] for l in d['levels'] if kind(l) == kd), None)
                if e is not None and e <= d.get('complete_below', 9):
                    xs.append(J2); ys.append(e)
            ax.plot(xs, ys, '-', color=c, lw=0.6, alpha=0.7, zorder=1)
            ax.plot(xs, ys, mk, color=c, ms=3.0, mfc=c if filled else 'white', mew=0.7, zorder=3, label=lab)
        ax.set_xlabel('$J_2/J_1$'); ax.set_xlim(-JMAX-0.01, JMAX+0.01); ax.set_ylim(-0.003, YMAX); ax.set_ylabel('$E-E_0$')
    ax = axs[2]
    for pts, c, lab in zip(sets, ('#8a8a8a', '#1f4e9c'), ('27 sites', '36 sites')):
        xs = [J2 for J2, d in pts if d.get('complete_below', 9) >= CUT]
        ys = [sum(l['states'] for l in d['levels'] if l['dE'] < CUT) for J2, d in pts if d.get('complete_below', 9) >= CUT]
        ax.plot(xs, ys, 'o-', color=c, ms=3.0, lw=0.7, mfc='white', mew=0.8, label=lab)
    ax.set_yscale('log'); ax.set_xlabel('$J_2/J_1$'); ax.set_ylabel(f'states with $E-E_0<{CUT:g}$'); ax.set_xlim(-JMAX-0.01, JMAX+0.01)
    ax.set_ylim(0.8, 150); ax.legend(loc='upper right')
    for a_, lab in zip(axs, 'abc'):
        a_.text(-0.24, 1.03, lab, transform=a_.transAxes, fontweight='bold', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.37, 1.0), ncol=5, columnspacing=1.2, handletextpad=0.2)
    fig.savefig(out); print('written', out)
else:
    files = [f for f in sys.argv[3:] if not f.startswith('ymax=')]
    YMAX = next((float(f[5:]) for f in sys.argv[3:] if f.startswith('ymax=')), 0.1)
    fig, axs = plt.subplots(1, len(files), figsize=(3.4*len(files), 2.9), squeeze=False)
    MK = {'A1': ('o', '#1f4e9c'), 'A2': ('o', '#b5482a'), 'B1': ('s', '#1f4e9c'), 'B2': ('s', '#b5482a'), 'E1': ('^', '#2a8a5a'),
          'E2': ('v', '#7a3fa0'), 'E': ('D', '#7a3fa0'), 'A': ('o', '#555555')}
    for ax, f in zip(axs[0], files):
        d = json.load(open(f))
        order = []
        for l in d['levels']:
            if l['k'] not in order:
                order.append(l['k'])
        order.sort(key=lambda k: (k != 'G', k != 'K', k != 'M', k))
        seen = set()
        for l in d['levels']:
            ir = l['irrep']
            if ir[:1] in 'ABE':
                lab = ir.split()[0]
            elif ir.startswith(('C3:0', 'C6:0')):
                lab = 'A'
            elif ir.startswith('C3:'):
                lab = 'E'
            elif 's+' in ir:
                lab = 'even'
            elif 's-' in ir:
                lab = 'odd'
            else:
                lab = ir
            MK.update(even=('x', '#333333'), odd=('+', '#333333'))
            mk, c = MK.get(lab, ('x', '#333333'))
            if l['dE'] > YMAX:
                continue
            ax.plot(order.index(l['k']), l['dE'], mk, color=c, ms=3.8, mfc='none' if lab.endswith('2') and lab[0] in 'AB' else c, mew=0.8,
                    label=None if lab in seen else {'even': 'mirror even', 'odd': 'mirror odd'}.get(lab, lab))
            seen.add(lab)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(['$\\Gamma$' if k == 'G' else ('$Q$' if k.startswith('Q') else k) for k in order])
        ax.set_xlim(-0.5, len(order)+0.9); ax.set_ylim(-0.004, YMAX); ax.set_ylabel('$E-E_0$'); ax.legend(loc='center right', handletextpad=0.2)
    for a_, l in zip(axs[0], 'abcd'):
        a_.text(-0.2, 1.03, l, transform=a_.transAxes, fontweight='bold', fontsize=9)
    fig.tight_layout(); fig.savefig(out); print('written', out)
