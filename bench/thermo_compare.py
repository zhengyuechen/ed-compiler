#!/usr/bin/env python3
"""Kagome 36 thermodynamics from thermo_run.py against curves read from the published figures
(refs/published-curves: Schnack, Schulenburg, Richter, PRB 98, 094423 (2018); Morita, PRB 108, 184405 (2023)).
For every published curve our values are computed at the published abscissae and the differences are reported per
temperature (or field) range; the half-split of the random vectors gives our statistical error at the same points.

Usage: thermo_compare.py <results dir with k36_<tag>_p*.jsonl> <refs/published-curves> <out prefix> tag=J2 [tag=J2 ...] [--fix]
(--fix: exact weights for the converged levels, see thermo_post.py)
Writes <out prefix>.json (differences), <out prefix>_curves.json (our curves on a grid) and <out prefix>.png."""
import sys, os, glob, json, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thermo_post

FIX = '--fix' in sys.argv
argv = [x for x in sys.argv if x != '--fix']
res_dir, ref_dir, prefix = argv[1:4]
tags = {t.split('=')[0]: float(t.split('=')[1]) for t in argv[4:]}
data = {}
for tag, J2 in tags.items():
    files = sorted(glob.glob(os.path.join(res_dir, f'k36_{tag}_p*.jsonl')))
    if files:
        try:
            data[J2] = thermo_post.load(files, fix=FIX)
        except AssertionError as e:
            print(f'{tag}: {e}')


def read(name, **sel):
    rows = []
    for r in csv.DictReader(open(os.path.join(ref_dir, name))):
        if all(abs(float(r[k])-v) < 1e-9 if isinstance(v, float) else r[k] == v for k, v in sel.items()):
            rows.append(r)
    return rows


def xy(rows, edge='on_frame_edge', inframe=None):
    keep = [r for r in rows if (edge not in r or r[edge] == '0') and (inframe is None or r[inframe] == '1')]
    x = np.array([float(r['x']) for r in keep]); y = np.array([float(r['y']) for r in keep])
    return x, y


def ours(J2, q, T, h=0.0):
    d = data[J2]
    full, _ = d.thermo(T, h); a, _ = d.thermo(T, h, 0); b, _ = d.thermo(T, h, 1)
    return full[q], 0.5*np.abs(a[q]-b[q])


RANGES = [(0.02, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 10.0)]
report = []


def compare_T(label, J2, q, x, y, scale=1.0):
    """Published (x = T, y) against our quantity q*scale at the same temperatures."""
    if J2 not in data:
        return
    sel = (x >= 0.02) & (x <= 10)
    x, y = x[sel], y[sel]
    o, err = ours(J2, q, x); o, err = o*scale, err*scale
    row = dict(curve=label, J2=J2, quantity=q, points=int(len(x)), ranges={})
    for lo, hi in RANGES:
        m = (x >= lo) & (x < hi)
        if m.any():
            row['ranges'][f'{lo}-{hi}'] = dict(max_abs_diff=float(np.abs(o[m]-y[m]).max()), max_half_diff=float(err[m].max()),
                                             max_value=float(np.abs(y[m]).max()))
    report.append(row)
    print(f'{label:46s}', ' '.join(f"[{k}] {v['max_abs_diff']:.1e} ({v['max_half_diff']:.0e})" for k, v in row['ranges'].items()))


print('largest |ours - published| per temperature range (in brackets: our half-split error)')
ln2 = np.log(2)
# Schnack 2018, N = 36, J2 = 0: C, S, chi
if 0.0 in data:
    for q, name, sc in (('C', 'schnack2018_fig1_C_vs_T.csv', 1.0), ('S', 'schnack2018_fig3_S_vs_T.csv', 1/ln2),
                        ('chi', 'schnack2018_fig4_chi_vs_T.csv', 1.0)):
        x, y = xy(read(name, panel='a', lattice='kagome', N='36'))
        compare_T(f'Schnack 2018 N=36 {q}', 0.0, q, x, y, sc)
    x, y = xy(read('schnack2018_fig1_C_vs_T.csv', panel='a', lattice='kagome', N='42'))
    compare_T('Schnack 2018 N=42 C (other size)', 0.0, 'C', x, y)
# Morita 2023, N = 36: C(T) for the J2 values of Fig. 4 that were run
for J2 in sorted(data):
    rows = read('morita2023_fig4_C_vs_T_N36.csv', J2=float(J2))
    if rows:
        panel = rows[0]['panel']
        x, y = xy([r for r in rows if r['panel'] == panel], inframe='in_frame')
        compare_T(f'Morita 2023 N=36 C, J2={J2:g}', J2, 'C', x, y)
    rows = [r for r in csv.DictReader(open(os.path.join(ref_dir, 'morita2023_fig5_entropy_vs_T_slices_N36.csv')))
            if abs(float(r['J2'])-J2) < 1e-9 and r['n_crossings_at_this_level'] == '1']
    if rows:
        x = np.array([float(r['T']) for r in rows]); y = np.array([float(r['S_over_N']) for r in rows])
        compare_T(f'Morita 2023 N=36 S (iso-lines), J2={J2:g}', J2, 'S', x, y)
# Morita 2023, magnetisation curves at fixed temperature (Fig. 8)
mag = []
for J2 in sorted(data):
    for r0 in ('0.05', '0.1', '0.2'):
        rows = read('morita2023_fig8_M_vs_h_N36.csv', J2=float(J2), temperature=float(r0))
        if not rows:
            continue
        h, y = xy(rows, inframe='in_frame')
        h, y = h[::4], y[::4]
        T = float(r0)
        o = np.array([data[J2].thermo([T], hh)[0]['m'][0] for hh in h])
        a = np.array([data[J2].thermo([T], hh, 0)[0]['m'][0] for hh in h]); b = np.array([data[J2].thermo([T], hh, 1)[0]['m'][0] for hh in h])
        d = float(np.abs(o-y).max()); e = float((0.5*np.abs(a-b)).max())
        mag.append(dict(J2=J2, T=T, points=int(len(h)), max_abs_diff=d, max_half_diff=e, h=h.tolist(), ours=o.tolist(), published=y.tolist()))
        print(f'Morita 2023 N=36 M/Msat vs h, J2={J2:g}, T={T:g}: largest difference {d:.1e} ({e:.0e}) over {len(h)} fields')
json.dump(dict(curves=report, magnetisation=[{k: v for k, v in m.items() if k not in ('h', 'ours', 'published')} for m in mag]),
          open(prefix+'.json', 'w'), indent=1)

# our curves on a grid, for the record and the figure
Tg = np.geomspace(0.01, 10, 301)
curves = {}
for J2, d in data.items():
    full, E0 = d.thermo(Tg); a, _ = d.thermo(Tg, 0.0, 0); b, _ = d.thermo(Tg, 0.0, 1)
    curves[str(J2)] = dict(summary=d.summary(), E0=E0, T=Tg.tolist(), **{k: v.tolist() for k, v in full.items()},
                           **{k+'_half_diff': (0.5*np.abs(a[k]-b[k])).tolist() for k in full})
json.dump(curves, open(prefix+'_curves.json', 'w'))

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                     'legend.fontsize': 6.5, 'legend.frameon': False, 'axes.linewidth': 0.6, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
GREY, BLUE = '#8a8a8a', '#1f4e9c'
COL = {-0.3: '#b5482a', 0.0: BLUE, 0.12: '#2a8a5a', 0.3: '#7a3fa0'}
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.3))
pub = dict(color=GREY, lw=2.6, alpha=0.55, solid_capstyle='round')
ax = axs[0, 0]                                                           # C(T), J2 = 0
if 0.0 in data:
    x, y = xy(read('schnack2018_fig1_C_vs_T.csv', panel='a', lattice='kagome', N='36')); ax.plot(x, y, **pub, label='Schnack et al. 2018')
    rows = read('morita2023_fig4_C_vs_T_N36.csv', J2=0.0); x, y = xy([r for r in rows if r['panel'] == rows[0]['panel']], inframe='in_frame')
    ax.plot(x, y, color='#c9a227', lw=1.6, alpha=0.8, label='Morita 2023')
    c = curves['0.0']; ax.plot(c['T'], c['C'], color=BLUE, lw=0.8, label='this work')
ax.set_xscale('log'); ax.set_xlim(0.01, 10); ax.set_ylim(0, 0.27); ax.set_xlabel('$T/J_1$'); ax.set_ylabel('$C/N$'); ax.legend(loc='upper left')
ax = axs[0, 1]                                                           # S(T), J2 = 0
if 0.0 in data:
    x, y = xy(read('schnack2018_fig3_S_vs_T.csv', panel='a', lattice='kagome', N='36')); ax.plot(x, y, **pub)
    c = curves['0.0']; ax.plot(c['T'], np.array(c['S'])/ln2, color=BLUE, lw=0.8)
ax.set_xscale('log'); ax.set_xlim(0.01, 10); ax.set_ylim(0, 1.02); ax.set_xlabel('$T/J_1$'); ax.set_ylabel('$S/(N\\ln 2)$')
ax = axs[0, 2]                                                           # chi(T), J2 = 0
if 0.0 in data:
    x, y = xy(read('schnack2018_fig4_chi_vs_T.csv', panel='a', lattice='kagome', N='36')); ax.plot(x, y, **pub)
    c = curves['0.0']; ax.plot(c['T'], c['chi'], color=BLUE, lw=0.8)
ax.set_xscale('log'); ax.set_xlim(0.01, 10); ax.set_ylim(0, 0.16); ax.set_xlabel('$T/J_1$'); ax.set_ylabel('$\\chi/N$')
ax = axs[1, 0]                                                           # C(T), J2 != 0
for J2 in sorted(data):
    if J2 == 0.0:
        continue
    rows = read('morita2023_fig4_C_vs_T_N36.csv', J2=float(J2))
    if rows:
        x, y = xy(rows, inframe='in_frame'); ax.plot(x, y, **pub)
    c = curves[str(J2)]; ax.plot(c['T'], c['C'], color=COL.get(J2, 'k'), lw=0.8, label=f'$J_2={J2:g}$')
ax.set_xscale('log'); ax.set_xlim(0.01, 10); ax.set_ylim(0, 0.34); ax.set_xlabel('$T/J_1$'); ax.set_ylabel('$C/N$')
if ax.get_legend_handles_labels()[0]:
    ax.legend(loc='upper right')
ax = axs[1, 1]                                                           # S(T), all J2, with Morita's iso-lines
for J2 in sorted(data):
    rows = [r for r in csv.DictReader(open(os.path.join(ref_dir, 'morita2023_fig5_entropy_vs_T_slices_N36.csv')))
            if abs(float(r['J2'])-J2) < 1e-9 and r['n_crossings_at_this_level'] == '1']
    if rows:
        ax.plot([float(r['T']) for r in rows], [float(r['S_over_N'])/ln2 for r in rows], 'o', ms=1.8, mfc='none', mew=0.4, color=GREY)
    c = curves[str(J2)]; ax.plot(c['T'], np.array(c['S'])/ln2, color=COL.get(J2, 'k'), lw=0.8, label=f'$J_2={J2:g}$')
ax.set_xscale('log'); ax.set_xlim(0.01, 10); ax.set_ylim(0, 1.02); ax.set_xlabel('$T/J_1$'); ax.set_ylabel('$S/(N\\ln 2)$'); ax.legend(loc='upper left')
ax = axs[1, 2]                                                           # M(h) at T = 0.1
for m in mag:
    if abs(m['T']-0.1) < 1e-9:
        ax.plot(m['h'], m['published'], **pub)
        ax.plot(m['h'], m['ours'], color=COL.get(m['J2'], 'k'), lw=0.8, label=f"$J_2={m['J2']:g}$")
ax.set_xlabel('$h/J_1$'); ax.set_ylabel('$M/M_{\\rm sat}$'); ax.set_xlim(0, 3.3); ax.set_ylim(0, 1.02); ax.legend(loc='upper left')
for a_, l in zip(axs.ravel(), 'abcdef'):
    a_.text(-0.2, 1.03, l, transform=a_.transAxes, fontweight='bold', fontsize=9)
fig.tight_layout()
fig.savefig(prefix+'.png')
print('written', prefix+'.json', prefix+'_curves.json', prefix+'.png')
