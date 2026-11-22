#!/usr/bin/env python3
"""N = 40 J1-J2 (tilted (6,2), (-2,6)): our ground-state and lowest S_z = 1 energies against Richter & Schulenburg 2010.
Usage: fig_n40.py <results dir with *.json> <output png>"""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RS = {0.0: (-27.09485025, -26.83322962), 0.1: (-25.46460260, -25.19683764), 0.2: (-23.90046918, -23.62413648),
      0.3: (-22.42728643, -22.13740165), 0.4: (-21.08836670, -20.77332213), 0.5: (-19.96304839, -19.59762345),
      0.55: (-19.51791526, -19.11727950), 0.6: (-19.18368038, -18.90110215), 0.65: (-20.04603255, -19.75459787),
      0.7: (-21.05530239, -20.75925837), 0.8: (-23.34020427, -23.01691919), 0.9: (-25.83691287, -25.46196339),
      1.0: (-28.43880892, -28.00735838)}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                     'legend.fontsize': 7, 'legend.frameon': False, 'axes.linewidth': 0.6, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                     'lines.markersize': 3.6})
rows = []
for f in glob.glob(os.path.join(sys.argv[1], '*.json')):
    for l in open(f):
        if l.strip():
            rows.append(json.loads(l))
gs = {round(r['J2'], 2): r['E'] for r in rows if r['nup'] == 20}
s1 = {}
for r in rows:
    if r['nup'] == 21 and r['E'] is not None:
        J = round(r['J2'], 2)
        if J not in s1 or r['E'] < s1[J][0]:
            s1[J] = (r['E'], tuple(r['k_over_pi']), r['irrep'])
J = sorted(set(gs) & set(RS))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax = axs[0]
ax.plot(sorted(RS), [RS[j][0]/40 for j in sorted(RS)], 'o', mfc='none', color='#6e6e6e', label='Richter & Schulenburg 2010')
ax.plot(J, [gs[j]/40 for j in J], 'x', color='#1f4e9c', label='this work, one H100')
ax.set_xlabel('$J_2/J_1$'); ax.set_ylabel('$E_0/N$'); ax.legend(loc='lower left')
ax.text(-0.22, 1.04, 'a', transform=ax.transAxes, fontweight='bold', fontsize=9)
ax = axs[1]
Jt = sorted(set(s1) & set(gs) & set(RS))
ax.plot(sorted(RS), [RS[j][1]-RS[j][0] for j in sorted(RS)], 'o', mfc='none', color='#6e6e6e')
ax.plot(Jt, [s1[j][0]-gs[j] for j in Jt], 'x', color='#1f4e9c')
ax.set_xlabel('$J_2/J_1$'); ax.set_ylabel(r'$E_0(S=1)-E_0$')
ax.text(-0.22, 1.04, 'b', transform=ax.transAxes, fontweight='bold', fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[2])
print('J2   E0(ours)        E0(R&S)         diff      E(S=1)(ours)    sector           E(S=1)(R&S)     diff')
for j in sorted(set(gs) | set(s1)):
    a = gs.get(j); b = s1.get(j); r = RS.get(j, (None, None))
    print(f'{j:<4} ' + (f'{a:.8f}  {r[0]:.8f}  {a-r[0]:+.1e}' if a is not None and r[0] else ' '*40) + '  ' +
          (f'{b[0]:.8f}  k/pi={b[1]} {b[2]:3s}  {r[1]:.8f}  {b[0]-r[1]:+.1e}' if b is not None and r[1] else ''))
