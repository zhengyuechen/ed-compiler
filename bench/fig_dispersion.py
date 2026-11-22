#!/usr/bin/env python3
"""Lowest S=1 excitation of the Heisenberg ring vs momentum (GPU, any-momentum engine) against (pi/2)|sin k|."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                     'legend.fontsize': 7, 'legend.frameon': False, 'axes.linewidth': 0.6, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
fig, ax = plt.subplots(figsize=(3.4, 2.4))
kk = np.linspace(0, np.pi, 200)
ax.plot(kk, np.pi/2*np.abs(np.sin(kk)), color='#6e6e6e', lw=1.0, label=r'$(\pi/2)|\sin k|$, $L\to\infty$')
summary = []
for L, col, mk in ((24, '#9ecae1', 's'), (32, '#1f4e9c', 'o')):
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results', '2026-11-21_000000_first-night', 'ris', f'ris_dispersion{L}.jsonl')) if l.strip()]
    e0 = min(r['E'] for r in rows if r['Sz'] == 0 and r['E'] is not None)
    s1 = sorted((r['k'], r['E']-e0) for r in rows if r['Sz'] == 1 and r['E'] is not None)
    ax.plot(*zip(*s1), mk, color=col, ms=3.2, label=f'L = {L}')
    summary.append((L, e0, s1, sum(r['total_seconds'] for r in rows), len(rows)))
ax.set_xlim(0, np.pi); ax.set_xticks([0, np.pi/2, np.pi]); ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.set_xlabel('momentum $k$ (relative to the ground state)'); ax.set_ylabel(r'$E_{S_z=1}(k)-E_0$  (J)')
ax.legend(loc='lower center', handletextpad=0.4)
fig.tight_layout(); out = os.path.join(HERE, 'summary', 'dispersion_2026-11-21.png'); fig.savefig(out)
for L, e0, s1, tt, n in summary:
    gap = s1[-1]
    print(f'L={L}: E0 = {e0:.10f} ({e0/L:.8f} per site); {n} sectors in {tt:.1f} s total; '
          f'gap at k=pi: {gap[1]:.5f}; at k=pi/2: ' + ', '.join(f'{d:.4f}' for k, d in s1 if abs(k-np.pi/2) < 1e-9))
print(out)
