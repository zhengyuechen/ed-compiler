#!/usr/bin/env python3
"""Summary figure for the 2026-11-21 report: (a) thread scaling on one RIS node, ring 32 k=0;
(b) time to the ground-state energy at 36 sites; (c) Hubbard half filling, time per Lanczos iteration."""
import json, os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, 'results', '2026-11-21_000000_first-night', 'ris')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7, 'legend.frameon': False, 'lines.linewidth': 1.1, 'lines.markersize': 3.6,
                     'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})
RED, BLUE, GREY, GREEN = '#c0392b', '#1f4e9c', '#6e6e6e', '#2e7d32'


def load(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


edc = load('ris_e2e_edc.jsonl'); xd = load('ris_e2e_xdiag.jsonl')
gpu_full = load('ris_e2e_gpu_full.jsonl'); edc36 = load('ris_e2e_edc36.jsonl'); xd36 = load('ris_e2e_xdiag36.jsonl')
hub_cpu = load('ris_hub_cpu.jsonl'); hub_gpu = load('ris_hub_gpu.jsonl')

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
# (a) scaling
ax = axs[0]
for rows, col, lab in ((edc, BLUE, 'edc, CPU'), (xd, RED, 'XDiag, CPU')):
    pts = sorted((r['threads'], r['seconds_per_iteration']) for r in rows if r['model'] == 'ring:32:sym')
    if pts:
        ax.plot(*zip(*pts), 'o-', color=col, label=lab)
g = [r for r in gpu_full if r['model'] == 'ring:32:sym']
if g:
    ax.axhline(g[0]['seconds_per_iteration'], color=GREEN, ls='--', label='edc, one H100')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
ax.set_xlabel('CPU threads'); ax.set_ylabel('time per Lanczos step (s)')
ax.legend(loc='lower left')
ax.text(-0.3, 1.04, 'a', transform=ax.transAxes, fontweight='bold', fontsize=9)
# (b) 36 sites, total time
ax = axs[1]
models = [('ring:36:sym', 'ring 36'), ('square:6x6:sym', 'square 6×6')]
codes = [(xd36, 'XDiag 64 cores', RED), (edc36, 'edc 64 cores', BLUE), (gpu_full, 'edc H100', GREEN)]
w = 0.26
for k, (rows, lab, col) in enumerate(codes):
    vals = []
    for m, _ in models:
        r = [x for x in rows if x['model'] == m]
        vals.append(r[0]['setup_seconds']+r[0]['lanczos_seconds'] if r else float('nan'))
    ax.bar([i+(k-1)*w for i in range(len(models))], vals, w, color=col, label=lab)
ax.set_yscale('log'); ax.set_xticks(range(len(models))); ax.set_xticklabels([n for _, n in models])
ax.set_ylabel('time to ground-state energy (s)')
ax.text(-0.3, 1.04, 'b', transform=ax.transAxes, fontweight='bold', fontsize=9)
# (c) Hubbard
ax = axs[2]
hm = [('chain:16:8.0:8:8', 'chain 16'), ('square:4x4:8.0:8:8', 'square 4×4')]
hc = [([r for r in hub_cpu if r['code'] == 'xdiag'], 'XDiag 64 cores', RED),
      ([r for r in hub_cpu if r['code'] == 'edc-kron'], 'edc 64 cores', BLUE), (hub_gpu, 'edc H100', GREEN)]
for k, (rows, lab, col) in enumerate(hc):
    vals = []
    for m, _ in hm:
        r = [x for x in rows if x['model'] == m]
        vals.append(r[0]['seconds_per_iteration'] if r else float('nan'))
    ax.bar([i+(k-1)*w for i in range(len(hm))], vals, w, color=col, label=lab)
ax.set_yscale('log'); ax.set_xticks(range(len(hm))); ax.set_xticklabels([n for _, n in hm])
ax.set_ylabel('time per Lanczos step (s)')
ax.text(-0.3, 1.04, 'c', transform=ax.transAxes, fontweight='bold', fontsize=9)
fig.tight_layout(w_pad=1.2)
out = os.path.join(HERE, 'summary', 'summary_2026-11-21.png')
fig.savefig(out); fig.savefig(out.replace('.png', '.pdf'))
print(out)
