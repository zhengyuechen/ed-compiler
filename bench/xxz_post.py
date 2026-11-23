#!/usr/bin/env python3
"""Specific heat of the XXZ kagome model against the anisotropy: features of C(T) and a figure.

    xxz_post.py <out.json> [--fig out.png] <N>:<alpha>:<file or glob>[,<file or glob>...] ...

Model: H = sum_bonds [S^z S^z + alpha (S^x S^x + S^y S^y)], temperatures in units of the S^z S^z coupling (the
convention of Popp, Ramirez and Syzranov 2025).  Files are outputs of thermo_run.py (levels with exact weights where
every chain of a block has converged, thermo_post.py --fix).  For every curve:
  main maximum (T_h, C_h);
  the maxima of C below it: T* is the one at the highest temperature, T_L the next;
  the minimum between T* and the main maximum (T_d, C_d), the entropy there S_d (per site), and the contrast C* - C_d;
  the difference of the two halves of the random vectors at these points (statistical error).
A maximum whose contrast is smaller than the half difference is not a resolved peak ('resolved': false)."""
import sys, os, json, glob, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import thermo_post as tp

S_ISING = 0.50183        # ground-state entropy per site of the kagome Ising antiferromagnet (Kano and Naya 1953)


def curve(files, T):
    d = tp.load(files, fix=True)
    full, E0 = d.thermo(T, 0.0)
    a, _ = d.thermo(T, 0.0, 0); b, _ = d.thermo(T, 0.0, 1)
    return dict(C=full['C'], S=full['S'], dC=0.5*np.abs(a['C']-b['C']), dS=0.5*np.abs(a['S']-b['S']), E0=E0,
                ln_weight=d.summary()['ln_total_weight_per_site'], blocks=len(d.recs))


def features(T, c):
    C, S, dC = c['C'], c['S'], c['dC']
    mx = [i for i in range(1, len(T)-1) if C[i] > C[i-1] and C[i] >= C[i+1]]
    mn = [i for i in range(1, len(T)-1) if C[i] < C[i-1] and C[i] <= C[i+1]]
    ih = max(mx, key=lambda i: C[i] if T[i] > 0.15 else -1)
    low = [i for i in mx if i < ih]
    out = dict(T_h=float(T[ih]), C_h=float(C[ih]))
    if not low:
        return dict(out, T_star=None)
    i_s = low[-1]
    dips = [i for i in mn if i_s < i < ih]
    i_d = min(dips, key=lambda i: C[i]) if dips else None
    out.update(T_star=float(T[i_s]), C_star=float(C[i_s]), dC_star=float(dC[i_s]))
    if i_d is not None:
        out.update(T_d=float(T[i_d]), C_d=float(C[i_d]), S_d=float(S[i_d]), dS_d=float(c['dS'][i_d]),
                   contrast=float(C[i_s]-C[i_d]), resolved=bool(C[i_s]-C[i_d] > max(dC[i_s], dC[i_d])))
    if len(low) > 1:
        out.update(T_L=float(T[low[-2]]), C_L=float(C[low[-2]]))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('items', nargs='+'); ap.add_argument('--fig', default='')
    ap.add_argument('--T', default='0.004:3:700')
    a = ap.parse_args()
    lo, hi, n = a.T.split(':'); T = np.geomspace(float(lo), float(hi), int(n))
    res = []
    for it in a.items:
        N, alpha, pats = it.split(':', 2)
        files = sorted(f for p in pats.split(',') for f in glob.glob(p))
        assert files, it
        c = curve(files, T)
        assert abs(c['ln_weight']-math.log(2)) < 1e-5, (it, c['ln_weight'])     # exact to 1e-12 before the weights of converged levels are replaced
        f = features(T, c)
        res.append(dict(N=int(N), alpha=float(alpha), files=[os.path.basename(x) for x in files], blocks=c['blocks'], E0=c['E0'],
                        **f, C=c['C'].tolist(), S=c['S'].tolist(), dC=c['dC'].tolist()))
        g = lambda k, fmt='.4f': format(f[k], fmt) if f.get(k) is not None else '—'
        print(f"N = {N:>2}, alpha = {float(alpha):<5}: main {g('T_h', '.3f')} / {g('C_h')};  T* {g('T_star')} / {g('C_star')} (± {g('dC_star')});  "
              f"dip {g('T_d')} / {g('C_d')}, S_d {g('S_d')};  contrast {g('contrast')} {'' if f.get('resolved', True) else '(not resolved)'};  T_L {g('T_L')}")
    json.dump(dict(T=T.tolist(), S_ising=S_ISING, curves=res), open(a.out, 'w'))
    print('wrote', a.out)
    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                             'legend.fontsize': 7, 'legend.frameon': False, 'savefig.dpi': 300, 'axes.linewidth': 0.6})
        Ns = sorted({r['N'] for r in res}); alphas = sorted({r['alpha'] for r in res})
        cmap = plt.get_cmap('viridis'); col = {al: cmap(0.92*i/max(1, len(alphas)-1)) for i, al in enumerate(alphas)}
        fig, axs = plt.subplots(1, len(Ns)+2, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.25]*len(Ns)+[1, 1], wspace=0.42))
        for ax, N, letter in zip(axs, Ns, 'abcdef'):
            for r in sorted((r for r in res if r['N'] == N), key=lambda r: r['alpha']):
                ax.plot(T, r['C'], '-', color=col[r['alpha']], lw=0.9, label=f"{r['alpha']:g}")
            ax.set_xscale('log'); ax.set_xlim(0.006, 3); ax.set_ylim(0, 0.21)
            ax.set_xlabel('$T$')
            ax.text(0.04, 0.95, letter, transform=ax.transAxes, fontweight='bold', fontsize=9, va='top')
        axs[0].set_ylabel('$C/N$')
        h, l = [], []
        for ax in axs[:len(Ns)]:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        order = np.argsort([float(x) for x in l])
        fig.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=len(l), bbox_to_anchor=(0.5, 1.1),
                   handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
        mk = {N: m for N, m in zip(Ns, ('o', 's', '^', 'D'))}
        gray = {N: g for N, g in zip(Ns, ('#b0b0b0', '#707070', '#111111', '#c0392b'))}
        ax = axs[len(Ns)]
        for N in Ns:
            rr = sorted((r for r in res if r['N'] == N and r.get('T_star') and r.get('resolved', False)), key=lambda r: r['alpha'])
            ax.plot([r['alpha'] for r in rr], [r['T_star'] for r in rr], mk[N]+'-', color=gray[N], ms=3, lw=0.7, mew=0)
            ru = [r for r in res if r['N'] == N and r.get('T_star') and not r.get('resolved', False)]
            ax.plot([r['alpha'] for r in ru], [r['T_star'] for r in ru], mk[N], color=gray[N], ms=3, mfc='white', mew=0.7)
        ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('$T^*$'); ax.set_xlim(0, 1.05); ax.set_ylim(0, None)
        ax.text(0.06, 0.95, 'abcdef'[len(Ns)], transform=ax.transAxes, fontweight='bold', fontsize=9, va='top')
        ax = axs[len(Ns)+1]
        for N in Ns:
            rr = sorted((r for r in res if r['N'] == N and r.get('S_d') is not None), key=lambda r: r['alpha'])
            ax.plot([r['alpha'] for r in rr], [r['S_d'] for r in rr], mk[N]+'-', color=gray[N], ms=3, lw=0.7, mew=0)
        ax.axhline(S_ISING, color='#2471a3', lw=0.7, dashes=(3, 2))
        ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('entropy at the minimum'); ax.set_xlim(0, 1.05); ax.set_ylim(0, 0.6)
        ax.text(0.06, 0.12, 'abcdef'[len(Ns)+1], transform=ax.transAxes, fontweight='bold', fontsize=9, va='top')
        for ax in axs:
            ax.tick_params(length=2.5, width=0.6)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        fig.savefig(a.fig, bbox_inches='tight')
        print('wrote', a.fig)
