#!/usr/bin/env python3
"""Thermodynamics from thermo_run.py output (all S_z sectors with N_up >= N/2; the sectors with N_up < N/2 are their
mirror images).  Every block contributes levels (E, w): a dense block its eigenvalues with weight mult; a Lanczos chain
the Ritz values of its tridiagonal matrix with weight mult * dim / R * |first component|^2.  In a field h along z the
level of a sector with magnetisation M = N_up - N/2 moves to E - h M.
  Z = sum w exp(-(E - h M)/T),  C = (<H^2> - <H>^2)/T^2,  S = ln Z + <H>/T,  chi = (<M^2> - <M>^2)/T,  m = <M>/(N/2)
(per site where it says so; k_B = 1, g mu_B = 1).  The chains are also split into two halves (even and odd chain index)
and everything is computed from each half alone: the difference of the halves shows the statistical error.

Usage: thermo_post.py out.json file.jsonl [...] [--T lo:hi:n] [--h h1,h2,..] [--allow-missing] [--fix]
As a module: load(files) -> Data; Data.thermo(T, h, which='all')."""
import sys, json, math, argparse
import numpy as np
from scipy.linalg import eigh_tridiagonal


class Data:
    def __init__(self, files, allow_missing=False, fix=False):
        self.fix = fix; self._pairs = {}; self.fixed_levels = 0
        recs = {}
        for f in files:
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                                             # a line cut by a time limit
                recs[(r['nup'], r['block'])] = r
        self.recs = recs
        self.N = N = next(iter(recs.values()))['N']; self.J = next(iter(recs.values()))['J']
        assert all(r['N'] == N and r['J'] == self.J for r in recs.values()), 'files of different models'
        self.missing = []
        for nup in range(N//2, N+1):
            tot = sum(r['mult']*r['dim'] for (n, _), r in recs.items() if n == nup)
            if tot != math.comb(N, nup):
                self.missing.append((nup, tot, math.comb(N, nup)))
        assert allow_missing or not self.missing, f'incomplete sectors (nup, found, expected): {self.missing}'
        self._flat = {}

    def levels(self, which='all'):
        """[(M, E, w)] for which = 'all', 0 (even chains) or 1 (odd chains)."""
        out = []
        N = self.N
        for (nup, _), r in sorted(self.recs.items()):
            if not r['dim']:
                continue
            M = nup-N//2
            if 'eigs' in r:
                out.append((M, np.array(r['eigs']), np.full(len(r['eigs']), float(r['mult']))))
                continue
            ch = list(range(r['R'])) if which == 'all' else [c for c in range(r['R']) if c % 2 == which]
            if not ch:
                ch = list(range(r['R']))
            w0 = r['mult']*r['dim']/len(ch)
            pairs = self._ritz(r)
            for c in ch:
                E, p = pairs[c]
                out.append((M, E, w0*p))
        return out

    def _ritz(self, r):
        """Ritz values and weights |first component|^2 of every chain of a block.  With self.fix: a Ritz value that
        every chain of the block has found (agreement within 1e-7) is a converged level; its weight, summed over the
        copies a chain holds of it, is replaced by its expectation 1/dim (a normalised eigenvector has squared overlap
        1/dim with a random vector on average).  This removes the statistical error of the levels that dominate at low
        temperature; it assumes that such a level is not degenerate inside its block."""
        key = (r['nup'], r['block'])
        if key in self._pairs:
            return self._pairs[key]
        pairs = []
        for A, B in zip(r['alphas'], r['betas']):
            A = np.array(A); B = np.array(B)
            scale = max(1.0, np.abs(A).max())
            cut = np.nonzero(B[:-1] < 1e-10*scale)[0]
            m = int(cut[0])+1 if len(cut) else len(A)
            E, U = eigh_tridiagonal(A[:m], B[:m-1])
            pairs.append([E, U[0]**2])
        nfix = 0
        if self.fix and len(pairs) > 1:
            E0s = pairs[0][0]; tol = 1e-7*max(1.0, abs(E0s[0]))
            cand = []
            for x in E0s[:40]:
                if cand and abs(x-cand[-1]) < tol:
                    continue
                if all(np.abs(E-x).min() < tol for E, _ in pairs[1:]):
                    cand.append(float(x))
            for x in cand:
                for E, p in pairs:
                    sel = np.abs(E-x) < tol
                    p[sel] = (1.0/r['dim'])/sel.sum()
            nfix = len(cand)
        self.fixed_levels += nfix
        self._pairs[key] = pairs
        return pairs

    def flat(self, which='all'):
        """(M, E, w) as flat arrays over all levels, both signs of M."""
        if which not in self._flat:
            lv = self.levels(which)
            sg = lambda M: (1, -1) if M else (1,)
            self._flat[which] = (np.concatenate([np.full(len(E), s*M) for M, E, w in lv for s in sg(M)]).astype(float),
                                 np.concatenate([E for M, E, w in lv for s in sg(M)]),
                                 np.concatenate([w for M, E, w in lv for s in sg(M)]))
        return self._flat[which]

    def thermo(self, T, h=0.0, which='all'):
        """dict of arrays over T at field h: E, C, S, chi (per site), m = <M>/(N/2); and the lowest level E0."""
        Ms, Es, Ws = self.flat(which)
        N = self.N
        Eh = Es-h*Ms
        E0 = Eh.min()
        res = dict(E=[], C=[], S=[], chi=[], m=[])
        for t in np.atleast_1d(T):
            p = Ws*np.exp(-(Eh-E0)/t); z = p.sum(); p /= z
            e1 = (p*Eh).sum(); e2 = (p*Eh*Eh).sum(); m1 = (p*Ms).sum(); m2 = (p*Ms*Ms).sum()
            res['E'].append(e1/N); res['C'].append((e2-e1*e1)/(t*t)/N); res['S'].append((math.log(z)+(e1-E0)/t)/N)
            res['chi'].append((m2-m1*m1)/t/N); res['m'].append(m1/(N/2))
        return {k: np.array(v) for k, v in res.items()}, float(E0)

    def summary(self):
        Ms, Es, Ws = self.flat('all')
        lv = self.levels('all')
        W = Ws.sum(); m1 = (Ws*Es).sum()/W; m2 = (Ws*Es*Es).sum()/W
        return dict(N=self.N, J=self.J, blocks=len(self.recs), lanczos_blocks=sum('alphas' in r for r in self.recs.values()),
                    dense_blocks=sum('eigs' in r for r in self.recs.values()), missing=self.missing,
                    E_min_by_M={str(M): float(min(E.min() for m_, E, w in lv if m_ == M)) for M in sorted({m_ for m_, _, _ in lv})},
                    ln_total_weight_per_site=math.log(W)/self.N,           # must be ln 2
                    var_H_infinite_T_per_site=(m2-m1*m1)/self.N)           # Heisenberg bonds: 3/16 sum J_b^2 / N


def load(files, allow_missing=False, fix=False):
    return Data(files, allow_missing, fix)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('files', nargs='+')
    ap.add_argument('--T', default='0.01:10:241'); ap.add_argument('--h', default='0')
    ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--fix', action='store_true', help='exact weight for the levels every chain of a block has converged')
    args = ap.parse_args()
    d = load(args.files, args.allow_missing, args.fix)
    lo, hi, n = args.T.split(':'); T = np.geomspace(float(lo), float(hi), int(n))
    hs_ = [float(v) for v in args.h.split(',')]
    out = d.summary(); out['T'] = T.tolist(); out['fields'] = {}
    for h in hs_:
        full, E0 = d.thermo(T, h)
        a, _ = d.thermo(T, h, 0); b, _ = d.thermo(T, h, 1)
        out['fields'][str(h)] = dict(E0=E0, **{k: v.tolist() for k, v in full.items()},
                                     **{k+'_half_diff': (0.5*np.abs(a[k]-b[k])).tolist() for k in full})
    json.dump(out, open(args.out, 'w'))
    f0 = out['fields'][str(hs_[0])]
    C = np.array(f0['C']); i = int(np.argmax(np.where(T > 0.3, C, -1)))
    print(f"N {out['N']} J {out['J']} blocks {out['blocks']} (Lanczos {out['lanczos_blocks']}, dense {out['dense_blocks']}) missing {out['missing']}")
    print(f"ln(total weight)/N {out['ln_total_weight_per_site']:.12f} (ln 2 = {math.log(2):.12f})")
    print(f"var(H)/N at infinite T {out['var_H_infinite_T_per_site']:.6f}")
    print(f"lowest level by M: {dict(list(out['E_min_by_M'].items())[:4])}")
    print(f"h = {hs_[0]}: main maximum of C/N: T = {T[i]:.4f}, C/N = {C[i]:.5f}; half difference there {f0['C_half_diff'][i]:.2e}")
