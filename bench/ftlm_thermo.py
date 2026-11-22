#!/usr/bin/env python3
"""Thermodynamics from ftlm.py output (one S_z sector): Z(T) = sum_blocks mult D/R sum_r sum_j |U^r_0j|^2 exp(-E^r_j/T),
energy and specific heat per site; checks that sum mult x dim equals the sector dimension.
Usage: ftlm_thermo.py N nup file.jsonl [...]"""
import sys, json, math
import numpy as np
from scipy.linalg import eigh_tridiagonal

N, nup = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for f in sys.argv[3:] for l in open(f) if l.strip()]
tot = sum(r['mult']*r['dim'] for r in rows)
assert tot == math.comb(N, nup), (tot, math.comb(N, nup))
levels = []
for r in rows:
    if not r['dim']:
        continue
    w = r['mult']*r['dim']/r['R']
    for A, B in zip(r['alphas'], r['betas']):
        E, U = eigh_tridiagonal(np.array(A), np.array(B[:-1]))
        levels.append((E, w*U[0]**2))
Emin = min(E.min() for E, _ in levels)
out = dict(blocks=len(rows), sector_dim=tot, lowest_ritz=Emin, T=[], E_per_site=[], C_per_site=[])
for T in np.geomspace(0.02, 10, 40):
    z = sum((c*np.exp(-(E-Emin)/T)).sum() for E, c in levels)
    e1 = sum((c*E*np.exp(-(E-Emin)/T)).sum() for E, c in levels)/z
    e2 = sum((c*E*E*np.exp(-(E-Emin)/T)).sum() for E, c in levels)/z
    out['T'].append(float(T)); out['E_per_site'].append(e1/N); out['C_per_site'].append((e2-e1*e1)/(T*T)/N)
print(json.dumps(out))
