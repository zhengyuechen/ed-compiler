#!/usr/bin/env python3
"""Independent check of one momentum sector: Lanczos in the full Sz sector (no symmetry tables), re-projecting onto
momentum q with explicit translation permutations at every step.  Compared with the CPU any-momentum engine."""
import sys, numpy as np, edc, edc_k
from scipy.linalg import eigvalsh_tridiagonal
L, nup = int(sys.argv[1]), int(sys.argv[2]); qs = [int(q) for q in sys.argv[3].split(',')]
m = edc.ring(L)
e = edc.Engine(m, nup, threads=8)
st = e.st.astype(np.int64)
rot = ((st << 1) | (st >> (L-1))) & ((1 << L)-1)               # site i -> i+1
perm = np.searchsorted(st, rot)                                  # index of T|s>
for q in qs:
    w = np.exp(-2j*np.pi*q/L)
    def project(v):
        out = np.zeros_like(v); cur = v.copy()
        for k in range(L):
            out += np.conj(w**k)*cur
            nxt = np.empty_like(cur); nxt[perm] = cur; cur = nxt  # apply T once more
        return out/L
    rng = np.random.default_rng(1)
    v = project(rng.standard_normal(e.dim)+1j*rng.standard_normal(e.dim)); v /= np.linalg.norm(v)
    vp = np.zeros_like(v); A, B = [], []; b = 0.0; Eold = None
    for it in range(400):
        hv = e.matvec(v.real.copy())+1j*e.matvec(v.imag.copy())
        a = np.vdot(v, hv).real; wv = project(hv-a*v-b*vp); b2 = np.linalg.norm(wv)
        A.append(a); B.append(b2)
        if it > 2:
            E = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0]
            if Eold is not None and abs(E-Eold) < 1e-11:
                break
            Eold = E
        vp, v, b = v, wv/b2, b2
    k = edc_k.KEngine(m, nup, edc.ring_translations(L), edc_k.ring_characters(L, q), threads=8)
    print(f'L={L} nup={nup} q={q}: projected Lanczos {E:.10f}   any-momentum engine {k.ground_energy()[0]:.10f}   dims full {e.dim} sector {k.dim}')
