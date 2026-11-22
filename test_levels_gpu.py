#!/usr/bin/env python3
"""Two lowest distinct levels per sector (lean GPU engine, nev = 2) vs dense diagonalisation with projectors."""
import numpy as np, edc_lattice as el, edc_gpu_lean as gl
from test_lattice import dense_by_momentum  # noqa: F401  (module runs its own CPU checks on import)

def dense_levels(model, nup, perms, chars, nev=2):
    L = model.n
    states = [s for s in range(1 << L) if bin(s).count('1') == nup]
    idx = {s: k for k, s in enumerate(states)}
    H = np.zeros((len(states),)*2)
    for k, s in enumerate(states):
        for (i, j), (jxy, jz) in model.bonds.items():
            a, b = (s >> i) & 1, (s >> j) & 1
            H[k, k] += jz*(0.25 if a == b else -0.25)
            if a != b and jxy: H[idx[s ^ ((1 << i) | (1 << j))], k] += jxy/2
    P = np.zeros(H.shape, complex)
    for p, c in zip(perms, chars):
        for k, s in enumerate(states):
            t = sum(((s >> i) & 1) << p[i] for i in range(L)); P[idx[t], k] += np.conj(c)
    P /= len(perms)
    w, v = np.linalg.eigh((P+P.conj().T)/2); B = v[:, w > 0.5]
    if not B.shape[1]:
        return []
    ev = np.linalg.eigvalsh(B.conj().T @ H @ B)
    out = []
    for e in ev:
        if not out or abs(e-out[-1]) > 1e-9:
            out.append(e)
    return out[:nev]

for name, cl, J2 in (('tilted (3,1)', el.tilted(3, 1), 0.3), ('4x3', el.Cluster((4, 0), (0, 3)), 0.6)):
    m = cl.j1j2(1.0, J2); worst, n = 0.0, 0
    for k in cl.momenta():
        lg = cl.little_group(k)
        for ir in ('A1', 'A2', 'B1', 'B2', 'E+', 'E-'):
            if [o for o in lg if o in el.IRREPS[ir]] != lg:
                continue
            perms, chars, _ = cl.group(k, ir)
            ref = dense_levels(m, cl.N//2, perms, chars)
            g = gl.GPULean(m, cl.N//2, cl, k, ir)
            if len(ref) < 2:
                continue
            E = g.ground_energy(tol=1e-13, nev=2)[0]
            worst = max(worst, max(abs(a-b) for a, b in zip(E, ref))); n += 1
    print(f'{name:14s} J2={J2}: {n} sectors, max |E_i - dense_i| (i = 0, 1) {worst:.1e}', flush=True)
