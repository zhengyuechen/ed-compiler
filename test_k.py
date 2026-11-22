#!/usr/bin/env python3
"""Check edc_k at every momentum against dense block diagonalisation with translation projectors."""
import numpy as np, edc, edc_k

def dense_by_momentum(model, nup, perms, chars_of):
    L = model.n
    states = [s for s in range(1 << L) if bin(s).count('1') == nup]
    idx = {s: k for k, s in enumerate(states)}
    H = np.zeros((len(states),)*2)
    for k, s in enumerate(states):
        for (i, j), (jxy, jz) in model.bonds.items():
            a, b = (s >> i) & 1, (s >> j) & 1
            H[k, k] += jz*(0.25 if a == b else -0.25)
            if a != b and jxy: H[idx[s ^ ((1 << i) | (1 << j))], k] += jxy/2
        for i in range(L): H[k, k] += model.h[i]*(((s >> i) & 1)-0.5)
    Ts = []
    for p in perms:
        T = np.zeros_like(H)
        for k, s in enumerate(states):
            t = sum(((s >> i) & 1) << p[i] for i in range(L)); T[idx[t], k] = 1
        Ts.append(T)
    out = []
    for chars in chars_of:
        P = sum(np.conj(c)*T for c, T in zip(chars, Ts))/len(perms)
        w, v = np.linalg.eigh((P+P.conj().T)/2); B = v[:, w > 0.5]
        out.append((B.shape[1], np.linalg.eigvalsh(B.conj().T @ H @ B)[0] if B.shape[1] else None))
    return out

def run(name, model, nup, perms, chars_of):
    ref = dense_by_momentum(model, nup, perms, chars_of)
    tot, worst = 0, 0.0
    for q, (chars, (dref, eref)) in enumerate(zip(chars_of, ref)):
        e = edc_k.KEngine(model, nup, perms, chars, threads=2)
        E = e.ground_energy(tol=1e-13)[0]
        tot += e.dim
        assert e.dim == dref, (name, q, e.dim, dref)
        if dref:
            worst = max(worst, abs(E-eref))
    print(f'{name:28s} sectors {len(chars_of):2d}  sum of dims {tot}  max |E - dense| {worst:.1e}')

for L in (10, 12):
    run(f'ring L={L}', edc.ring(L), L//2, edc.ring_translations(L), [edc_k.ring_characters(L, q) for q in range(L)])
run('J1-J2 ring L=12, J2=0.3', edc.ring(12, 1.0, 0.3), 6, edc.ring_translations(12), [edc_k.ring_characters(12, q) for q in range(12)])
run('square 4x3', edc.square(4, 3), 6, edc.torus_translations(4, 3), [edc_k.torus_characters(4, 3, qx, qy) for qy in range(3) for qx in range(4)])
m = edc.ring(12); [m.field(i, 0.3) for i in range(12)]
run('ring L=12, uniform field, nup=5', m, 5, edc.ring_translations(12), [edc_k.ring_characters(12, q) for q in range(12)])
