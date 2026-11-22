#!/usr/bin/env python3
"""Correctness checks for edc against exact values and dense diagonalisation."""
import sys, time
import numpy as np
import edc


def dense_energy(model, nup):
    """Brute-force reference: build H in the Sz sector by explicit loops over bonds."""
    L = model.n
    states = [s for s in range(1 << L) if bin(s).count('1') == nup]
    index = {s: k for k, s in enumerate(states)}
    H = np.zeros((len(states), len(states)))
    for k, s in enumerate(states):
        for (i, j), (jxy, jz) in model.bonds.items():
            si, sj = (s >> i) & 1, (s >> j) & 1
            H[k, k] += jz*(0.25 if si == sj else -0.25)
            if si != sj and jxy:
                H[index[s ^ ((1 << i) | (1 << j))], k] += jxy/2
        for i in range(L):
            H[k, k] += model.h[i]*(((s >> i) & 1)-0.5)
    return np.linalg.eigvalsh(H)[0]


def check(name, model, nup, group=None, ref=None, stored=True):
    e = edc.Engine(model, nup, group=group)
    E, it, t = e.ground_energy(tol=1e-13)
    out = f'{name:34s} dim {e.dim:>9,}  E0 {E:.12f}  ({it} it)'
    if stored and e.gen.uniform_coef:
        e.build_stored()
        Es, its, _ = e.ground_energy(tol=1e-13, stored=True)
        out += f'  stored {Es:.12f}'
    else:
        Es = E
    if ref is not None:
        out += f'  ref {ref:.12f}  diff {max(abs(E-ref), abs(Es-ref)):.1e}'
    print(out, flush=True)
    return E


rng = np.random.default_rng(3)
# 1. Heisenberg ring, no symmetry and with translations
for L in (10, 12):
    m = edc.ring(L)
    ref = dense_energy(m, L//2)
    check(f'ring L={L}', m, L//2, ref=ref)
    # rings with L = 4n+2 have their ground state at k = pi; the k = 0 reference is then the dense k = 0 value
    ref0 = ref if L % 4 == 0 else -4.09220734673867   # L = 10, k = 0 (dense projection)
    check(f'ring L={L}, k=0', m, L//2, group=edc.ring_translations(L), ref=ref0)
# 2. J1-J2 ring at the Majumdar-Ghosh point: E0 = -3L/8 exactly
for L in (12, 16):
    check(f'J1-J2 ring L={L}, J2=1/2', edc.ring(L, 1.0, 0.5), L//2, ref=-3*L/8)
    check(f'J1-J2 ring L={L}, J2=1/2, k=0', edc.ring(L, 1.0, 0.5), L//2, group=edc.ring_translations(L), ref=-3*L/8)
# 3. square lattice 4x4: E0 = -11.228483208428 (well-known value)
check('square 4x4', edc.square(4, 4), 8, ref=-11.228483208428)
check('square 4x4, translations', edc.square(4, 4), 8, group=edc.torus_translations(4, 4), ref=-11.228483208428)
check('square 4x3 (dense ref)', edc.square(4, 3), 6, ref=dense_energy(edc.square(4, 3), 6))
# 4. random couplings and random field (non-uniform coefficients, matrix-free only)
m = edc.SpinHalfModel(12)
for i in range(12):
    m.exchange(i, (i+1) % 12, rng.uniform(0.5, 1.5))
    m.zz(i, (i+1) % 12, rng.uniform(0.5, 1.5))
    m.heisenberg(i, (i+3) % 12, rng.uniform(-0.3, 0.3))
    m.field(i, rng.uniform(-1, 1))
check('random XXZ + field L=12', m, 6, ref=dense_energy(m, 6))
check('random XXZ + field L=12, nup=5', m, 5, ref=dense_energy(m, 5))
