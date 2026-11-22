#!/usr/bin/env python3
"""Triangular, honeycomb and kagome clusters (edc_lattices): CPU any-group engine (edc_k) against dense projectors.
(1) Rotation subgroup only (abelian, all irreps 1D): the sectors' dimensions add up to the full S_z space and their
spectra together equal the full spectrum.  (2) Full little group, 1D irreps: every sector's lowest level."""
import math
import numpy as np
import edc_k, edc_lattices as L


def dense(model, nup):
    n = model.n
    states = np.array([s for s in range(1 << n) if bin(s).count('1') == nup])
    idx = {int(s): i for i, s in enumerate(states)}
    H = np.zeros((len(states),)*2)
    for i, s in enumerate(states):
        for (a, b), (jxy, jz) in model.bonds.items():
            x, y = (s >> a) & 1, (s >> b) & 1
            H[i, i] += jz*(0.25 if x == y else -0.25)
            if x != y and jxy:
                H[idx[int(s ^ ((1 << a) | (1 << b)))], i] += jxy/2
    return states, idx, H


def sector_spectrum(states, idx, H, perms, chars):
    P = np.zeros(H.shape, complex)
    cols = np.arange(len(states))
    for p, c in zip(perms, chars):
        img = np.zeros_like(states)
        for a, b in enumerate(p):
            img |= ((states >> a) & 1) << b
        P[[idx[int(t)] for t in img], cols] += np.conj(c)
    P /= len(perms)
    w, v = np.linalg.eigh((P+P.conj().T)/2); B = v[:, w > 0.5]
    return np.linalg.eigvalsh(B.conj().T @ H @ B) if B.shape[1] else np.array([])


CASES = [('triangular', (2, 2), (-2, 4), (1.0, 0.3), None), ('kagome', (2, 0), (0, 2), (1.0,), None),
         ('kagome', (1, 1), (-1, 2), (1.0, 0.2), 4), ('honeycomb', (2, 0), (0, 2), (1.0, 0.25), None),
         ('honeycomb', (1, 1), (-1, 2), (1.0,), None), ('square', (4, 0), (0, 3), (1.0, 0.4), None),
         ('triangular', (3, 0), (0, 4), (1.0,), None)]
for kind, A1, A2, J, nup in CASES:
    cl = L.LatticeCluster(kind, A1, A2)
    nup = nup if nup is not None else cl.N//2
    m = cl.heisenberg(*J)
    states, idx, H = dense(m, nup)
    full = np.linalg.eigvalsh(H)
    # (1) rotations only
    rot = cl.rotations(); spec, dsum = [], 0
    for k in cl.momenta():
        for ir in cl.irrep_table(k, rot):
            perms, chars, ops = cl.group(k, ir, allowed=rot)
            ev = sector_spectrum(states, idx, H, perms, chars); spec.append(ev); dsum += len(ev)
    allev = np.sort(np.concatenate(spec))
    c1 = f'dims {dsum}/{len(states)}, spectrum {np.max(np.abs(allev-full)):.1e}' if dsum == len(states) else f'DIMS {dsum}/{len(states)}'
    # (2) full little group, 1D irreps, CPU engine
    worst, n = 0.0, 0
    for k in cl.momenta():
        for ir in cl.irrep_table(k):
            perms, chars, ops = cl.group(k, ir)
            ev = sector_spectrum(states, idx, H, perms, chars)
            e = edc_k.KEngine(m, nup, perms, chars)
            assert e.dim == len(ev), (kind, k, ir, e.dim, len(ev))
            if len(ev):
                worst = max(worst, abs(e.ground_energy(tol=1e-13)[0]-ev[0])); n += 1
    print(f'{kind:10s} {A1} {A2} N={cl.N:2d} nup={nup} J={J}: rotations: {c1}; full little group: {n} sectors, '
          f'max |E_engine - dense| {worst:.1e}; E0 = {full[0]:.12f}', flush=True)
