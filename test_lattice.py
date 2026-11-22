#!/usr/bin/env python3
"""Cluster geometry + space-group 1D irreps at every momentum: CPU any-group engine (edc_k) vs dense projectors."""
import numpy as np, edc, edc_k, edc_lattice as el

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

def run(name, cl, J2):
    m = cl.j1j2(1.0, J2)
    ops_all = cl.point_ops()
    sectors = []
    for k in cl.momenta():
        lg = cl.little_group(k)
        for ir in ('A1', 'A2', 'B1', 'B2', 'E+', 'E-'):
            names = [o for o in lg if o in el.IRREPS[ir]]
            if names != lg:          # irrep must be defined on the whole little group
                continue
            if ir in ('E+', 'E-') and any(o in lg for o in ('Mx', 'My', 'Md', "Md'")):
                continue
            sectors.append((k, ir, lg))
    worst, nsec = 0.0, 0
    for k, ir, lg in sectors:
        perms, chars, _ = cl.group(k, ir, lg)
        ref = dense_by_momentum(m, cl.N//2, perms, [chars])[0]
        e = edc_k.KEngine(m, cl.N//2, perms, chars, threads=4)
        assert e.dim == ref[0], (name, k, ir, e.dim, ref[0])
        if e.dim:
            worst = max(worst, abs(e.ground_energy(tol=1e-13)[0]-ref[1])); nsec += 1
    print(f'{name:28s} N={cl.N:2d} n1={cl.n1:2d} m={cl.m}  point ops {cl.point_ops()}  sectors {nsec:3d}  max |E - dense| {worst:.1e}', flush=True)

run('square 4x3, J2=0.4', el.Cluster((4, 0), (0, 3)), 0.4)
run('square 3x4, J2=0.5', el.Cluster((3, 0), (0, 4)), 0.5)
run('tilted (3,1), J2=0.3', el.tilted(3, 1), 0.3)
run('tilted (2,2), J2=0.6', el.tilted(2, 2), 0.6)
run('tilted (3,1) J2=0', el.tilted(3, 1), 0.0)
# 4x4: no dense reference here; Gamma A1 must give the known energy
cl = el.Cluster((4, 0), (0, 4)); perms, chars, ops = cl.group((0.0, 0.0), 'A1')
e = edc_k.KEngine(cl.j1j2(1.0, 0.0), 8, perms, chars, threads=4)
print('square 4x4 Gamma A1: ops', ops, 'dim', e.dim, 'E0', e.ground_energy()[0], '(exact -11.228483208428)')
