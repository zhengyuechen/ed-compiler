#!/usr/bin/env python3
"""Independent check of edc_hubbard: dense Fock-space Hamiltonian with interleaved ordering (0up,0dn,1up,...)."""
import numpy as np, edc_hubbard as eh

def dense_ground(model, nup, ndn):
    L = model.n; M = 2*L
    I, Z = np.eye(2), np.diag([1., -1.])
    a = np.array([[0., 1.], [0., 0.]])           # annihilation on one mode, basis (empty, full)
    def ann(k):
        ops = [Z]*k + [a] + [I]*(M-k-1)
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out
    c = [ann(k) for k in range(M)]
    mode = lambda i, s: 2*i+s
    H = np.zeros((2**M,)*2)
    for (i, j, t) in model.hops:
        for s in (0, 1):
            ci, cj = c[mode(i, s)], c[mode(j, s)]
            H -= t*(ci.T @ cj + cj.T @ ci)
    for i in range(L):
        H += model.U * (c[mode(i, 0)].T @ c[mode(i, 0)]) @ (c[mode(i, 1)].T @ c[mode(i, 1)])
    nu = sum(c[mode(i, 0)].T @ c[mode(i, 0)] for i in range(L)).diagonal()
    nd = sum(c[mode(i, 1)].T @ c[mode(i, 1)] for i in range(L)).diagonal()
    sel = np.flatnonzero((np.round(nu) == nup) & (np.round(nd) == ndn))
    return np.linalg.eigvalsh(H[np.ix_(sel, sel)])[0]

for model, name, nu, nd in ((eh.chain(6, 1.0, 4.0), 'chain 6, U=4, 3+3', 3, 3),
                            (eh.chain(6, 1.0, 8.0), 'chain 6, U=8, 2+3', 2, 3),
                            (eh.chain(5, 1.0, 2.0), 'chain 5, U=2, 2+2', 2, 2),
                            (eh.square(3, 2, 1.0, 6.0), 'square 3x2, U=6, 3+2', 3, 2)):
    ref = dense_ground(model, nu, nd)
    e = eh.HubbardEngine(model, nu, nd, threads=2)
    Ek = e.ground_energy(tol=1e-13)[0]; Ef = e.ground_energy(tol=1e-13, kernel='flat')[0]
    print(f'{name:24s} dim {e.dim:6d}  kron {Ek:.12f}  flat {Ef:.12f}  dense {ref:.12f}  diff {max(abs(Ek-ref), abs(Ef-ref)):.1e}')
