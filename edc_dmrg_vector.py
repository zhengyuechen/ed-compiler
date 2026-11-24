#!/usr/bin/env python3
"""A state of the layer DMRG as a vector in the product basis, and exact references, for small cylinders (N <= 26):
checks with the original Hamiltonian written on the spins, independent of the DMRG environments.

    v = to_vector(eng)                      complex amplitudes over the states of the S_z sector (order: sector_states)
    codes, H = sector_hamiltonian(lat, nup) the XXZ model on lat.global_bonds() in that sector (sparse, real)
    E, res = measure(v, H)                  <v|H|v> / <v|v> and |H v - E v| / |v|: energy and residual of the WHOLE state
    w, V = exact_states(lat, nup, K, k)     the k lowest levels of momentum K and their vectors (rows of V)

Layer basis -> spins: layer l holds the spins offset[l] + a (bit a of the layer code); column s of kinds[l].Ur is the
layer state s of the DMRG (label (m, k)) in the product basis of the layer.  The tensors are real, the vectors complex."""
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import edc_dmrg_state as st
from edc_dmrg_excited import _slabs


def sector_states(N, nup):
    c = np.array([sum(1 << b for b in cb) for cb in itertools.combinations(range(N), nup)], dtype=np.int64); c.sort()
    return c


def to_vector(eng):
    n = eng.n; md = eng.model; lat = md.lat; assert lat.N <= 26, 'small cylinders only'
    A = np.ones((1, 1))
    for r in range(n):
        p = eng.plan[r]; M = np.asarray(st._host(eng.M[r]), dtype=float); S = p.c.S
        lo = np.concatenate([[0], np.cumsum(p.DL)]).astype(int); ro = np.concatenate([[0], np.cumsum(p.DR)]).astype(int); T = np.zeros((int(lo[-1]), S, int(ro[-1])))
        for i, s, j, c0, w in _slabs(p):
            T[lo[i]:lo[i+1], s, ro[j]:ro[j]+w] = M[p.Moff[i]:p.Moff[i+1]].reshape(int(p.DL[i]), int(p.C[i]))[:, c0:c0+w]
        A = np.tensordot(A, T, axes=([-1], [0]))
    A = A.reshape(A.shape[1:-1]).astype(complex)                       # (S_0, ..., S_{n-1}): amplitudes in the layer bases
    for l in range(n):
        A = np.moveaxis(np.tensordot(md.kinds[md.site_kind[l]].Ur, A, axes=([1], [l])), 0, l)
    full = A.transpose(tuple(range(n-1, -1, -1))).ravel()              # index = sum_l code_l << offset[l]
    nup = int(eng.bond[n].m[0]); codes = sector_states(lat.N, nup); v = full[codes]
    assert abs(np.vdot(full, full).real - np.vdot(v, v).real) < 1e-9, 'weight outside the S_z sector'
    return v


def sector_hamiltonian(lat, nup, Delta=1.0):
    N = lat.N; codes = sector_states(N, nup); n = len(codes); diag = np.zeros(n); rows, cols, vals = [], [], []
    for i, j, J in lat.global_bonds():
        bi, bj = (codes >> i) & 1, (codes >> j) & 1; diag += J*Delta*np.where(bi == bj, 0.25, -0.25)
        sel = np.nonzero(bi != bj)[0]; rows.append(np.searchsorted(codes, codes[sel] ^ ((1 << i) | (1 << j)))); cols.append(sel); vals.append(np.full(len(sel), 0.5*J))
    return codes, (sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) + sp.diags(diag)).tocsr()


def measure(v, H):
    n2 = np.vdot(v, v).real; w = H @ v; E = np.vdot(v, w).real/n2
    return float(E), float(np.linalg.norm(w - E*v)/np.sqrt(n2))


def momentum_projector(lat, codes, K):
    """v -> its component with T v = exp(-2 pi i K / c) v (the convention of the layer basis)."""
    N = lat.N; c = lat.c; tp = lat.translation(); t = np.zeros(len(codes), dtype=np.int64)
    for a in range(N):
        t |= ((codes >> a) & 1) << tp[a]
    perm = np.searchsorted(codes, t)

    def proj(v):
        acc = v.astype(complex); w = v
        for s in range(1, c):
            u = np.empty_like(w); u[perm] = w; w = u; acc = acc + np.exp(2j*np.pi*K*s/c)*w
        return acc/c
    return proj


def exact_states(lat, nup, K, k=4, Delta=1.0, tol=1e-12, shift=None):
    """The k lowest levels with momentum K and their vectors (rows).  Lanczos on P H P - shift P (shift above the
    wanted levels, default 0: fine while they are negative)."""
    codes, H = sector_hamiltonian(lat, nup, Delta); proj = momentum_projector(lat, codes, K); n = len(codes)
    op = sla.LinearOperator((n, n), matvec=lambda v: proj(H @ proj(v)), dtype=complex)
    v0 = proj(np.random.default_rng(K).standard_normal(n).astype(complex))
    w, V = sla.eigsh(op, k=k, which='SA', v0=v0, tol=tol, ncv=max(4*k, 24)); o = np.argsort(w)
    return w[o], V[:, o].T.copy(), H, proj
