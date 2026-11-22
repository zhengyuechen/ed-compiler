#!/usr/bin/env python3
"""Static spin structure factor of a ground state computed by a symmetric engine (edc_sub.SubEngine).

    S(Q) = (1/N) sum_{i,j} cos(Q.(r_i - r_j)) <S_i.S_j>,      M^2(Q) = S(Q) / (N + 2)   (Richter & Schulenburg 2010)

with r_i the Cartesian site positions (basis offsets included, so kagome and honeycomb peaks sit in the extended
zone).  The operator O = sum_{i<j} c_ij S_i.S_j, c_ij = mean over the star of Q of cos(Q.(r_i - r_j)), commutes with
every lattice symmetry, so it acts inside a symmetry sector; S(Q) = (3N/4 + 2 <O>) / N.  Q must be compatible with
the cluster (Q.A integer multiple of 2 pi for both periodic vectors A), otherwise r_i - r_j is not well defined."""
import math
import numpy as np
import edc


def positions(cl):
    if hasattr(cl, 'cartesian'):
        return np.array([cl.cartesian(t) for t in range(cl.N)])
    return np.array(cl.coord, float)


def periodic_vectors(cl):
    C = cl.cart if hasattr(cl, 'cart') else np.eye(2)
    return [C @ cl.M[:, c].astype(float) for c in range(2)]


def reciprocal(cl):
    """Cartesian reciprocal vectors b1, b2 of the Bravais lattice (b_i . a_j = 2 pi delta_ij)."""
    C = cl.cart if hasattr(cl, 'cart') else np.eye(2)
    B = 2*math.pi*np.linalg.inv(C).T
    return B[:, 0], B[:, 1]


def point_matrices(cl):
    """Cartesian matrices of the cluster's point operations."""
    if hasattr(cl, 'cart'):
        C, Ci = cl.cart, np.linalg.inv(cl.cart)
        return [C @ cl.ops[o].astype(float) @ Ci for o in cl.point_ops()]
    import edc_lattice as el
    return [np.array(el.D4[o], float) for o in cl.point_ops()]


def star(cl, Q):
    Q = np.array(Q, float)
    out = []
    for R in point_matrices(cl):
        q = R @ Q
        if not any(np.allclose(q, p, atol=1e-9) for p in out):
            out.append(q)
    return out


def compatible(cl, Q):
    return all(abs(x-round(x)) < 1e-9 for x in (np.dot(Q, A)/(2*math.pi) for A in periodic_vectors(cl)))


def structure_factor_model(cl, Q):
    assert compatible(cl, Q), f'Q = {Q} is not a momentum of this cluster'
    Qs = star(cl, Q)
    r = positions(cl)
    m = edc.SpinHalfModel(cl.N)
    for i in range(cl.N):
        d = r[i+1:]-r[i]
        c = np.mean([np.cos(d @ q) for q in Qs], axis=0)
        for j, cj in zip(range(i+1, cl.N), c):
            if abs(cj) > 1e-12:
                m.heisenberg(i, j, float(cj))
    return m, Qs


def structure_factor(engine, psi, cl, Q):
    """S(Q) and M^2(Q) of the normalised sector vector psi (from edc_hubbard_sym.ground_state_vector)."""
    from edc_hubbard_sym import _vec_ops
    O, Qs = structure_factor_model(cl, Q)
    apply = engine.operator(O)
    xp = engine.xp
    y = xp.zeros_like(psi)
    apply(psi, y)
    _, dot = _vec_ops(xp, engine.real)
    val = dot(psi, y).real
    del y
    N = cl.N
    S = (0.75*N+2*val)/N
    return dict(Q=[float(v) for v in Q], star=len(Qs), S=S, M2=S/(N+2))
