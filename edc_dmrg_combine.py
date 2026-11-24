#!/usr/bin/env python3
"""Linear combinations of saved states of the layer DMRG, and the Hamiltonian in the span of a set of them.

    dims, tensors, norm = combine(eng, sources, coeffs)      the MPS of sum_a c_a |psi_a>, canonical with the centre at layer 0
    R = subspace(model, sources, cls)                        overlap matrix S_ab = <a|b>, H_ab = <a|H|b>, Ritz levels and vectors
    ritz_state(eng, sources, coefficients)                   one of the Ritz vectors installed in an engine (to measure or save)

sources: files of edc_dmrg_state (or engines), all of the same model and (S_z, K) sector.

How: the sum of two matrix-product states is a matrix-product state whose bond spaces are the direct sums of theirs
(per bond label: sizes add; the first tensor is the concatenation, weighted with the coefficients, the last one the
stack).  That state is brought to canonical form (centre at layer 0) without truncation beyond numerical rank, and then
it is an ordinary state of the engine: its norm and its energy come from the engine's own environments.  For a pair
    N+- = |a +- b|^2,  E+- = <a +- b|H|a +- b> / N+-
    <a|b> = (N+ - N-) / 4,      <a|H|b> = (N+ E+ - N- E-) / 4
with the checks (N+ + N-) / 2 = 2 and (N+ E+ + N- E-) / 2 = E_a + E_b.  No mirror map and no mixed contraction of two
different states with the Hamiltonian is needed.  The Ritz levels (generalised eigenvalues of H c = E S c) are upper
bounds of the lowest levels of the sector and resolve states that single DMRG runs mix.

Cost: the pair state has the bond sizes of both states together; one canonicalisation, one growth of the environments
and one product per sign.  Arrays are numpy for a host engine and cupy for a GPU back end with the tensors on the device."""
import copy
import numpy as np
import edc_dmrg as ed
import edc_dmrg_state as st
from edc_dmrg_excited import Stored, _slabs


def _xp(eng):
    if type(eng.M[0]) is not np.ndarray:
        import cupy as cp
        return cp
    return np


def _isqrt_basis(G, xp, cut, wmax=None):
    """Eigenvectors V (columns) and eigenvalues w of the symmetric G above cut * wmax."""
    w, V = xp.linalg.eigh(G); top = float(w[-1]) if wmax is None else wmax; keep = w > cut*max(top, 1e-300)
    return w[keep], V[:, keep]


def combine(eng, sources, coeffs, cut=1e-13):
    """sum_a coeffs[a] |sources[a]> as (sector sizes of every bond, tensors in the M layout, norm of the sum); the tensors
    are right-orthonormal for layers 1 ... n-1 and the centre tensor (layer 0) is normalised."""
    xp = _xp(eng); md = eng.model; n = eng.n; assert n >= 2
    S = [s if isinstance(s, Stored) else Stored(eng, s, 0.0, xp) for s in sources]
    bond = []
    for r in range(n+1):
        b = copy.copy(eng.bond[r]); b.dims = (S[0].bond[r].dims.copy() if r in (0, n) else sum(s.bond[r].dims for s in S)).astype(eng.bond[r].dims.dtype); bond.append(b)
    off = [[np.zeros_like(bond[r].dims) if r in (0, n) else sum((s.bond[r].dims for s in S[:a]), np.zeros_like(bond[r].dims)) for r in range(n+1)] for a in range(len(S))]
    plan = [ed.Plan(md, r, bond[r], bond[r+1]) for r in range(n)]; T = []
    for r in range(n):                                                 # direct sum, slab by slab
        P = plan[r]; t = xp.zeros(P.size); pos = {(i, s): c0 for i, s, j, c0, w in _slabs(P)}
        for a, s_ in enumerate(S):
            pa = s_.plan[r]; Ma = s_.M[r]; ca = float(coeffs[a]) if r == 0 else 1.0
            for i, s, j, c0, w in _slabs(pa):
                Di, Ci = int(pa.DL[i]), int(pa.C[i]); blk = Ma[int(pa.Moff[i]):int(pa.Moff[i+1])].reshape(Di, Ci)[:, c0:c0+w]
                tgt = t[int(P.Moff[i]):int(P.Moff[i+1])].reshape(int(P.DL[i]), int(P.C[i])); r0 = int(off[a][r][i]); q0 = pos[(i, s)] + int(off[a][r+1][j])
                tgt[r0:r0+Di, q0:q0+w] += ca*blk
        T.append(t)
    for r in range(n-1, 0, -1):                                        # right-orthonormal tensors from the right end; the factor goes left
        P = plan[r]; t = T[r]; Ls = {}; Qs = {}; new = np.zeros_like(bond[r].dims); G = {}
        for i in np.flatnonzero((P.DL > 0) & (P.C > 0)):
            A = t[int(P.Moff[i]):int(P.Moff[i+1])].reshape(int(P.DL[i]), int(P.C[i])); G[int(i)] = (A, A @ A.T)
        wmax = max((float(xp.linalg.eigvalsh(g)[-1]) for A, g in G.values()), default=0.0)
        for i, (A, g) in G.items():
            w, V = _isqrt_basis(g, xp, cut, wmax)
            if not w.size:
                continue
            Q1 = (V/xp.sqrt(w)).T @ A; L1 = V*xp.sqrt(w)               # A ~ L1 Q1; a second pass makes the rows of Q orthonormal to rounding
            w2, V2 = _isqrt_basis(Q1 @ Q1.T, xp, 0.0); Qs[i] = (V2/xp.sqrt(w2)).T @ Q1; Ls[i] = L1 @ (V2*xp.sqrt(w2)); new[i] = Qs[i].shape[0]
        pp = plan[r-1]; told = T[r-1]; bond[r].dims = new; Pn = ed.Plan(md, r, bond[r], bond[r+1]); ppn = ed.Plan(md, r-1, bond[r-1], bond[r])
        tn = xp.zeros(Pn.size)
        for i, Q in Qs.items():
            tn[int(Pn.Moff[i]):int(Pn.Moff[i+1])] = Q.ravel()
        tp = xp.zeros(ppn.size); posn = {(i, s): c0 for i, s, j, c0, w in _slabs(ppn)}
        for i, s, j, c0, w in _slabs(pp):
            if j not in Ls:
                continue
            Di, Ci = int(pp.DL[i]), int(pp.C[i]); k = Ls[j].shape[1]
            tp[int(ppn.Moff[i]):int(ppn.Moff[i+1])].reshape(Di, int(ppn.C[i]))[:, posn[(i, s)]:posn[(i, s)]+k] = told[int(pp.Moff[i]):int(pp.Moff[i+1])].reshape(Di, Ci)[:, c0:c0+w] @ Ls[j]
        T[r] = tn; T[r-1] = tp; plan[r] = Pn; plan[r-1] = ppn
    norm = float(xp.linalg.norm(T[0])); T[0] = T[0]/norm
    return [b.dims.copy() for b in bond], T, norm


def ritz_state(eng, sources, coefficients, cut=1e-13):
    """Install sum_a c_a |a> (normalised) in eng, environments rebuilt; returns its energy."""
    dims, T, norm = combine(eng, sources, coefficients, cut); st.install(eng, dims, T, 0)
    return st.energy(eng, 0)[0]


def subspace(model, sources, cls=ed.DMRG, cut=1e-13, drop=1e-8, log=None, **kw):
    """Overlap and Hamiltonian matrices of the saved states and the Ritz levels in their span.
    Returns dict(E (diagonal energies), S, H, checks (largest deviation of the two sum rules), levels, vectors (columns:
    coefficients of the sources, normalised so that the combination has norm 1), kept (dimension after removing
    directions of S below `drop`))."""
    m = len(sources); h0 = st.read_head(sources[0]) if isinstance(sources[0], str) else None
    eng = cls(model, h0['nup'], h0['K'], **kw) if h0 else None; assert eng is not None, 'sources must be files'
    xp = _xp(eng); stored = [Stored(eng, s, 0.0, xp) for s in sources]; E = np.zeros(m); S = np.eye(m); H = np.zeros((m, m)); chk = 0.0
    for a in range(m):
        E[a] = ritz_state(eng, [stored[a]], [1.0], cut); H[a, a] = E[a]
        if log:
            log(f'state {a}: E = {E[a]:.10f}')
    for a in range(m):
        for b in range(a+1, m):
            N = {}; EN = {}
            for sg in (+1.0, -1.0):
                dims, T, norm = combine(eng, [stored[a], stored[b]], [1.0, sg], cut); st.install(eng, dims, T, 0); N[sg] = norm*norm; EN[sg] = st.energy(eng, 0)[0]
            S[a, b] = S[b, a] = (N[1.0]-N[-1.0])/4; H[a, b] = H[b, a] = (N[1.0]*EN[1.0]-N[-1.0]*EN[-1.0])/4
            chk = max(chk, abs((N[1.0]+N[-1.0])/2-2), abs((N[1.0]*EN[1.0]+N[-1.0]*EN[-1.0])/2-(E[a]+E[b])))
            if log:
                log(f'pair {a}, {b}: <a|b> = {S[a, b]:+.8f}, <a|H|b> = {H[a, b]:+.10f}, <a|H|b> - <a|b> (E_a + E_b)/2 = {H[a, b]-S[a, b]*(E[a]+E[b])/2:+.3e}')
    w, V = np.linalg.eigh(S); keep = w > drop; X = V[:, keep]/np.sqrt(w[keep]); lev, C = np.linalg.eigh(X.T @ H @ X); vec = X @ C
    return dict(E=E.tolist(), S=S.tolist(), H=H.tolist(), checks=chk, levels=lev.tolist(), vectors=vec.tolist(), kept=int(keep.sum()), S_eigenvalues=w.tolist())


def mixed_layer_sz(eng, source):
    """<a|S_z(layer l)|b> for every layer l, with b the state held by eng (tensors only are needed: st.install(...,
    environments=False) or load_state(..., environments=False)) and a = source (file, engine or Stored).  Both states
    normalised.  Uses that the label m of bond r is the number of up spins in the layers left of it: with the overlap of
    the two states split at bond r into a left and a right part per label i,
        <a|N_left(r)|b> = sum_i m_i <OL[r]_i, OR[r]_i>,     <a|S_z(l)|b> = <a|N_left(l+1)|b> - <a|N_left(l)|b> - (n_l / 2) <a|b>.
    For a = b this is st.layer_sz.  Returns (list of the n layer values, <a|b>)."""
    xp = _xp(eng); n = eng.n; s = source if isinstance(source, Stored) else Stored(eng, source, 0.0, xp)
    for r in range(n):
        s.grow_left(eng, r, eng.M[r])
    for r in range(n-1, 0, -1):
        s.grow_right(eng, r, eng.M[r])
    s.OR[0] = None
    nl = []
    for r in range(n+1):
        if r in (0, n):
            tot = None
        L = s.OL[r] if r > 0 else None; R = s.OR[r] if r < n else None
        if r == 0:                                                     # N_left(0) = 0
            nl.append(0.0); continue
        if r == n:
            ov = float(sum(float(xp.sum(L[i])) for i in range(len(L)) if L[i].size)); nl.append(float(sum(float(eng.bond[n].m[i])*float(xp.sum(L[i])) for i in range(len(L)) if L[i].size))); continue
        nl.append(float(sum(float(eng.bond[r].m[i])*float(xp.sum(L[i]*R[i])) for i in range(len(L)) if L[i].size)))
    ov = float(sum(float(xp.sum(s.OL[n][i])) for i in range(len(s.OL[n])) if s.OL[n][i].size))
    sizes = [len(y) for y in eng.model.lat.layers]
    return [nl[l+1] - nl[l] - sizes[l]/2.0*ov for l in range(n)], ov


def ritz_profiles(model, sources, vectors, cls=ed.DMRG, log=None):
    """<S_z(layer)> of the combinations sum_a vectors[a, k] |a> (columns k), from the matrix elements of mixed_layer_sz:
    no state with the summed bond dimension is formed.  Returns (profiles: k x n, Z: m x m x n matrix elements)."""
    m = len(sources); h0 = st.read_head(sources[0]); eng = cls(model, h0['nup'], h0['K']); xp = _xp(eng); Z = None
    stored = [Stored(eng, f, 0.0, xp) for f in sources]
    for b in range(m):
        f = np.load(sources[b], allow_pickle=False); st.install(eng, [f[f'd{r}'] for r in range(eng.n+1)], [f[f'M{r}'] for r in range(eng.n)], 0, environments=False)
        for a in range(b+1):
            prof, ov = mixed_layer_sz(eng, stored[a])
            if Z is None:
                Z = np.zeros((m, m, len(prof)))
            Z[a, b] = Z[b, a] = prof
            if log:
                log(f'<{a}|S_z|{b}>: overlap {ov:+.6f}, sum over layers {sum(prof):+.6f}')
    V = np.asarray(vectors)
    return np.einsum('ak,bk,abl->kl', V, V, Z), Z
