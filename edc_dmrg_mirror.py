#!/usr/bin/env python3
"""The end exchange of a cylinder applied to a saved state of the layer DMRG: the exact mirror image of a
state, without a new DMRG run.

    s, perm = end_exchange(lat)                 layer l -> n-1-l, spin at y -> spin at y + s; perm[site] = image site
    eng = mirror(src, model)                    an engine holding the image of the saved state src (file or engine)
    mirror_file(src, model, path)               the same, written as a state file

The map.  Exchanging the two ends of a kagome YC cylinder with chain rings at both ends maps every bond to a bond when
combined with a shift around the cylinder: y -> y + 1/2 for an odd number of rings, y -> y for an even number (found by end_exchange, which tries every shift in quarter
cells).  A shift commutes with
the translation around the cylinder, so the map G keeps the momentum K and the number of up spins: G maps the sector
of a state to itself.

On the matrix-product state.  The layer tensors are taken in reversed order and transposed (the left bond becomes the
right bond); a bond label (m, k) of everything to the left becomes (nup - m, K - k) of everything to the right; the
layer state s of layer l becomes sum_s' W_l[s', s] |s'> on layer n-1-l, with W_l = Ur_{n-1-l}^+ Pi_l Ur_l (Pi_l moves
the spins of layer l onto their images).  W_l keeps the labels (m, k), and in the real layer bases it is
exp(-i pi t k / c) times a real orthogonal matrix, t = 2 s (k = the momentum label of the layer state, 0 <= k < c),
because the bases are invariant under the reflection y -> 1/2 - y times complex conjugation and G turns that
reflection into the reflection times t translations.  The phases of all layers multiply to
exp(-i pi t (K + c n_carry) / c), where n_carry counts the bonds at which the momentum label wraps around
(k_left + k_layer >= c); a sign (-1)^t per wrap on the slab gives a real state, and
    G |psi>  =  exp(-i pi t K / c) |mirror(psi)>.
The tensors stay in their canonical form (the centre moves from c to n-1-c); nothing is truncated.  For K = 0 (or
t = 0) the image is exact with no phase; otherwise the global phase does not matter for overlaps |<a|G b>|, energies
or subspaces.  On an eigenstate of one level, <psi|mirror(psi)> = +-1 is its mirror parity in this convention.

Checked in test_dmrg_mirror.py against the permutation of the whole vector on small cylinders."""
from fractions import Fraction
import numpy as np
import edc_dmrg as ed
import edc_dmrg_state as st


def end_exchange(lat):
    """(s, perm): the first shift s (quarter cells) for which layer l -> n-1-l, y -> y + s maps every bond to a bond."""
    n = len(lat.layers); off = lat.offset; pos = {}
    for l, ys in enumerate(lat.layers):
        for i, y in enumerate(ys):
            pos[(l, y)] = int(off[l]+i)
    bonds = {(min(a, b), max(a, b), round(float(J), 12)) for a, b, J in lat.global_bonds()}
    for q in range(4*lat.c):
        s = Fraction(q, 4)
        if not all((n-1-l, (y+s) % lat.c) in pos for l, ys in enumerate(lat.layers) for y in ys):
            continue
        perm = [pos[(n-1-l, (y+s) % lat.c)] for l, ys in enumerate(lat.layers) for y in ys]
        if {(min(perm[a], perm[b]), max(perm[a], perm[b]), round(float(J), 12)) for a, b, J in lat.global_bonds()} == bonds:
            return s, perm
    raise ValueError(f'{lat.name}: no end exchange with a shift around the cylinder maps the bonds onto themselves')


def layer_maps(model, perm, t):
    """For every layer l the real matrix R_l[s', s] (layer n-1-l basis x layer l basis) with W_l = exp(-i pi t k / c) R_l."""
    lat = model.lat; n = model.n; c = model.c; out = []
    assert model.mod == c, 'the mirror map needs the momentum labels (labels=sz,k)'
    for l in range(n):
        l2 = n-1-l; ks, kd = model.kinds[model.site_kind[l]], model.kinds[model.site_kind[l2]]
        loc = [perm[int(lat.offset[l])+a] - int(lat.offset[l2]) for a in range(ks.n)]
        assert sorted(loc) == list(range(kd.n)), 'the image of a layer is not one layer'
        codes = np.arange(ks.S, dtype=np.int64); new = np.zeros(ks.S, dtype=np.int64)
        for a in range(ks.n):
            new |= ((codes >> a) & 1) << loc[a]
        X = np.zeros_like(ks.Ur); X[new] = ks.Ur; W = kd.Ur.conj().T @ X
        big = np.abs(W) > 1e-10
        assert np.all((kd.sm[:, None] == ks.sm[None, :])[big]) and np.all((kd.sk[:, None] == ks.sk[None, :])[big]), 'the map does not keep the labels'
        Wr = W*np.exp(1j*np.pi*t*kd.sk/c)[:, None]
        assert np.abs(Wr.imag).max() < 1e-10, 'layer map is not exp(-i pi t k / c) times a real matrix'
        R = Wr.real.copy(); assert np.abs(R.T @ R - np.eye(ks.S)).max() < 1e-10, 'layer map is not orthogonal'
        out.append(R)
    return out


def _source(src, model, cls):
    if isinstance(src, str):
        return st.load_state(src, model, cls=cls, environments=False)
    return src


def mirror(src, model, cls=ed.DMRG, environments=True, **kw):
    """An engine (class cls) holding mirror(src): G |src> = exp(-i pi t K / c) |mirror(src)>, t = 2 x shift.  src: a state file or an engine."""
    old = _source(src, model, ed.DMRG); n = old.n; c = model.c; mod = model.mod
    can = st.canonical(old); co = can['centre']; assert co is not None, 'the source is not in a canonical form'
    nup, K = int(old.bond[n].m[0]), int(old.bond[n].k[0])
    s, perm = end_exchange(model.lat); assert (2*s).denominator == 1, 'only shifts by whole or half cells'; tw = int(2*s); R = layer_maps(model, perm, tw)
    eng = cls(model, nup, K, **kw)
    # bonds: new bond n - r holds the labels (nup - m, K - k) of old bond r
    idx = []
    for r in range(n+1):
        bo, bn = old.bond[n-r], eng.bond[r]
        j = bn.lookup(nup - bo.m, (K - bo.k) % mod); assert np.all(j >= 0) and len(set(j.tolist())) == bn.n == bo.n, f'labels of bond {r} do not match'
        dims = np.zeros(bn.n, dtype=np.int32); dims[j] = bo.dims; bn.dims[:] = dims; idx.append(j)  # idx[r][old label of bond n-r] = new label
    tensors = [None]*n
    for l in range(n):                                                 # old layer l -> new layer l2
        l2 = n-1-l; po = old.plan[l]; pn = ed.Plan(model, l2, eng.bond[l2], eng.bond[l2+1])
        ks, kd = model.kinds[model.site_kind[l]], model.kinds[model.site_kind[l2]]
        Mo = np.asarray(st._host(old.M[l]), dtype=np.float64); Mn = np.zeros(pn.size)
        inew_of = idx[n-l-1]                                           # new left bond l2 = old bond l+1 (old right bond)
        jnew_of = idx[n-l]                                             # new right bond l2+1 = old bond l (old left bond)
        kL = eng.bond[l2].k
        for i in np.nonzero(po.DL)[0]:
            Di = int(po.DL[i]); Ai = Mo[po.Moff[i]:po.Moff[i+1]].reshape(Di, int(po.C[i]))
            for q, src_idx in ks.blocks.items():
                if po.cst[i, src_idx[0]] < 0:
                    continue
                j = int(po.mr[i, src_idx[0]]); Dj = int(po.DR[j])
                if Dj == 0:
                    continue
                stack = np.stack([Ai[:, po.cst[i, u]:po.cst[i, u]+Dj] for u in src_idx])          # (states, Di, Dj)
                dst_idx = kd.blocks[q]; new = np.einsum('ps,sab->pba', R[l][np.ix_(dst_idx, src_idx)], stack)
                i2, j2 = int(inew_of[j]), int(jnew_of[i]); sign = -1.0 if tw % 2 and int(kL[i2]) + q[1] >= c else 1.0
                assert int(pn.DL[i2]) == Dj and pn.Moff[i2+1]-pn.Moff[i2] == Dj*int(pn.C[i2])
                B = Mn[pn.Moff[i2]:pn.Moff[i2+1]].reshape(Dj, int(pn.C[i2]))
                for u, sp in enumerate(dst_idx):
                    assert int(pn.mr[i2, sp]) == j2 and int(pn.cw[i2, sp]) == Di
                    B[:, pn.cst[i2, sp]:pn.cst[i2, sp]+Di] = sign*new[u]
        tensors[l2] = Mn
    st.install(eng, [b.dims.copy() for b in eng.bond], tensors, n-1-co, environments)
    eng.mirror_of = dict(shift=str(s), centre=n-1-co, phase_over_pi=-tw*K/c)
    return eng


def mirror_file(src, model, path, **meta):
    """Write mirror(src) to path; the head records the source."""
    eng = mirror(src, model, environments=True)
    return st.save_state(eng, path, mirror_of=src if isinstance(src, str) else 'engine', shift=eng.mirror_of['shift'], **meta)
