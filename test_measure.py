#!/usr/bin/env python3
"""Structure factor in symmetry sectors (edc_sub.SubEngine.operator + edc_hubbard_sym.ground_state_vector +
edc_measure) against an independent reference: the sector basis built explicitly in the full S_z space as normalised
projections P|r> = (1/|G|) sum_g chi(g)^* U(g)|r>, H and the structure-factor operator as sparse matrices in the full
space, ground state by dense diagonalisation of V^+ H V, S(Q) evaluated on the full-space vector.
Every sector of each cluster (complete scan, edc_hubbard_sym.sectors), both the dense path and Lanczos + replay.
Usage: test_measure.py [cpu|gpu]"""
import sys, math, itertools
import numpy as np, scipy.sparse as sp
import edc, edc_lattice as el, edc_lattices as L, edc_hubbard_sym as hs, edc_sub, edc_measure as em

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'


def basis(N, nup):
    s = np.array([sum(1 << b for b in c) for c in itertools.combinations(range(N), nup)], np.int64)
    s.sort()
    return s


def full_operator(model, states):
    assert not np.any(model.h)
    idx = {int(v): i for i, v in enumerate(states)}
    rows, cols, vals = [], [], []
    diag = np.zeros(len(states))
    for (i, j), (jxy, jz) in model.bonds.items():
        bi, bj = (states >> i) & 1, (states >> j) & 1
        diag += jz*np.where(bi == bj, 0.25, -0.25)
        if jxy:
            for a in np.flatnonzero(bi != bj):
                t = int(states[a]) ^ ((1 << i) | (1 << j))
                rows.append(idx[t]); cols.append(a); vals.append(0.5*jxy)
    D = len(states)
    return sp.csr_matrix((vals, (rows, cols)), shape=(D, D))+sp.diags(diag)


def sector_basis(states, N, perms, chars, flip):
    """Columns P|r>/|P|r>| for one r per orbit (zero projections dropped)."""
    idx = {int(v): i for i, v in enumerate(states)}
    full = (1 << N)-1
    imgs = []
    for p in perms:
        im = np.zeros_like(states)
        for b in range(N):
            im |= ((states >> b) & 1) << p[b]
        imgs.append(im)
    els = [(im, c) for im, c in zip(imgs, chars)]
    if flip:
        els += [(full ^ im, c*flip) for im, c in zip(imgs, chars)]
    omin = np.min(np.stack([im for im, _ in els]), axis=0)
    cols = []
    for a in np.flatnonzero(omin == states):
        v = {}
        for im, c in els:
            t = idx[int(im[a])]
            v[t] = v.get(t, 0)+np.conj(c)
        nrm = math.sqrt(sum(abs(x)**2 for x in v.values()))
        if nrm > 1e-9:
            cols.append({t: x/nrm for t, x in v.items()})
    D = len(states)
    r = [t for c in cols for t in c]; cc = [n for n, c in enumerate(cols) for t in c]; vv = [x for c in cols for x in c.values()]
    return sp.csc_matrix((vv, (r, cc)), shape=(D, len(cols)), dtype=complex)


def group_of(cl, k, ir, pops):
    if pops is None:
        return cl.group(k, ir)[:2]
    if hasattr(cl, 'irrep_table'):
        return cl.group(k, ir, allowed=pops)[:2]
    return cl.group(k, ir, [o for o in cl.little_group(k) if o in el.IRREPS[ir] and o in pops])[:2]


def momenta_Q(cl, count):
    """A few compatible Cartesian Q in the extended zone: Q = 2 pi (C M)^{-T} n."""
    C = cl.cart if hasattr(cl, 'cart') else np.eye(2)
    G = 2*math.pi*np.linalg.inv(C @ cl.M.astype(float)).T
    out = []
    for n in sorted(itertools.product(range(-3, 4), repeat=2), key=lambda n: (abs(n[0])+abs(n[1]), n)):
        Q = G @ np.array(n, float)
        if np.linalg.norm(Q) > 1e-9 and not any(any(np.allclose(q, R) for R in em.star(cl, Q2)) for Q2 in out for q in [Q]):
            out.append(Q)
        if len(out) == count:
            return out
    return out


def check(name, cl, model, nQ=3):
    N = cl.N; nup = N//2
    states = basis(N, nup)
    H = full_operator(model, states)
    Qs = momenta_Q(cl, nQ)
    Os = [full_operator(em.structure_factor_model(cl, Q)[0], states) for Q in Qs]
    worst = {'default': 0.0, 'lanczos': 0.0}; maxres = 0.0; nsec = ncmp = 0
    for k, ir, fl, pops, mult in hs.sectors(cl, nup, N-nup):
        perms, chars = group_of(cl, k, ir, pops)
        V = sector_basis(states, N, perms, chars, fl)
        if V.shape[1] == 0:
            continue
        e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=DEVICE, point_ops=pops)
        assert e.dim == V.shape[1], (name, k, ir, fl, e.dim, V.shape[1])
        Hs = (V.conj().T @ H @ V).toarray()
        w, U = np.linalg.eigh((Hs+Hs.conj().T)/2)
        if len(w) > 1 and w[1]-w[0] < 1e-8:          # degenerate inside the sector: S(Q) of "the" ground state undefined
            nsec += 1; continue
        psi_ref = V @ U[:, 0]
        ref = [float(np.vdot(psi_ref, O @ psi_ref).real) for O in Os]
        ref = [(0.75*N+2*v)/N for v in ref]
        runs = [('default', lambda: e.ground_energy(tol=1e-13))]
        if e.dim > 30:
            runs.append(('lanczos', lambda: hs.lanczos(e, tol=1e-13, small=0)))
        for tag, run in runs:
            E, it, _ = run()
            psi, Ev, res = hs.ground_state_vector(e)
            assert abs(Ev-w[0]) < 1e-9 and res < 1e-5, (name, tag, k, ir, fl, Ev, w[0], res)
            dense = e.dim <= 400 and tag == 'default'
            maxres = maxres if dense else max(maxres, res)
            for Q, r in zip(Qs, ref):
                S = em.structure_factor(e, psi, cl, Q)['S']
                worst[tag] = max(worst[tag], abs(S-r))
                # dense: exact; Lanczos: a Ritz vector, error ~ residual / gap
                assert abs(S-r) < (1e-10 if dense else max(1e-7, 10*res)), (name, tag, k, ir, fl, Q, S, r, res)
            ncmp += 1
        nsec += 1
    print(f'{name}: N={N}, {nsec} sectors, {ncmp} runs x {len(Qs)} Q; max |dS| default {worst["default"]:.1e}, '
          f'Lanczos {worst["lanczos"]:.1e} (max residual {maxres:.1e})', flush=True)


if __name__ == '__main__':
    sq = el.Cluster((4, 0), (0, 4))
    m = edc.SpinHalfModel(16)
    for s, (x, y) in enumerate(sq.coord):
        for (dx, dy), J in (((1, 0), 1.0), ((0, 1), 1.0), ((1, 1), 0.5), ((1, -1), 0.5)):
            m.heisenberg(s, sq.site(x+dx, y+dy), J)
    check('square 4x4 J1-J2 (J2 = 0.5)', sq, m)
    kg = L.LatticeCluster('kagome', (2, 0), (0, 2)); check('kagome 12', kg, kg.heisenberg(1.0))
    tr = L.LatticeCluster('triangular', (2, 2), (-2, 4)); check('triangular 12', tr, tr.heisenberg(1.0, 0.1))
    hc = L.LatticeCluster('honeycomb', (3, 0), (0, 3)); check('honeycomb 18', hc, hc.heisenberg(1.0))
    print('all passed')
