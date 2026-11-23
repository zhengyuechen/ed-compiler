#!/usr/bin/env python3
"""Checks of bench/corr_tools.py and bench/corr_full.py on the 12-site kagome torus against explicit spin operators."""
import os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench'))
import numpy as np
import edc, edc_lattices as L
import corr_tools as ct, corr_full as cf

cl = L.LatticeCluster('kagome', (2, 0), (0, 2)); N = cl.N
geo = ct.geometry(cl)
assert len(geo['bonds']) == 2*N and len(geo['tris']) == 2*N//3
for t, P in zip(geo['tris'], geo['tri_xy']):          # anticlockwise, unit triangles of side 1/2
    a, b, c = P
    assert (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]) > 0
    assert all(abs(np.linalg.norm(P[u]-P[(u+1) % 3])-0.5) < 1e-12 for u in range(3))

# ---- spin operators on 2^N states as sparse matrices, site a = bit a
import scipy.sparse as sps
sz, sp = sps.csr_matrix(np.array([[-0.5, 0], [0, 0.5]])), sps.csr_matrix(np.array([[0, 0], [1.0, 0]]))
sx, sy = (sp+sp.T)/2, (sp-sp.T)/2j
def site_op(o, a):
    out = sps.identity(1, dtype=complex, format='csr')
    for s in range(N-1, -1, -1):
        out = sps.kron(out, o if s == a else sps.identity(2), format='csr')
    return out
S = [[site_op(o, a) for o in (sx, sy, sz)] for a in range(N)]
def dot(i, j):
    return sum(S[i][c] @ S[j][c] for c in range(3))
def chir(i, j, k):
    eps = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
    return sum(s*S[i][a] @ S[j][b] @ S[k][c] for (a, b, c), s in eps.items())

for nup, J2 in ((7, 0.0), (6, 0.3)):
    eng = edc.Engine(cl.heisenberg(1.0, J2), nup, threads=2)
    basis = cf.Basis(eng)
    psi = np.random.default_rng(nup).standard_normal(eng.dim); psi /= np.linalg.norm(psi)
    raw = cf.measure(cl, basis, psi, geo, chunk=100)
    full = np.zeros(2**N, complex); full[basis.st.astype(np.int64)] = psi
    ex = lambda O: (full.conj() @ (O @ full))
    for i, j in itertools.product(range(N), repeat=2):
        assert abs(raw['Gz'][i, j]-ex(S[i][2] @ S[j][2]).real) < 1e-12
        assert abs(raw['SS'][i, j]-ex(dot(i, j)).real) < 1e-12
    B = [dot(i, j) for i, j in geo['bonds']]
    Z = [S[i][2] @ S[j][2] for i, j in geo['bonds']]
    X = [chir(*t) for t in geo['tris']]
    vB, vZ, vX = [O @ full for O in B], [O @ full for O in Z], [O @ full for O in X]
    for a in range(len(B)):
        assert abs(raw['eb'][a]-ex(B[a]).real) < 1e-12 and abs(raw['ezz'][a]-ex(Z[a]).real) < 1e-12
        for b in range(len(B)):
            assert abs(raw['Gb'][a, b]-(vB[a].conj() @ vB[b]).real) < 1e-12
            assert abs(raw['Gzz'][a, b]-(vZ[a].conj() @ vZ[b]).real) < 1e-12
        assert abs(raw['Gb'][a, a]-(3/16-raw['eb'][a]/2)) < 1e-12              # (S.S)^2 = 3/16 - S.S/2
    for a, t in enumerate(geo['tris']):
        for b in range(len(X)):
            assert abs(raw['Gx'][a, b]-(vX[a].conj() @ vX[b]).real) < 1e-12
        tb = sum(raw['SS'][t[u], t[(u+1) % 3]] for u in range(3))
        assert abs(raw['Gx'][a, a]-(3/32-tb/8)) < 1e-12                        # chi^2 = 3/16 on total spin 1/2, 0 on 3/2

    # ---- projectors: idempotent, orthogonal, complete; kinds of the exact levels
    kinds = [f"{s['name']}:{ir}" for s in ct.stars(cl) for ir in s['table']]
    v = np.random.default_rng(5).standard_normal(eng.dim)
    parts = {}
    for kd in kinds:
        P = cf.projector(cl, basis, kd)
        parts[kd] = P(v)
        assert np.linalg.norm(P(parts[kd])-parts[kd]) < 1e-12
    assert np.linalg.norm(sum(parts.values())-v) < 1e-12
    for a, b in itertools.combinations(kinds, 2):
        assert abs(parts[a] @ parts[b]) < 1e-12
    H = np.zeros((eng.dim, eng.dim))
    for sh, Jv in ((1, 1.0), (2, J2)):
        if Jv:
            for i, j in cl.bonds(sh):
                idx = basis.swap_index(i, j)
                H[idx, np.arange(eng.dim)] += 0.5*Jv
                H[np.arange(eng.dim), np.arange(eng.dim)] -= 0.25*Jv
    x = np.random.default_rng(1).standard_normal(eng.dim)
    assert np.linalg.norm(H @ x-eng.matvec(x)) < 1e-11
    w, U = np.linalg.eigh(H)
    for kd in kinds:
        P = cf.projector(cl, basis, kd)
        wt = np.array([np.linalg.norm(P(U[:, n]))**2 for n in range(eng.dim)])
        # weight of a degenerate level in a kind, summed over the level, is an integer
        lev = np.flatnonzero(wt > 1e-8)
        if len(lev) == 0:
            continue
        e_kind = w[lev[0]]
        psi_k, E, res, steps = cf.lowest(eng, P, seed=3, rtol=1e-11)
        assert abs(E-e_kind) < 1e-9 and res < 1e-8, (kd, E, e_kind, res)
        assert np.linalg.norm(P(psi_k)-psi_k) < 1e-10

    # ---- level average = space-group average of one member; modes() reproduces the spectrum of C
    out, meta = cf.run('2,0,0,2', nup, J2, 'G:E2', threads=2, log=lambda s: None)
    lev = np.flatnonzero(np.abs(w-meta['E']) < 1e-8)
    P = cf.projector(cl, basis, 'G:E2')
    members = [u for u in (U[:, n] for n in lev)]
    Q = np.array([P(u) for u in members]).T
    q, _ = np.linalg.qr(Q[:, np.linalg.norm(Q, axis=0) > 1e-6])
    q = q[:, :2]
    Gb = np.mean([cf.measure(cl, basis, q[:, c], geo)['Gb'] for c in range(q.shape[1])], axis=0)
    sp_b = ct.OpSpace(cl, 'b', geo)
    assert np.max(np.abs(sp_b.average(Gb)-out['Gb'])) < 1e-9, 'level average'
    for kind, Gk, ek in (('z', 'Gz', 'mz'), ('zz', 'Gzz', 'ezz'), ('b', 'Gb', 'eb'), ('x', 'Gx', None)):
        sp = ct.OpSpace(cl, kind, geo)
        C = out[Gk]-(np.outer(out[ek], out[ek]) if ek else 0)
        sm = ct.summary(sp, C)
        ev = sorted(np.linalg.eigvalsh(C))
        mv = sorted(x for m in sm['modes'] for x in m['values'] for _ in range(m['dim']))
        assert np.allclose(ev, mv, atol=1e-9), kind
        assert min(ev) > -1e-9
    print(f'N_up = {nup}, J2 = {J2}: ok')

# ---- the uniform patterns: total S^z does not fluctuate, the bond sum is the energy
sp = ct.OpSpace(cl, 'z', geo)
C = out['Gz']-np.outer(out['mz'], out['mz'])
g = next(m for m in ct.modes(sp, C) if m['star'] == 'G' and m['irrep'] == 'A1')
assert abs(g['values'][0]) < 1e-10

# ---- bench/corr_sym.py (symmetric engine, translations only) against the full basis, where the lowest state of the
#      zero-momentum sector is a single fully symmetric level
import corr_sym as cs
done = 0
for nup, J2 in ((7, 0.0), (6, 0.3), (8, 0.0), (5, 0.0), (6, -0.2)):
    eng = edc.Engine(cl.heisenberg(1.0, J2), nup, threads=2); basis = cf.Basis(eng)
    lows = {}
    for kd in [f"G:{ir}" for ir in ct.TABLES[12][1]]:
        try:
            lows[kd] = cf.lowest(eng, cf.projector(cl, basis, kd), seed=3, rtol=1e-11)[1]
        except Exception:
            pass
    order = sorted(lows, key=lows.get)
    if order[0] != 'G:A1' or lows[order[1]]-lows[order[0]] < 1e-3:
        continue
    full, mf = cf.run('2,0,0,2', nup, J2, 'G:A1', threads=2, log=lambda s: None)
    sym, ms = cs.run('2,0,0,2', nup, J2, device='cpu', log=lambda s: None)
    assert abs(mf['E']-ms['E']) < 1e-9
    for k in ('mz', 'Gz', 'ezz', 'Gzz'):
        assert np.max(np.abs(full[k]-sym[k])) < 1e-9, k
    typ, kind = sym['bond_type'], sym['tri_kind']
    Nc = cl.Nc
    for u in range(6):
        assert abs(full['eb'][typ == u].mean()-sym['eb0'][u]) < 1e-9
        for v in range(6):
            assert abs(full['Gb'][np.ix_(typ == u, typ == v)].sum()/Nc-sym['Gb0'][u, v]) < 1e-9
    for k in range(2):
        for l in range(2):
            assert abs(full['Gx'][np.ix_(kind == k, kind == l)].sum()/Nc-sym['Gx0'][k, l]) < 1e-9
    spb = ct.OpSpace(cl, 'b', geo)
    Cb = full['Gb']-np.outer(full['eb'], full['eb'])
    c0 = sym['Gb0']-Nc*np.outer(sym['eb0'], sym['eb0'])
    a = {(m['star'], m['irrep']): m['values'] for m in ct.modes(spb, Cb) if m['star'] == 'G'}
    b = {(m['star'], m['irrep']): m['values'] for m in ct.modes(spb, cs.embed(spb, c0, typ)) if m['star'] == 'G'}
    assert a.keys() == b.keys() and all(np.allclose(a[k], b[k], atol=1e-9) for k in a)
    if J2 == 0:
        assert abs(a[('G', 'A1')][-1]) < 1e-9      # the sum of all bonds is the Hamiltonian: it does not fluctuate
    print(f'corr_sym against corr_full: N_up = {nup}, J2 = {J2}: ok ({ms["dim"]} states)')
    done += 1
assert done >= 1
print('test_corr: all checks passed')
