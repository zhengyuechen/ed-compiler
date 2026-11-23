#!/usr/bin/env python3
"""edc_glayout.LayoutEngine against edc_sub.SubEngine, block by block.
For every S_z sector and every block of edc_hubbard_sym.sectors: the dimension; the whole spectrum (blocks up to 200
states) or the lowest level through the shared Lanczos (larger blocks).  Lattices: kagome (three sublattices),
triangular (three classes of cells), square (2 x 2 cell, four parts).  Models: J1-J2, XXZ, a third shell (bonds inside
one part).  Also: several levels, retune, operator (structure factor), batched.
Usage: test_glayout.py [gpu] [quick]"""
import sys, time
import numpy as np
import edc, edc_lattices as L, edc_hubbard_sym as hs, edc_sub, edc_glayout, edc_measure as em

device = 'gpu' if 'gpu' in sys.argv else 'cpu'; quick = 'quick' in sys.argv


def xxz(cl, shells, delta=1.0):
    m = edc.SpinHalfModel(cl.N)
    for sh, Jv in enumerate(shells, start=1):
        if Jv:
            for a, b in cl.bonds(sh):
                m.exchange(a, b, Jv); m.zz(a, b, Jv*delta)
    return m


def dense(e):
    xp = e.xp; dt = np.float64 if e.real else np.complex128; Hm = np.zeros((e.dim, e.dim), dt); x = xp.zeros(e.dim, dt); y = xp.zeros(e.dim, dt)
    for j in range(e.dim):
        x[:] = 0; x[j] = 1; e.matvec(x, y); Hm[:, j] = y.get() if xp is not np else y
    return Hm


def compare(kind, A1, A2, shells, delta=1.0, nups=None, maxdim=40000):
    cl = L.LatticeCluster(kind, A1, A2); N = cl.N; model = xxz(cl, shells, delta); worst = 0.0; nb = nd = 0; p = None; t0 = time.perf_counter()
    for nup in (nups if nups is not None else range((N+1)//2, N+1)):
        tot = 0
        for k, ir, fl, pops, mult in hs.sectors(cl, nup, N-nup):
            s = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops, stored=0)
            if s.dim == 0:
                try:
                    g = edc_glayout.LayoutEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops); assert g.dim == 0
                except ValueError:
                    pass
                continue
            g = edc_glayout.LayoutEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops); p = g.p
            assert g.dim == s.dim, (kind, nup, k, ir, fl, g.dim, s.dim)
            tot += g.dim*mult; nb += 1
            if g.dim > maxdim:
                continue
            if g.dim <= 200:
                Hg = dense(g); assert np.abs(Hg-Hg.conj().T).max() < 1e-12, 'not Hermitian'
                d = float(np.abs(np.linalg.eigvalsh(Hg)-np.linalg.eigvalsh(dense(s))).max()); nd += 1
            else:
                d = abs(g.ground_energy(tol=1e-13)[0]-s.ground_energy(tol=1e-13)[0])
            worst = max(worst, d)
            assert d < 1e-9, (kind, A1, A2, nup, k, ir, fl, g.dim, d)
    print(f'  {kind} {A1},{A2}, {N} sites, shells {shells}, delta {delta}: {p} parts, {nb} blocks ({nd} dense), largest difference {worst:.1e}; '
          f'{time.perf_counter()-t0:.0f} s', flush=True)


print(f'layout engine against the sub engine ({device})')
compare('kagome', (2, 0), (0, 2), (1.0, 0.3))
compare('kagome', (2, 0), (0, 2), (1.0, 0.0, 0.25), delta=0.7)                 # third shell: bonds inside one sublattice; XXZ
compare('kagome', (3, 0), (0, 2), (1.0, -0.2), nups=None if not quick else (9, 12))
compare('triangular', (3, 0), (0, 3), (1.0, 0.15))                             # J2 on the triangular lattice joins sites of one class
compare('triangular', (2, 2), (-2, 4), (1.0, 0.1), delta=1.3)
compare('square', (4, 0), (0, 4), (1.0, 0.5))
compare('square', (4, 2), (-2, 4), (1.0, 0.4), nups=(10, 11, 13) if quick else None)
if not quick:
    compare('kagome', (2, 2), (-2, 2), (1.0, 0.3), nups=(12, 13, 16))
    compare('triangular', (3, 3), (-3, 6), (1.0, 0.1), nups=(18, 21))
    compare('square', (4, 4), (-4, 4), (1.0, 0.5), nups=(22, 25), maxdim=600000)

# several levels, retune, operator, batched on one block
cl = L.LatticeCluster('kagome', (3, 0), (0, 2)); N = cl.N; m1 = xxz(cl, (1.0, 0.2)); nup = 9
k, ir, fl, pops, mult = hs.sectors(cl, nup, N-nup)[0]
g = edc_glayout.LayoutEngine(m1, nup, cl, k, ir, fl, device=device, point_ops=pops)
s = edc_sub.SubEngine(m1, nup, cl, k, ir, fl, device=device, point_ops=pops, stored=0)
lg, ls = g.ground_energy(tol=1e-12, nev=4)[0], s.ground_energy(tol=1e-12, nev=4)[0]
assert np.abs(np.array(lg)-np.array(ls)).max() < 1e-9, (lg, ls)
m2 = xxz(cl, (1.0, -0.1)); g.retune(m2); s2 = edc_sub.SubEngine(m2, nup, cl, k, ir, fl, device=device, point_ops=pops, stored=0)
Eg, Es = g.ground_energy(tol=1e-13)[0], s2.ground_energy(tol=1e-13)[0]; assert abs(Eg-Es) < 1e-10, (Eg, Es)
psi_g, _, res = hs.ground_state_vector(g); psi_s, _, _ = hs.ground_state_vector(s2); assert res < 1e-5, res
Q = em.reciprocal(cl)[0]*0.0
for Qv in ([0.0, 0.0], list(2*np.pi*np.linalg.inv(cl.M.astype(float)).T @ np.array([1.0, 0.0]))):
    if em.compatible(cl, np.array(Qv)):
        a, b = em.structure_factor(g, psi_g, cl, np.array(Qv))['S'], em.structure_factor(s2, psi_s, cl, np.array(Qv))['S']
        assert abs(a-b) < 1e-7, (Qv, a, b)
xp = g.xp; X = xp.asarray(np.random.default_rng(1).standard_normal((g.dim, 3)).astype(g.dtype)); Y = xp.zeros_like(X); g.batched(3)(X, Y)
y1 = xp.zeros(g.dim, g.dtype); g.matvec(xp.ascontiguousarray(X[:, 1]), y1); assert float(abs(Y[:, 1]-y1).max()) < 1e-13
print('  several levels, retune, structure factor, batched: ok')
if device == 'gpu':                                                # the fold computed in the kernel against the fold through tables
    worst = 0.0
    for kind, A1, A2, sh, nup in (('kagome', (3, 0), (0, 2), (1.0, 0.3), 9), ('kagome', (2, 2), (-2, 2), (1.0, 0.2), 13), ('triangular', (2, 2), (-2, 4), (1.0, 0.1), 6),
                                  ('square', (4, 2), (-2, 4), (1.0, 0.5), 10), ('square', (4, 0), (0, 4), (1.0, 0.5), 9)):
        cl = L.LatticeCluster(kind, A1, A2); m = xxz(cl, sh, 0.8)
        for k, ir, fl, pops, mult in hs.sectors(cl, nup, cl.N-nup):
            ga = edc_glayout.LayoutEngine(m, nup, cl, k, ir, fl, device='gpu', point_ops=pops, fold='fly')
            if ga.dim == 0:
                continue
            gb = edc_glayout.LayoutEngine(m, nup, cl, k, ir, fl, device='gpu', point_ops=pops, fold='table'); assert ga.dim == gb.dim
            x = xp.asarray(np.random.default_rng(3).standard_normal(ga.dim).astype(ga.dtype)); ya = xp.zeros_like(x); yb = xp.zeros_like(x)
            if not ga.real:
                x = x + 1j*xp.asarray(np.random.default_rng(4).standard_normal(ga.dim))
            ga.matvec(x, ya); gb.matvec(x, yb); worst = max(worst, float(abs(ya-yb).max()))
            gc = edc_glayout.LayoutEngine(m, nup, cl, k, ir, fl, device='gpu', point_ops=pops, fold='scatter'); yc = xp.zeros_like(x)
            gc.matvec(x, yc); worst = max(worst, float(abs(yc-yb).max()))
            gd = edc_glayout.LayoutEngine(m, nup, cl, k, ir, fl, device='gpu', point_ops=pops, expand='fly'); assert gd.dim == gb.dim
            yd = xp.zeros_like(x); gd.matvec(x, yd)
            # the two bases hold the same representatives in the same order (home entries, ascending), so vectors compare directly
            worst = max(worst, float(abs(yd-yb).max())); del ga, gb, gc, gd
    assert worst < 1e-12, worst
    print(f'  fold in the kernel, fold from the layout side and no tables at all against the table form, five clusters, every block of one sector: {worst:.1e}')
print('test_glayout: ok')
