#!/usr/bin/env python3
"""Hubbard engine on triangular, honeycomb, kagome (and square) clusters: every sector of the complete list
(edc_hubbard_sym.sectors) against the independent projector reference of test_hubbard_sym; the sector spectra, each
counted with its momentum-orbit multiplicity, must rebuild the full spectrum.  Usage: test_hubbard_lattices.py [cpu|gpu]"""
import sys
import numpy as np, scipy.sparse as sps
import edc_lattices as L, edc_hubbard_sym as hs
src = open(__file__.replace('test_hubbard_lattices.py', 'test_hubbard_sym.py')).read()
exec(src[:src.index('ROT = (')])                   # full_hamiltonian, perm_sign (reference implementation)
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cpu'


def ref_sector(H, uu, dd, idx, L_, nup, ndn, perms, chars, fl):
    elems = [(p, c, 0) for p, c in zip(perms, chars)]
    if fl:
        elems += [(p, c*fl, 1) for p, c in zip(perms, chars)]
    imgs, sig = [], []
    for p, c, f in elems:
        a, b = (dd, uu) if f else (uu, dd)
        imgs.append(idx(hs.apply_perm(p, a), hs.apply_perm(p, b)))
        sig.append(((-1.0)**(nup*ndn) if f else 1.0)*perm_sign(p, a)*perm_sign(p, b))
    imgs = np.array(imgs); keyv = uu*(1 << L_)+dd
    reps = np.flatnonzero(keyv == keyv[imgs].min(axis=0))
    cols = np.repeat(np.arange(len(reps)), len(elems))
    vv = (np.conj(np.array([c for _, c, _ in elems]))[None, :]*np.array(sig)[:, reps].T).ravel()
    Pm = sps.csc_matrix((vv, (imgs[:, reps].T.ravel(), cols)), shape=(H.shape[0], len(reps)))
    nrm = np.sqrt(np.asarray(abs(Pm.multiply(Pm.conj())).sum(axis=0))).ravel(); keep = np.flatnonzero(nrm > 1e-8)
    Q = Pm[:, keep] @ sps.diags(1/nrm[keep]); Hs = (Q.conj().T @ H @ Q).toarray()
    return np.linalg.eigvalsh((Hs+Hs.conj().T)/2) if len(keep) else np.array([])


for kind, A1, A2, U, nu, nd in (('triangular', (2, 2), (-2, 4), 6.0, 3, 3), ('triangular', (2, 2), (-2, 4), 4.0, 4, 3),
                                ('honeycomb', (2, 0), (0, 2), 4.0, 4, 4), ('kagome', (1, 1), (-1, 2), 4.0, 4, 4),
                                ('kagome', (2, 0), (0, 2), 6.0, 3, 3), ('square', (3, 0), (0, 3), 4.0, 4, 4)):
    cl = L.LatticeCluster(kind, A1, A2); bonds = hs.hubbard_bonds(cl, 1.0)
    su, sd = hs.species_strings(cl.N, nu), hs.species_strings(cl.N, nd)
    H, uu, dd, idx = full_hamiltonian(cl.N, bonds, U, su, sd)
    full = np.linalg.eigvalsh(H.toarray()) if H.shape[0] <= 6000 else None
    allev, worst, tot, nsec = [], 0.0, 0, 0
    for k, ir, fl, pops, mult in hs.sectors(cl, nu, nd):
        perms, chars, _ = cl.group(k, ir, allowed=pops)
        ev = ref_sector(H, uu, dd, idx, cl.N, nu, nd, perms, chars, fl)
        e = hs.HubbardSym(cl, bonds, U, nu, nd, k, ir, fl, device=DEV, point_ops=pops)
        assert e.dim == len(ev), (kind, k, ir, fl, e.dim, len(ev))
        tot += mult*e.dim; allev += [ev]*mult
        if e.dim:
            worst = max(worst, abs(e.ground_energy(tol=1e-13)[0]-ev[0])); nsec += 1
    allev = np.sort(np.concatenate(allev))
    spec = f'full spectrum {np.max(np.abs(allev-full)):.1e}' if full is not None else ''
    print(f'{kind:10s} {A1} {A2} N={cl.N:2d} U={U} N_up={nu} N_dn={nd}: {nsec} sectors, sum mult x dim {tot}/{H.shape[0]}, '
          f'{spec}; max |E_engine - ref| {worst:.1e}', flush=True)
