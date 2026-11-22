#!/usr/bin/env python3
"""Hubbard model with lattice symmetries: generated engine (edc_hubbard_sym) against an independent reference.

Reference: the full (N_up, N_dn) Hamiltonian in plain numpy (hop signs from moving c_y and c+_x through the ordered
product, not from the 'occupied between' rule), U(g) from explicit inversion counting over site pairs, sector bases
from the projector P = (1/|G|) sum_g chi(g)^* U(g) applied to orbit representatives.
Checks: (1) the sector spectra together equal the full spectrum (small cases) and the sector sizes add up to the full
dimension; (2) the engine's two lowest levels in every sector equal the reference's.
Usage: test_hubbard_sym.py [cpu|gpu]"""
import sys, math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import edc_lattice as el
import edc_hubbard_sym as hs

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
pc = np.bitwise_count


def full_hamiltonian(L, bonds, U, su, sd):
    Du, Dd = len(su), len(sd)
    uu, dd = np.repeat(su, Dd), np.tile(sd, Du)
    n = Du*Dd
    idx = lambda u, d: np.searchsorted(su, u)*Dd+np.searchsorted(sd, d)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [U*pc(uu & dd).astype(float)]
    for (a, b), t in bonds.items():
        for sp in (0, 1):
            s = uu if sp == 0 else dd
            ok = np.flatnonzero(((s >> a) & 1) != ((s >> b) & 1))
            s0 = s[ok]
            y = np.where((s0 >> a) & 1, a, b); x = np.where((s0 >> a) & 1, b, a)    # c+_x c_y, y occupied
            n1 = pc(s0 & ((np.int64(1) << y)-1))                                    # c_y moves past occupied < y
            s1 = s0 ^ (np.int64(1) << y)
            n2 = pc(s1 & ((np.int64(1) << x)-1))                                    # c+_x moves to its place
            s2 = s1 | (np.int64(1) << x)
            sign = np.where((n1+n2) % 2, -1.0, 1.0)
            new = idx(s2, dd[ok]) if sp == 0 else idx(uu[ok], s2)
            rows.append(new); cols.append(ok); vals.append(-t*sign)
    H = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return H, uu, dd, idx


def perm_sign(p, s):
    inv = np.zeros_like(s)
    L = len(p)
    for a in range(L):
        for b in range(a+1, L):
            if p[a] > p[b]:
                inv ^= ((s >> a) & 1) & ((s >> b) & 1)
    return np.where(inv, -1.0, 1.0)


ROT = ('E', 'C4', 'C2', 'C4^3')


def reference_sectors(cl, bonds, U, nup, ndn, flips, point_ops):
    L = cl.N
    su, sd = hs.species_strings(L, nup), hs.species_strings(L, ndn)
    H, uu, dd, idx = full_hamiltonian(L, bonds, U, su, sd)
    n = H.shape[0]
    out = {}
    for k in cl.momenta():
        lg = [o for o in cl.little_group(k) if point_ops is None or o in point_ops]
        seen = set()
        for ir in el.IRREPS:
            if [o for o in lg if o in el.IRREPS[ir]] != lg:
                continue
            key = tuple(complex(el.IRREPS[ir][o]) for o in lg)
            if key in seen:
                continue
            seen.add(key)
            perms, chars, _ = cl.group(k, ir, lg)
            for fl in flips:
                elems = [(p, c, 0) for p, c in zip(perms, chars)]
                if fl:
                    elems += [(p, c*fl, 1) for p, c in zip(perms, chars)]
                # images and signs of every basis state under every element
                imgs, sig = [], []
                for p, c, f in elems:
                    a, b = (dd, uu) if f else (uu, dd)
                    s0 = (-1.0)**(nup*ndn) if f else 1.0
                    imgs.append(idx(hs.apply_perm(p, a), hs.apply_perm(p, b)))
                    sig.append(s0*perm_sign(p, a)*perm_sign(p, b))
                imgs = np.array(imgs); keyv = uu*(1 << L)+dd
                reps = np.flatnonzero(keyv == keyv[imgs].min(axis=0))
                cols = np.repeat(np.arange(len(reps)), len(elems))
                rws = imgs[:, reps].T.ravel()
                vv = (np.conj(np.array([c for _, c, _ in elems]))[None, :]*np.array(sig)[:, reps].T).ravel()
                Pm = sps.csc_matrix((vv, (rws, cols)), shape=(n, len(reps)))
                nrm = np.sqrt(np.asarray(abs(Pm.multiply(Pm.conj())).sum(axis=0))).ravel()
                keep = np.flatnonzero(nrm > 1e-8)
                Q = Pm[:, keep] @ sps.diags(1/nrm[keep])
                Hs = (Q.conj().T @ H @ Q).toarray()
                ev = np.linalg.eigvalsh((Hs+Hs.conj().T)/2) if len(keep) else np.array([])
                out[(k, ir, fl)] = ev
    return out, H


def two_lowest(ev):
    o = []
    for e in ev:
        if not o or abs(e-o[-1]) > 1e-9:
            o.append(e)
    return o[:2]


CASES = [('tilted (2,2)', el.tilted(2, 2), 4.0, 3, 3, True),
         ('tilted (2,2)', el.tilted(2, 2), 4.0, 3, 2, False),
         ('4x3', el.Cluster((4, 0), (0, 3)), 8.0, 3, 3, True),
         ('tilted (3,1)', el.tilted(3, 1), 4.0, 4, 4, True),
         ('tilted (3,1)', el.tilted(3, 1), 6.0, 5, 4, False)]
for name, cl, U, nu, nd, flip in CASES:
    bonds = hs.hubbard_bonds(cl, 1.0)
    flips = (1, -1) if flip else (0,)
    for pops, label in ((ROT, 'rotations'), (None, 'full point group, 1D irreps')):
        ref, H = reference_sectors(cl, bonds, U, nu, nd, flips, pops)
        n = H.shape[0]
        dims = sum(len(v) for v in ref.values())
        spec = ''
        if pops is ROT:                               # all irreps 1D: the sectors must reproduce the whole space
            allev = np.sort(np.concatenate(list(ref.values())))
            if n <= 4000:
                full = np.linalg.eigvalsh(H.toarray())
                spec = f'; full spectrum {np.max(np.abs(full-allev)):.1e}'
            else:
                # ARPACK with few vectors can miss copies of a degenerate level: ask for 20, compare the lowest 8
                v0 = np.random.default_rng(1).standard_normal(n)
                full = np.sort(spl.eigsh(H, k=20, ncv=80, v0=v0, tol=1e-12, which='SA', return_eigenvectors=False))
                spec = f'; eight lowest {np.max(np.abs(full[:8]-allev[:8])):.1e}'
        worst, nsec, dsum = 0.0, 0, 0
        for (k, ir, fl), ev in ref.items():
            e = hs.HubbardSym(cl, bonds, U, nu, nd, k, ir, fl, device=DEVICE, point_ops=pops)
            assert e.dim == len(ev), (k, ir, fl, e.dim, len(ev))
            dsum += e.dim
            if not len(ev):
                continue
            r = two_lowest(ev)
            E = e.ground_energy(tol=1e-13, nev=len(r))[0]
            E = [E] if not isinstance(E, list) else E
            worst = max(worst, max(abs(a-b) for a, b in zip(E, r))); nsec += 1
        print(f'{name:13s} N={cl.N:2d} U={U} N_up={nu} N_dn={nd} flip={flip!s:5s} {label}: reference dims {dims}/{n}{spec}; '
              f'engine ({DEVICE}): {nsec} sectors, dims equal, max |E_i - ref_i| (i = 0, 1) {worst:.1e}', flush=True)
