#!/usr/bin/env python3
"""Where a state sits along the cylinder, read from a finished layer-DMRG engine (host engine, or GPU back end v1 or v3):

  layer_sz      <S_z> of every layer, from the weights of the bond labels;
  left, right   with the centre on layer r: left[r] = <H(layers 0..r-1)> from the left environment LH[r] (the
                Hamiltonian of those layers in the kept basis), right[r] = <H(layers r+1..n-1)> from RH[r+1];
  energy_steps  left[r+1] - left[r] (bonds inside layer r plus the bonds between layers r-1 and r); the last from E;
  rows          energy of every row of bonds: 'ring' (the 2c bonds inside a ring), 'in' (the 2c bonds between a
                ring and the three-spin... the c-spin layer before it), 'out' (between a ring and the layer after it).
                'out' of ring r = left step of layer r+1, 'in' of ring r = right step of layer r-1, 'ring' of ring r =
                (left step of r) - 'in', and independently (right step of r) - 'out'; ring_mismatch is the largest
                difference between the two, row_sum_minus_E the sum of all rows minus the energy of the run.  Both are
                independent checks: the rows come from environments, E from the layer eigenvalue.

Rows assume the kagome YC layering (rings at even layers, c-spin layers at odd layers, no last layer).
Moves the centre from the first layer to the last without truncating; call after the run."""
import numpy as np


def _host(a):                                                          # v3 keeps tensors and environments on the device (cupy)
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def profiles(eng, D, nup, E):
    n = eng.n; mleft = []; left = []; right = []; dev = hasattr(eng, '_dev')
    if dev:
        import edc_dmrg_gpu as eg
    for r in range(n):
        p = eng.plan[r]; b = eng.bond[r]; M = _host(eng.M[r]); M = M/np.linalg.norm(M); w = np.zeros(p.nL); el = 0.0; er = 0.0
        LH = _host(eng.LH[r])
        for i in range(p.nL):
            Di, Ci = int(p.DL[i]), int(p.C[i])
            if not Di or not Ci:
                continue
            Mi = M[p.Moff[i]:p.Moff[i+1]].reshape(Di, Ci); w[i] = float(np.sum(Mi*Mi))
            if r > 0:
                el += float(np.sum(Mi*(LH[p.LHoff[i]:p.LHoff[i+1]].reshape(Di, Di) @ Mi)))
        if dev:                                                        # v3 moves from the device product of the last solve of this
            H = eg.Product(eng, r, eng.prof); x = eng.M[r]/float((eng.M[r] @ eng.M[r])**0.5)   # layer; build it for the unchanged tensor
            U = _host(H.to_u(x)); eng._dev = (r, H, x)
        else:
            U = np.empty(p.size); eng.lib.m2u(p.ref, M.ctypes.data, U.ctypes.data)
        if r < n-1:
            RH = _host(eng.RH[r+1])
            for j in range(p.nR):
                Rj, Dj = int(p.R[j]), int(p.DR[j])
                if Rj and Dj:
                    Uj = U[p.Uoff[j]:p.Uoff[j+1]].reshape(Rj, Dj); er += float(np.sum(Uj*(Uj @ RH[p.RHoff[j]:p.RHoff[j+1]].reshape(Dj, Dj))))
        mleft.append(float((b.m*w).sum())); left.append(el); right.append(er)
        if r < n-1:
            eng.move_right(r, D, 0.0)
    sizes = np.array([len(y) for y in eng.model.lat.layers]); sz = np.diff(mleft + [float(nup)]) - sizes/2.0
    e = np.diff(left); f = -np.diff(right)                             # e[r]: left step of layer r (r <= n-2); f[r-1]: right step of layer r (r >= 1)
    out = dict(layer_sz=[float(x) for x in sz], left=left, right=right, energy_steps=[float(x) for x in e] + [float(E-left[-1])])
    if n % 2 == 1 and all(len(sizes) > 2 and sizes[r] == 2*sizes[1] for r in range(0, n, 2)):
        rows = []; mis = 0.0
        for r in range(0, n, 2):
            ein = f[r-2] if r >= 2 else None                           # bonds between layer r-1 and ring r = right step of layer r-1
            eout = e[r+1] if r+1 <= n-2 else None                      # bonds between ring r and layer r+1 = left step of layer r+1
            a = (e[r] - (ein or 0.0)) if r <= n-2 else None; c = (f[r-1] - (eout or 0.0)) if r >= 1 else None
            if a is not None and c is not None:
                mis = max(mis, abs(a-c))
            ring = a if c is None else c if a is None else 0.5*(a+c)
            if ein is not None:
                rows.append(dict(kind='in', ring=r//2, x=r-0.5, E=float(ein)))
            rows.append(dict(kind='ring', ring=r//2, x=float(r), E=float(ring)))
            if eout is not None:
                rows.append(dict(kind='out', ring=r//2, x=r+0.5, E=float(eout)))
        out.update(rows=rows, ring_mismatch=float(mis), row_sum_minus_E=float(sum(q['E'] for q in rows) - E))
    return out


def exact_rows(lat, nup):
    """<S_z> per layer and the energy of every row of bonds from the exact lowest state of the S_z sector (small cylinders)."""
    import itertools, scipy.sparse as sp, scipy.sparse.linalg as sla
    N = lat.N; states = [sum(1 << i for i in c) for c in itertools.combinations(range(N), nup)]; idx = {s: k for k, s in enumerate(states)}
    off = lat.offset; group = {}
    for l, bl in enumerate(lat.intra):
        for a, b, J in bl:
            group.setdefault(('ring', float(l)), []).append((int(off[l]+a), int(off[l]+b), J))
    for l, bl in enumerate(lat.inter):
        for a, b, J in bl:
            group.setdefault(('out', l+0.5) if l % 2 == 0 else ('in', l+0.5), []).append((int(off[l]+a), int(off[l+1]+b), J))

    def matrix(bonds):
        rows, cols, vals = [], [], []
        for k, s in enumerate(states):
            d = 0.0
            for a, b, J in bonds:
                if ((s >> a) & 1) == ((s >> b) & 1):
                    d += J/4
                else:
                    d -= J/4; rows.append(idx[s ^ (1 << a) ^ (1 << b)]); cols.append(k); vals.append(J/2)
            rows.append(k); cols.append(k); vals.append(d)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(states),)*2)
    Hs = {g: matrix(b) for g, b in group.items()}; H = sum(Hs.values()); E, v = sla.eigsh(H, k=6, which='SA', tol=1e-13, ncv=60)
    o = np.argsort(E); E = E[o]; v = v[:, o[0]]; single = bool(E[1]-E[0] > 1e-8)   # a degenerate level has no unique profile
    sz = [sum(float(np.sum(v*v*np.array([((s >> i) & 1) - 0.5 for s in states]))) for i in range(off[l], off[l+1])) for l in range(len(lat.layers))]
    return float(E[0]), sz, {g: float(v @ (Hs[g] @ v)) for g in Hs}, single


if __name__ == '__main__':                                             # check against exact diagonalisation, host engine
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    import edc_dmrg as ed
    worst = 0.0
    for Lx, c, D in ((3, 2, 200), (2, 3, 300), (3, 3, 400)):
        lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat)
        for nup in (lat.N//2, lat.N//2+1):
            Ex, szx, ex, single = exact_rows(lat, nup); best = None
            for K in range(c//2+1):
                E, eng, info = ed.ground_state(model, D, nup=nup, K=K, tol=1e-9)
                if best is None or E < best[0]:
                    best = (E, K, profiles(eng, D, nup, E))
            E, K, pr = best; d1 = max(abs(a-b) for a, b in zip(pr['layer_sz'], szx)); d2 = max(abs(q['E'] - ex[(q['kind'], q['x'])]) for q in pr['rows'])
            worst = max(worst, abs(E-Ex), abs(pr['row_sum_minus_E']), pr['ring_mismatch'], *((d1, d2) if single else ()))
            print(f'{lat.name}, N = {lat.N}, S_z = {nup-lat.N/2:g}, K = {K}: E - exact = {E-Ex:+.1e}; rows: sum - E = {pr["row_sum_minus_E"]:+.1e}, ring mismatch '
                  f'{pr["ring_mismatch"]:.1e}; largest difference from exact: layer S_z {d1:.1e}, rows {d2:.1e}' + ('' if single else '  (exact level degenerate: not compared)'))
    print('ok' if worst < 1e-5 else 'FAILED')
