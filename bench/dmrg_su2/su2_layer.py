#!/usr/bin/env python3
"""Total spin and momentum together on the layers of a kagome cylinder: the smallest exact case.

A layer (a chain ring or a layer of single spins) gets a basis of multiplets |j, k, alpha> : total spin j of the layer,
momentum k around the cylinder, and a counter.  The coupling of two layers, sum_q X_q . Y_-q, is a scalar product of
two rank-1 spin operators, so only reduced matrix elements between multiplets are stored (one "reduced term" instead
of the three components z, +, -).  A cylinder of three layers is then solved in the basis
        | ((a s) j_b, r) total spin 0, total momentum K >
(a, s, r: multiplets of the left, middle and right layer), which is the form the tensor of the middle layer has in a
DMRG with labels (j, k) on the bonds.  The recoupling coefficients are not taken from formulas: they are computed
once from explicit Clebsch-Gordan tables, so no phase convention can go wrong unnoticed.

Checks: the number of basis states equals the number of singlets of each momentum, and all levels equal the singlet
levels of the same cylinder from the product basis.

Usage: su2_layer.py [c]      (c = 2: 10 spins, c = 4: 20 spins)"""
import sys, os, math, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, scipy.sparse as sp, edc_dmrg as ed
from functools import lru_cache

f = math.factorial


@lru_cache(None)
def cg(tj1, tm1, tj2, tm2, tJ, tM):
    """Clebsch-Gordan coefficient <j1 m1; j2 m2 | J M>, arguments doubled (Racah's formula, Condon-Shortley phases)."""
    if tm1+tm2 != tM or tJ < abs(tj1-tj2) or tJ > tj1+tj2 or (tj1+tj2+tJ) % 2 or abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tM) > tJ:
        return 0.0
    if (tj1-tm1) % 2 or (tj2-tm2) % 2 or (tJ-tM) % 2:
        return 0.0
    h = lambda x: x//2
    pre = (tJ+1)*f(h(tJ+tj1-tj2))*f(h(tJ-tj1+tj2))*f(h(tj1+tj2-tJ))/f(h(tj1+tj2+tJ)+1)
    pre *= f(h(tJ+tM))*f(h(tJ-tM))*f(h(tj1-tm1))*f(h(tj1+tm1))*f(h(tj2-tm2))*f(h(tj2+tm2)); s = 0.0
    for k in range(0, h(tj1+tj2-tJ)+1):
        d = [k, h(tj1+tj2-tJ)-k, h(tj1-tm1)-k, h(tj2+tm2)-k, h(tJ-tj2+tm1)+k, h(tJ-tj1-tm2)+k]
        if min(d) >= 0:
            s += (-1)**k/np.prod([float(f(x)) for x in d])
    return math.sqrt(pre)*s


def unit(tjp, tj):
    """The three components q = -1, 0, 1 of a rank-1 operator with reduced element 1 between spin j and spin j':
    <j' m'| U_q |j m> = <j m; 1 q | j' m'>."""
    return [np.array([[cg(tj, tm, 2, 2*q, tjp, tmp) for tm in range(-tj, tj+1, 2)] for tmp in range(-tjp, tjp+1, 2)]) for q in (-1, 0, 1)]


def coupled(tja, tjs, tjb):
    """|((ja js) jb, jb) 0> as an array over (ma, ms, mr)."""
    v = np.zeros((tja+1, tjs+1, tjb+1))
    for ia, tma in enumerate(range(-tja, tja+1, 2)):
        for i_s, tms in enumerate(range(-tjs, tjs+1, 2)):
            tmb = tma+tms
            if abs(tmb) <= tjb:
                v[ia, i_s, (-tmb+tjb)//2] = cg(tja, tma, tjs, tms, tjb, tmb)*cg(tjb, tmb, tjb, -tmb, 0, 0)
    return v


@lru_cache(None)
def w_left(tja, tjs, tjb, tjap, tjsp):
    """<((ja' js') jb, jb) 0| U(ja' ja) . U(js' js) |((ja js) jb, jb) 0>: a term on the left bond and the layer."""
    if tjap < 0 or tjsp < 0:
        return 0.0
    A, B = unit(tjap, tja), unit(tjsp, tjs); v = coupled(tja, tjs, tjb); w = coupled(tjap, tjsp, tjb)
    if not (np.any(v) and np.any(w)):
        return 0.0
    return float(sum((-1)**q*np.einsum('abr,xa,yb,xyr->', v, A[q+1], B[-q+1], w) for q in (-1, 0, 1)))


@lru_cache(None)
def w_right(tja, tjs, tjb, tjsp, tjbp):
    """<((ja js') jb', jb') 0| U(js' js) . U(jb' jb) |((ja js) jb, jb) 0>: a term on the layer and the right block."""
    if tjsp < 0 or tjbp < 0:
        return 0.0
    A, B = unit(tjsp, tjs), unit(tjbp, tjb); v = coupled(tja, tjs, tjb); w = coupled(tja, tjsp, tjbp)
    if not (np.any(v) and np.any(w)):
        return 0.0
    return float(sum((-1)**q*np.einsum('abr,yb,zr,ayz->', v, A[q+1], B[-q+1], w) for q in (-1, 0, 1)))


class LayerSU2:
    """Multiplets of one site kind in the coordinates of its real (S_z, k) basis."""
    def __init__(self, kd):
        self.kd = kd; n = kd.n; d = kd.S; dense = lambda t: sp.csr_matrix((t[2], (t[0], t[1])), shape=(d, d)).toarray()
        Sp = dense(tuple(np.real(x) if i == 2 else x for i, x in enumerate(kd.transform(sum(kd.sp[1:], kd.sp[0]), 1, 0)))); Sm = Sp.T
        self.H = dense(kd.H); self.mult = []; self.vec = {}           # mult: (2j, k); vec[(i, 2m)]: component with S_z = m
        for k in range(kd.c):
            for m in range(n, -1, -1):
                idx = kd.blocks.get((m, k)); tj = 2*m - n
                if idx is None or tj < 0:
                    continue
                up = kd.blocks.get((m+1, k)); B = Sp[np.ix_(up, idx)] if up is not None else np.zeros((0, len(idx)))
                if B.shape[0]:
                    u, s, vh = np.linalg.svd(B); null = vh[np.sum(s > 1e-10):]
                else:
                    null = np.eye(len(idx))
                for row in null:
                    v = np.zeros(d); v[idx] = row; i = len(self.mult); self.mult.append((tj, k)); self.vec[(i, tj)] = v
                    for tm in range(tj, -tj, -2):
                        jj, mm = tj/2, tm/2; v = Sm @ v/math.sqrt(jj*(jj+1)-mm*(mm-1)); self.vec[(i, tm-2)] = v
        assert sum(t+1 for t, k in self.mult) == d
        self.by = {}
        for i, q in enumerate(self.mult):
            self.by.setdefault(q, []).append(i)
        self.Hred = {q: np.array([[self.vec[(a, q[0])] @ self.H @ self.vec[(b, q[0])] for b in ix] for a in ix]) for q, ix in self.by.items()}

    def reduced(self, op):
        """Reduced elements of a rank-1 operator given by its z component (out, in, value) in the real layer basis:
        {(label', label): matrix over the multiplets of the two labels}."""
        d = self.kd.S; O = sp.csr_matrix((op[2], (op[0], op[1])), shape=(d, d)); out = {}
        for q1, ix1 in self.by.items():
            for q0, ix0 in self.by.items():
                tm = min(q1[0], q0[0]); c0 = cg(q0[0], tm, 2, 0, q1[0], tm)
                if abs(c0) < 1e-12:
                    continue
                M = np.array([[self.vec[(a, tm)] @ (O @ self.vec[(b, tm)]) for b in ix0] for a in ix1])/c0
                if np.abs(M).max() > 1e-11:
                    out[(q1, q0)] = M
        return out


def singlet_levels(c):
    """Three layers (ring, single spins, ring) of the YC(2c) cylinder; all singlet levels of every momentum, in the
    multiplet basis with numerically derived recoupling."""
    lat = ed.kagome_cylinder(2, c); model = ed.Model(lat); assert model.n == 3
    L = [LayerSU2(model.kinds[s]) for s in model.site_kind]; b01, b12 = model.bkinds[model.bond_kind[0]], model.bkinds[model.bond_kind[1]]
    z01 = [t for t in range(b01.NT) if b01.dm[t] == 0]; z12 = [t for t in range(b12.NT) if b12.dm[t] == 0]
    X0 = [L[0].reduced(b01.X[t]) for t in z01]; Y1 = [L[1].reduced(b01.Y[t]) for t in z01]
    X1 = [L[1].reduced(b12.X[t]) for t in z12]; Y2 = [L[2].reduced(b12.Y[t]) for t in z12]
    info = dict(states=[lay.kd.S for lay in L], multiplets=[len(lay.mult) for lay in L], terms=(b01.NT, b12.NT), reduced_terms=(len(z01), len(z12)),
                nnz_states=sum(len(b01.X[t][2]) for t in range(b01.NT)), nnz_reduced=sum(int(np.sum(np.abs(M) > 1e-11)) for X in X0 for M in X.values()))
    out = {}
    for K in range(c):
        basis = [(a, s, r) for a, (tja, ka) in enumerate(L[0].mult) for s, (tjs, ks) in enumerate(L[1].mult) for r, (tjr, kr) in enumerate(L[2].mult)
                 if (ka+ks+kr) % c == K and abs(tja-tjs) <= tjr <= tja+tjs and (tja+tjs+tjr) % 2 == 0]
        pos = {b: i for i, b in enumerate(basis)}; H = np.zeros((len(basis), len(basis)))
        for (a, s, r), i in pos.items():
            (tja, ka), (tjs, ks), (tjr, kr) = L[0].mult[a], L[1].mult[s], L[2].mult[r]
            for lay, x, slot in ((L[0], a, 0), (L[1], s, 1), (L[2], r, 2)):                    # Hamiltonians inside the layers
                q = lay.mult[x]; ix = lay.by[q]; col = lay.Hred[q][:, ix.index(x)]
                for y, v in zip(ix, col):
                    key = [a, s, r]; key[slot] = y; H[pos[tuple(key)], i] += v
            for t, tt in enumerate(z01):                                                       # layer 0 with layer 1
                for (qa, q0), MX in X0[t].items():
                    if q0 != (tja, ka):
                        continue
                    for (qs, q1), MY in Y1[t].items():
                        if q1 != (tjs, ks):
                            continue
                        w = w_left(tja, tjs, tjr, qa[0], qs[0])
                        if w == 0.0:
                            continue
                        ca = MX[:, L[0].by[q0].index(a)]; cs = MY[:, L[1].by[q1].index(s)]
                        for a2, va in zip(L[0].by[qa], ca):
                            for s2, vs in zip(L[1].by[qs], cs):
                                if (a2, s2, r) in pos:
                                    H[pos[(a2, s2, r)], i] += b01.cf[tt]*w*va*vs
            for t, tt in enumerate(z12):                                                       # layer 1 with layer 2
                for (qs, q1), MX in X1[t].items():
                    if q1 != (tjs, ks):
                        continue
                    for (qr, q2), MY in Y2[t].items():
                        if q2 != (tjr, kr):
                            continue
                        w = w_right(tja, tjs, tjr, qs[0], qr[0])
                        if w == 0.0:
                            continue
                        cs = MX[:, L[1].by[q1].index(s)]; cr = MY[:, L[2].by[q2].index(r)]
                        for s2, vs in zip(L[1].by[qs], cs):
                            for r2, vr in zip(L[2].by[qr], cr):
                                if (a, s2, r2) in pos:
                                    H[pos[(a, s2, r2)], i] += b12.cf[tt]*w*vs*vr
        assert np.abs(H-H.T).max() < 1e-10, f'not symmetric: {np.abs(H-H.T).max()}'
        out[K] = np.linalg.eigvalsh(H) if len(basis) else np.zeros(0)
    return lat, out, info


def reference(lat):
    """Singlet levels of every momentum from the product basis: S_z = 0 levels that are absent from S_z = 1."""
    import scipy.sparse.linalg as sla
    N = lat.N; bonds = lat.global_bonds(); tp = lat.translation(); c = lat.c; res = {}
    def sector(nup):
        st = np.array([sum(1 << b for b in cb) for cb in itertools.combinations(range(N), nup)], dtype=np.int64); st.sort(); n = len(st)
        diag = np.zeros(n); rows, cols, vals = [], [], []
        for i, j, J in bonds:
            bi, bj = (st >> i) & 1, (st >> j) & 1; diag += J*np.where(bi == bj, 0.25, -0.25); sel = np.nonzero(bi != bj)[0]
            rows.append(np.searchsorted(st, st[sel] ^ ((1 << i) | (1 << j)))); cols.append(sel); vals.append(np.full(len(sel), 0.5*J))
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) + sp.diags(diag)
        t = np.zeros(n, dtype=np.int64)
        for a in range(N):
            t |= ((st >> a) & 1) << tp[a]
        return H, sp.csr_matrix((np.ones(n), (np.searchsorted(st, t), np.arange(n))), shape=(n, n))
    return sector


if __name__ == '__main__':
    c = int(sys.argv[1]) if len(sys.argv) > 1 else 2; t0 = time.perf_counter(); lat, lev, info = singlet_levels(c); t1 = time.perf_counter()
    print(f'{lat.name}: N = {lat.N}; layer states {info["states"]} -> multiplets {info["multiplets"]}; coupling terms {info["terms"]} -> reduced {info["reduced_terms"]}; '
          f'non-zeros of the left-layer operators {info["nnz_states"]} -> {info["nnz_reduced"]}')
    sector = reference(lat); N = lat.N; ref = {}
    for nup in (N//2, N//2+1):
        H, T = sector(nup); n = H.shape[0]
        if n <= 4000:
            Hd = H.toarray(); Td = T.toarray()
            for K in range(c):
                P = sum(np.exp(2j*np.pi*K*s/c)*np.linalg.matrix_power(Td, s) for s in range(c))/c; w, V = np.linalg.eigh(P); V = V[:, w > 0.5]
                ref[(nup, K)] = np.linalg.eigvalsh(V.conj().T @ Hd @ V) if V.shape[1] else np.zeros(0)
    for K in range(c):
        line = f'  K = {K}: {len(lev[K])} singlets in the multiplet basis, lowest {lev[K][0]:.10f}'
        if (N//2, K) in ref:
            a, b = ref[(N//2, K)], ref[(N//2+1, K)]; sing = sorted(a.tolist())
            for x in b:                                               # remove the levels that continue to S_z = 1
                j = int(np.argmin(np.abs(np.array(sing)-x))); assert abs(sing[j]-x) < 1e-8; sing.pop(j)
            line += f' | product basis: {len(sing)} singlets, lowest {sing[0]:.10f} | largest difference over all levels {np.abs(np.array(sing)-lev[K]).max():.1e}'
        else:
            e = ed.exact_levels(lat.global_bonds(), N, N//2, 1.0, tperm=lat.translation(), c=c)[K]; e1 = ed.exact_levels(lat.global_bonds(), N, N//2+1, 1.0, tperm=lat.translation(), c=c)[K]
            line += f' | product basis: lowest S_z = 0 level {e:.10f} (lowest S_z = 1 level {e1:.10f})'
        print(line)
    print(f'  time for the multiplet calculation {t1-t0:.1f} s')
