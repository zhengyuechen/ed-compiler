#!/usr/bin/env python3
"""Layer DMRG with total spin and momentum on every bond: CPU reference.

The chain is the layer chain of edc_dmrg (kagome YC cylinders: rings of 2c spins and layers of c spins).  Every bond
label is (2j, k): the total spin and the momentum around the cylinder of everything to its left, as in edc_dmrg, but
with j in place of S_z, so a bond index runs over MULTIPLETS.  Layers carry the multiplet bases of su2_layer.LayerSU2
(real, at every momentum); every coupling of two layers is a sum of scalar products X_t . Y_t of vector operators, so
only reduced matrix elements are stored (one term per q instead of three).

Target.  A ghost multiplet of spin S and momentum -K is appended at the right end and the chain together with it is a
singlet.  So the lowest state with total spin S and momentum K is a ground-state run for every S (S = 0, 1, 1/2, ...).
Bond n (right end) has the single label (2S, K).

Bases.  Left multiplets a' = (a s) j (left block, then the layer); right multiplets b' = (s b) j (the layer, then the
right block).  The tensor at the centre r is C[a, s, b] in the basis |((a s) j_b, b) 0>; left-canonical tensors
A[a, s, a'] and right-canonical B[b', s, b] are orthonormal in the ordinary sense, block by block.  Moving the centre
changes the coupling order of three multiplets with total spin 0, which is a sign phi(j_a, j_s, j_b).

Every recoupling coefficient is computed from explicit Clebsch-Gordan tables (no 6j formula, no phase convention to
get wrong); the scalar products in the effective Hamiltonian are su2_layer.w_left / w_right, checked there against
exact singlet levels of 10 and 20 spins.

Single-site sweeps with a sketched subspace expansion as in edc_dmrg (sketch of the terms with an open leg, plus a few
random directions); the schedule of edc_dmrg.ground_state.  Plain numpy, block by block: a reference for correctness,
not for speed.

Usage: su2_dmrg.py Lx c D S2 K [tail]       (S2 = 2S; prints every sweep)"""
import sys, os, math, time
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
import edc_dmrg as ed
from su2_layer import cg, unit, w_left, w_right, LayerSU2


def tri(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def couplings(ta, tb):
    return range(abs(ta - tb), ta + tb + 1, 2)


@lru_cache(None)
def cpl(t1, t2, tJ):
    """V[m1, m2, M] = <j1 m1; j2 m2 | J M> (doubled spins; index of m = (tm + tj) / 2)."""
    V = np.zeros((t1 + 1, t2 + 1, tJ + 1))
    for i1, m1 in enumerate(range(-t1, t1 + 1, 2)):
        for i2, m2 in enumerate(range(-t2, t2 + 1, 2)):
            if abs(m1 + m2) <= tJ:
                V[i1, i2, (m1 + m2 + tJ)//2] = cg(t1, m1, t2, m2, tJ, m1 + m2)
    return V


def _reduce(T, t1, t2):
    """T[iq][m2, m1] = <j2 m2| O_q |j1 m1> of a rank-1 operator -> its reduced element (least squares over m, q)."""
    C = cpl(t1, 2, t2); num = den = 0.0
    for iq in range(3):
        Cq = C[:, iq, :].T
        num += float((T[iq]*Cq).sum()); den += float((Cq*Cq).sum())
    return num/den if den > 0 else 0.0


@lru_cache(None)
def t_second(ta, ts1, t1, ts2, t2):
    """<((a s2) j2) || U(s2 <- s1) || ((a s1) j1)>: a unit rank-1 operator on the second member of (a s)."""
    if not (tri(ta, ts1, t1) and tri(ta, ts2, t2) and tri(t1, 2, t2) and tri(ts1, 2, ts2)):
        return 0.0
    U = unit(ts2, ts1); A1 = cpl(ta, ts1, t1); A2 = cpl(ta, ts2, t2)
    return _reduce([np.einsum('asM,st,atm->Mm', A2, U[iq], A1) for iq in range(3)], t1, t2)


@lru_cache(None)
def t_first(ts1, tb, t1, ts2, t2):
    """<((s2 b) j2) || U(s2 <- s1) || ((s1 b) j1)>: a unit rank-1 operator on the first member of (s b)."""
    if not (tri(ts1, tb, t1) and tri(ts2, tb, t2) and tri(t1, 2, t2) and tri(ts1, 2, ts2)):
        return 0.0
    U = unit(ts2, ts1); A1 = cpl(ts1, tb, t1); A2 = cpl(ts2, tb, t2)
    return _reduce([np.einsum('sbM,st,tbm->Mm', A2, U[iq], A1) for iq in range(3)], t1, t2)


@lru_cache(None)
def phi(ta, ts, tc):
    """<((a s) j_c, c) 0 | (a, (s c) j_a) 0>, a sign: the same singlet of three multiplets in two coupling orders."""
    S0 = cpl(tc, tc, 0)[:, :, 0]; V1 = np.einsum('asM,Mc->asc', cpl(ta, ts, tc), S0)
    S1 = cpl(ta, ta, 0)[:, :, 0]; V2 = np.einsum('aM,scM->asc', S1, cpl(ts, tc, ta))
    return float((V1*V2).sum())


class SU2Model:
    """Layers with multiplet bases and the couplings between neighbouring layers as scalar products of reduced vector
    operators: terms[l] = [(cf, dk, X on layer l, Y on layer l+1)] for the bond between layers l and l+1."""
    def __init__(self, lat):
        self.lat = lat; self.ab = ed.Model(lat); self.n = self.ab.n; self.c = lat.c
        assert all(f < 0 for f in self.ab.far_kind), 'couplings two layers apart: not written for total spin'
        kinds = [LayerSU2(kd) for kd in self.ab.kinds]; self.lay = [kinds[s] for s in self.ab.site_kind]
        self.sectors = [{q: len(ix) for q, ix in L.by.items()} for L in self.lay]
        self.terms = []
        for l in range(self.n - 1):
            bk = self.ab.bkinds[self.ab.bond_kind[l]]; T = []
            for t in range(bk.NT):
                if bk.dm[t] == 0:
                    T.append((float(bk.cf[t]), int(bk.dk[t]), self.lay[l].reduced(bk.X[t]), self.lay[l+1].reduced(bk.Y[t])))
            self.terms.append(T)
        # the layer Hamiltonians, and the term operators indexed by their source sector
        self.Hs = [L.Hred for L in self.lay]
        self.by_src = lambda red: {q0: [(q1, M) for (q1, qq), M in red.items() if qq == q0] for q0 in {qq for _, qq in red}}


ALPHAS = ed.ALPHAS


class SU2DMRG:
    def __init__(self, model, S2, K, D0=4, seed=0):
        self.m = m = model; self.n = n = m.n; c = m.c; self.rng = rng = np.random.default_rng(seed)
        self.S2, self.K = S2, K % c; self.sketch = 0.2; self.random = 0.03; self.noise = 0.0
        add = lambda L, q: [(tj, (L[1] + q[1]) % c) for tj in couplings(L[0], q[0])]
        left = [{(0, 0)}]
        for r in range(n):
            left.append({x for L in left[-1] for q in m.sectors[r] for x in add(L, q)})
        right = [None]*(n + 1); right[n] = {(S2, self.K)}
        for r in range(n - 1, -1, -1):
            right[r] = {(tj, (L[1] - q[1]) % c) for L in right[r+1] for q in m.sectors[r] for tj in couplings(L[0], q[0])}
        assert (S2, self.K) in left[n], 'this sector does not exist'
        self.labels = [sorted(left[r] & right[r]) for r in range(n + 1)]
        self.D = [dict.fromkeys(self.labels[r], 0) for r in range(n + 1)]; self.D[0][(0, 0)] = 1; self.D[n][(S2, self.K)] = 1
        # operators of the terms by source sector (layer side) for the products
        self.Xsrc = [[m.by_src(X) for (_, _, X, _) in m.terms[l]] for l in range(n - 1)]
        self.Ysrc = [[m.by_src(Y) for (_, _, _, Y) in m.terms[l]] for l in range(n - 1)]
        self.T = [None]*n
        for r in range(n - 1, 0, -1):                                   # random right-canonical start
            B = {}
            for L in self.labels[r]:
                cols = self._cols(r, L)
                nc = sum(m.sectors[r][q]*self.D[r+1][lb] for q, lb in cols)
                d = min(D0, nc); self.D[r][L] = d
                if d:
                    Q = np.linalg.qr(rng.standard_normal((nc, d)))[0].T; o = 0
                    for q, lb in cols:
                        w = m.sectors[r][q]*self.D[r+1][lb]; B[(L, q, lb)] = Q[:, o:o + w].reshape(d, m.sectors[r][q], self.D[r+1][lb]); o += w
            self.T[r] = B
        self.T[0] = self._random_center(0); self._normalize(0)
        self.HL = [None]*(n + 1); self.OL = [None]*(n + 1); self.HR = [None]*(n + 1); self.OR = [None]*(n + 1)
        self.HL[0] = {(0, 0): np.zeros((1, 1))}; self.OL[0] = []; self.HR[n] = {(S2, self.K): np.zeros((1, 1))}; self.OR[n] = []
        for r in range(n - 1, 0, -1):
            self._grow_right(r)
        self.stats = dict(products=0, energies=[], truncation=0.0, t_solve=0.0, t_move=0.0, t_env=0.0)

    # ---- index sets
    def _valid(self, r, la, q, lb):
        return tri(la[0], q[0], lb[0]) and (la[1] + q[1]) % self.m.c == lb[1]

    def _cols(self, r, la):                                             # (sector, right label) of the layer-r blocks with left label la
        return [(q, lb) for q in self.m.sectors[r] for lb in self.labels[r+1] if self.D[r+1][lb] and self._valid(r, la, q, lb)]

    def _rows(self, r, lb):                                             # (left label, sector) of the layer-r blocks with right label lb
        return [(la, q) for la in self.labels[r] if self.D[r][la] for q in self.m.sectors[r] if self._valid(r, la, q, lb)]

    def _keys(self, r):
        return [(la, q, lb) for la in self.labels[r] if self.D[r][la] for (q, lb) in self._cols(r, la)]

    def _random_center(self, r):
        return {(la, q, lb): self.rng.standard_normal((self.D[r][la], self.m.sectors[r][q], self.D[r+1][lb])) for (la, q, lb) in self._keys(r)}

    def _normalize(self, r):
        nrm = math.sqrt(sum(float((x*x).sum()) for x in self.T[r].values())); [x.__itruediv__(nrm) for x in self.T[r].values()]

    # ---- the effective Hamiltonian of the layer at the centre
    def _left_part(self, r, C, out):
        m = self.m; Hs = m.Hs[r]; HL = self.HL[r]
        for (la, q, lb), x in C.items():
            o = out[(la, q, lb)]; o += np.einsum('st,atb->asb', Hs[q], x)
            if r > 0:
                o += np.einsum('xa,asb->xsb', HL[la], x)
        if r > 0:
            for t, (cf, dk, _, _) in enumerate(m.terms[r-1]):
                OL = self.OLsrc[r][t]; Y = self.Ysrc[r-1][t]
                for (la, q, lb), x in C.items():
                    for la2, O in OL.get(la, ()):
                        for q2, Ym in Y.get(q, ()):
                            k2 = (la2, q2, lb)
                            if k2 in out:
                                w = w_left(la[0], q[0], lb[0], la2[0], q2[0])
                                if w:
                                    out[k2] += (cf*w)*np.einsum('xa,ys,asb->xyb', O, Ym, x)

    def _right_part(self, r, C, out, with_layer=False):
        m = self.m; HR = self.HR[r+1]
        for (la, q, lb), x in C.items():
            o = out[(la, q, lb)]; o += np.einsum('asb,bc->asc', x, HR[lb])
            if with_layer:
                o += np.einsum('st,atb->asb', m.Hs[r][q], x)
        if r < self.n - 1:
            for t, (cf, dk, _, _) in enumerate(m.terms[r]):
                OR = self.ORsrc[r+1][t]; X = self.Xsrc[r][t]
                for (la, q, lb), x in C.items():
                    for q2, Xm in X.get(q, ()):
                        for lb2, O in OR.get(lb, ()):
                            k2 = (la, q2, lb2)
                            if k2 in out:
                                w = w_right(la[0], q[0], lb[0], q2[0], lb2[0])
                                if w:
                                    out[k2] += (cf*w)*np.einsum('ys,zb,asb->ayz', Xm, O, x)

    def heff(self, r, C):
        out = {k: np.zeros_like(x) for k, x in C.items()}; self._left_part(r, C, out); self._right_part(r, C, out); return out

    def solve(self, r, tol=1e-6, maxit=40):
        t0 = time.perf_counter(); C = self.T[r]; keys = list(C); shapes = [C[k].shape for k in keys]; sizes = [int(np.prod(s)) for s in shapes]
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(int); n = int(off[-1])
        flat = lambda D: np.concatenate([D[k].ravel() for k in keys])
        unflat = lambda v: {k: v[off[i]:off[i+1]].reshape(shapes[i]) for i, k in enumerate(keys)}
        v = flat(C); v /= np.linalg.norm(v)
        if self.noise > 0:
            xi = self.rng.standard_normal(n); v = v + (self.noise/np.linalg.norm(xi))*xi; v /= np.linalg.norm(v)
        V = [v]; al = []; be = []; E = None; y = np.array([1.0])
        for it in range(min(maxit, n)):
            w = flat(self.heff(r, unflat(V[-1]))); self.stats['products'] += 1
            a = float(V[-1] @ w); al.append(a); w -= a*V[-1]
            if be:
                w -= be[-1]*V[-2]
            for u in V:
                w -= (u @ w)*u
            b = float(np.linalg.norm(w)); ev, Sv = np.linalg.eigh(np.diag(al) + np.diag(be, 1) + np.diag(be, -1)); E = float(ev[0]); y = Sv[:, 0]
            if b*abs(y[-1]) < tol*max(1.0, abs(E)) or b < 1e-13 or len(V) == n:
                break
            be.append(b); V.append(w/b)
        x = sum(cc*u for cc, u in zip(y, V)); x /= np.linalg.norm(x); self.T[r] = unflat(x); self.stats['t_solve'] += time.perf_counter() - t0
        return E

    # ---- environments
    def _src(self, O):                                                  # {(L2, L1): M} -> {L1: [(L2, M)]}
        d = {}
        for (L2, L1), M in O.items():
            d.setdefault(L1, []).append((L2, M))
        return d

    def _grow_left(self, r):                                            # A = T[r] left-canonical; makes HL, OL at bond r+1
        m = self.m; A = self.T[r]; HL = {L: np.zeros((self.D[r+1][L],)*2) for L in self.labels[r+1] if self.D[r+1][L]}
        tmp = {k: np.zeros_like(x) for k, x in A.items()}; self._left_part(r, A, tmp)     # (H_L + H_s + left couplings) A
        for (la, q, L), x in A.items():
            HL[L] += np.einsum('asb,asc->bc', x, tmp[(la, q, L)])
        for L in HL:
            HL[L] = 0.5*(HL[L] + HL[L].T)
        OL = []
        if r < self.n - 1:
            for t, (cf, dk, X, _) in enumerate(m.terms[r]):
                O = {}; Xs = self.Xsrc[r][t]
                for (la, q1, L1), x1 in A.items():
                    for q2, Xm in Xs.get(q1, ()):
                        for L2 in self.labels[r+1]:
                            k2 = (la, q2, L2)
                            if k2 in A:
                                w = t_second(la[0], q1[0], L1[0], q2[0], L2[0])
                                if w:
                                    M = w*np.einsum('asb,st,atc->bc', A[k2], Xm, x1)
                                    O[(L2, L1)] = O.get((L2, L1), 0) + M
                OL.append(O)
        self.HL[r+1] = HL; self.OL[r+1] = OL; self.OLsrc[r+1] = [self._src(O) for O in OL]

    def _grow_right(self, r):                                           # B = T[r] right-canonical; makes HR, OR at bond r
        m = self.m; B = self.T[r]; HR = {L: np.zeros((self.D[r][L],)*2) for L in self.labels[r] if self.D[r][L]}
        # (H_s + H_R + right couplings) in the (s b) coupling of the right multiplets: the same scalar coefficients with
        # the roles (a s) -> (s b): w_left(j_s, j_b, j, j_s', j_b')
        for (L, q, lb), x in B.items():
            HR[L] += np.einsum('xsb,st,ytb->xy', x, m.Hs[r][q], x) + np.einsum('xsb,bc,ysc->xy', x, self.HR[r+1][lb], x)
        if r < self.n - 1:
            for t, (cf, dk, _, _) in enumerate(m.terms[r]):
                OR = self.ORsrc[r+1][t]; X = self.Xsrc[r][t]
                for (L, q1, lb1), x1 in B.items():
                    for q2, Xm in X.get(q1, ()):
                        for lb2, O in OR.get(lb1, ()):
                            k2 = (L, q2, lb2)
                            if k2 in B:
                                w = w_left(q1[0], lb1[0], L[0], q2[0], lb2[0])
                                if w:
                                    HR[L] += (cf*w)*np.einsum('xsc,st,cd,ytd->xy', B[k2], Xm, O, x1)
        for L in HR:
            HR[L] = 0.5*(HR[L] + HR[L].T)
        ORl = []
        if r > 0:
            for t, (cf, dk, _, Y) in enumerate(m.terms[r-1]):
                O = {}; Ys = self.Ysrc[r-1][t]
                for (L1, q1, lb), x1 in B.items():
                    for q2, Ym in Ys.get(q1, ()):
                        for L2 in self.labels[r]:
                            k2 = (L2, q2, lb)
                            if k2 in B:
                                w = t_first(q1[0], lb[0], L1[0], q2[0], L2[0])
                                if w:
                                    O[(L2, L1)] = O.get((L2, L1), 0) + w*np.einsum('xsc,st,ytc->xy', B[k2], Ym, x1)
                ORl.append(O)
        self.HR[r] = HR; self.OR[r] = ORl; self.ORsrc[r] = [self._src(O) for O in ORl]

    @property
    def OLsrc(self):
        if not hasattr(self, '_OLsrc'):
            self._OLsrc = [None]*(self.n + 1); self._OLsrc[0] = []
        return self._OLsrc

    @property
    def ORsrc(self):
        if not hasattr(self, '_ORsrc'):
            self._ORsrc = [None]*(self.n + 1); self._ORsrc[self.n] = []
        return self._ORsrc

    # ---- moves
    @staticmethod
    def _keep(svals, Dmax, cutoff, weight):
        """Keep the largest singular values over all labels, counting a multiplet of spin j as one (multiplets) or
        2j + 1 states (weight = True would count states; the default counts multiplets as block2 does)."""
        allS = np.sort(np.concatenate([s for s in svals.values()] or [np.zeros(1)]))[::-1]
        thr = max(allS[Dmax] if len(allS) > Dmax else 0.0, cutoff*allS[0])
        return {L: int(np.sum(s > thr)) for L, s in svals.items()}

    def _sketch_cols(self, D, room, alpha, grow):
        if alpha <= 0 or room <= 0:
            return 0
        k = max(4, D) if grow else max(2, int(math.ceil(self.sketch*D)))
        return min(k, room)

    def move_right(self, r, Dmax, alpha, cutoff=1e-12):
        t0 = time.perf_counter(); m = self.m; C = self.T[r]; grow = sum(self.D[r+1].values()) < Dmax
        labs = [L for L in self.labels[r+1]]; rows = {L: self._rows(r, L) for L in labs}
        roff = {L: np.concatenate([[0], np.cumsum([self.D[r][la]*m.sectors[r][q] for la, q in rows[L]])]).astype(int) for L in labs}
        # expansion columns: left part (same label) and every right term with an open right leg
        cols = {L: [] for L in labs}
        if alpha > 0:
            HC = {k: np.zeros_like(x) for k, x in C.items()}; self._left_part(r, C, HC)
            for L in labs:
                if self.D[r+1][L]:
                    cols[L].append({(la, q): HC[(la, q, L)] for la, q in rows[L] if (la, q, L) in HC})
            for t, (cf, dk, X, _) in enumerate(m.terms[r]):
                Xs = self.Xsrc[r][t]; acc = {}
                for (la, q1, L1), x in C.items():
                    for q2, Xm in Xs.get(q1, ()):
                        for L2 in labs:
                            if (la, q2) in [rq for rq in rows[L2]]:
                                w = t_second(la[0], q1[0], L1[0], q2[0], L2[0])
                                if w:
                                    d = acc.setdefault((L2, L1), {}); d[(la, q2)] = d.get((la, q2), 0) + w*np.einsum('ys,asb->ayb', Xm, x)
                for (L2, L1), d in acc.items():
                    cols[L2].append(d)
        res = {}
        for L in labs:
            R = int(roff[L][-1])
            if R == 0:
                continue
            Dold = self.D[r+1][L]; A = np.zeros((R, Dold))
            for i, (la, q) in enumerate(rows[L]):
                if (la, q, L) in C:
                    A[roff[L][i]:roff[L][i+1]] = C[(la, q, L)].reshape(-1, Dold)
            kv = self._sketch_cols(Dold, R - Dold, alpha, grow)
            if kv:
                Bk = np.zeros((R, kv))
                for d in cols[L]:
                    w = next(iter(d.values())).shape[-1]; Om = self.rng.standard_normal((w, kv))
                    for i, (la, q) in enumerate(rows[L]):
                        if (la, q) in d:
                            Bk[roff[L][i]:roff[L][i+1]] += d[(la, q)].reshape(-1, w) @ Om
                Bk = Bk/math.sqrt(kv) + (self.random/math.sqrt(R))*self.rng.standard_normal((R, kv))
                A = np.concatenate([A, alpha*Bk], axis=1)
            if A.shape[1]:
                res[L] = np.linalg.svd(A, full_matrices=False)
        keep = self._keep({L: s for L, (_, s, _) in res.items()}, Dmax, cutoff, False)
        old = dict(self.D[r+1]); new = dict.fromkeys(labs, 0)
        for L, k in keep.items():
            new[L] = k
        Anew = {}; Mx = {}; kept = 0.0
        for L, (U, s, Vh) in res.items():
            k = new[L]
            if not k:
                continue
            for i, (la, q) in enumerate(rows[L]):
                Anew[(la, q, L)] = U[roff[L][i]:roff[L][i+1], :k].reshape(self.D[r][la], m.sectors[r][q], k)
            if old[L]:
                Mx[L] = s[:k, None]*Vh[:k, :old[L]]; kept += float(np.sum(Mx[L]**2))
        self.stats['truncation'] += max(0.0, 1.0 - kept)
        Bn = self.T[r+1]; self.D[r+1] = new; self.T[r] = Anew
        Cn = {}
        for (L, q, lc) in self._keys(r + 1):
            x = np.zeros((new[L], m.sectors[r+1][q], self.D[r+2][lc]))
            if L in Mx and (L, q, lc) in Bn:
                x += phi(L[0], q[0], lc[0])*np.einsum('xb,bsc->xsc', Mx[L], Bn[(L, q, lc)])
            Cn[(L, q, lc)] = x
        self.T[r+1] = Cn; self._normalize(r + 1); self.stats['t_move'] += time.perf_counter() - t0
        t0 = time.perf_counter(); self._grow_left(r); self.stats['t_env'] += time.perf_counter() - t0

    def move_left(self, r, Dmax, alpha, cutoff=1e-12):
        t0 = time.perf_counter(); m = self.m; C = self.T[r]; grow = sum(self.D[r].values()) < Dmax
        Ct = {(la, q, lb): phi(la[0], q[0], lb[0])*x for (la, q, lb), x in C.items()}   # basis |(a, (s b) j_a) 0>
        labs = list(self.labels[r]); colsk = {L: self._cols(r, L) for L in labs}
        coff = {L: np.concatenate([[0], np.cumsum([m.sectors[r][q]*self.D[r+1][lb] for q, lb in colsk[L]])]).astype(int) for L in labs}
        rowsx = {L: [] for L in labs}
        if alpha > 0:
            HC = {k: np.zeros_like(x) for k, x in C.items()}; self._right_part(r, C, HC, with_layer=True)
            for L in labs:
                if self.D[r][L]:
                    rowsx[L].append({(q, lb): phi(L[0], q[0], lb[0])*HC[(L, q, lb)] for q, lb in colsk[L] if (L, q, lb) in HC})
            if r > 0:
                for t, (cf, dk, _, Y) in enumerate(m.terms[r-1]):
                    Ys = self.Ysrc[r-1][t]; acc = {}
                    for (L1, q1, lb), x in Ct.items():
                        for q2, Ym in Ys.get(q1, ()):
                            for L2 in labs:
                                if (q2, lb) in colsk[L2]:
                                    w = t_first(q1[0], lb[0], L1[0], q2[0], L2[0])
                                    if w:
                                        d = acc.setdefault((L2, L1), {}); d[(q2, lb)] = d.get((q2, lb), 0) + w*np.einsum('ys,asb->ayb', Ym, x)
                    for (L2, L1), d in acc.items():
                        rowsx[L2].append(d)
        res = {}
        for L in labs:
            Cc = int(coff[L][-1])
            if Cc == 0:
                continue
            Dold = self.D[r][L]; A = np.zeros((Dold, Cc))
            for i, (q, lb) in enumerate(colsk[L]):
                if (L, q, lb) in Ct:
                    A[:, coff[L][i]:coff[L][i+1]] = Ct[(L, q, lb)].reshape(Dold, -1)
            kv = self._sketch_cols(Dold, Cc - Dold, alpha, grow)
            if kv:
                Bk = np.zeros((kv, Cc))
                for d in rowsx[L]:
                    h = next(iter(d.values())).shape[0]; Om = self.rng.standard_normal((kv, h))
                    for i, (q, lb) in enumerate(colsk[L]):
                        if (q, lb) in d:
                            Bk[:, coff[L][i]:coff[L][i+1]] += Om @ d[(q, lb)].reshape(h, -1)
                Bk = Bk/math.sqrt(kv) + (self.random/math.sqrt(Cc))*self.rng.standard_normal((kv, Cc))
                A = np.concatenate([A, alpha*Bk], axis=0)
            if A.shape[0]:
                res[L] = np.linalg.svd(A, full_matrices=False)
        keep = self._keep({L: s for L, (_, s, _) in res.items()}, Dmax, cutoff, False)
        old = dict(self.D[r]); new = dict.fromkeys(labs, 0)
        for L, k in keep.items():
            new[L] = k
        Bnew = {}; Mx = {}; kept = 0.0
        for L, (U, s, Vh) in res.items():
            k = new[L]
            if not k:
                continue
            for i, (q, lb) in enumerate(colsk[L]):
                Bnew[(L, q, lb)] = Vh[:k, coff[L][i]:coff[L][i+1]].reshape(k, m.sectors[r][q], self.D[r+1][lb])
            if old[L]:
                Mx[L] = U[:old[L], :k]*s[:k]; kept += float(np.sum(Mx[L]**2))
        self.stats['truncation'] += max(0.0, 1.0 - kept)
        Ap = self.T[r-1]; self.D[r] = new; self.T[r] = Bnew
        Cn = {}
        for (lx, q, L) in self._keys(r - 1):
            x = np.zeros((self.D[r-1][lx], m.sectors[r-1][q], new[L]))
            if L in Mx and (lx, q, L) in Ap:
                x += np.einsum('xsa,ab->xsb', Ap[(lx, q, L)], Mx[L])
            Cn[(lx, q, L)] = x
        self.T[r-1] = Cn; self._normalize(r - 1); self.stats['t_move'] += time.perf_counter() - t0
        t0 = time.perf_counter(); self._grow_right(r); self.stats['t_env'] += time.perf_counter() - t0

    def sweep(self, Dmax, alpha, tol=1e-6, noise=0.0):
        self.stats['truncation'] = 0.0; self.noise = noise; E = None
        for r in range(self.n - 1):
            E = self.solve(r, tol); self.move_right(r, Dmax, alpha)
        for r in range(self.n - 1, 0, -1):
            E = self.solve(r, tol); self.move_left(r, Dmax, alpha)
        self.noise = 0.0; E = self.solve(0, tol); self.stats['energies'].append(E)
        return E

    def bond_multiplets(self):
        return [sum(d.values()) for d in self.D]

    def bond_states(self):
        return [sum((L[0] + 1)*d for L, d in D.items()) for D in self.D]


def ground_state(model, D, S2=0, K=0, sweeps=None, tol=3e-5, seed=0, log=None):
    """Lowest state with total spin S2/2 and momentum K, the schedule of edc_dmrg.ground_state; D counts multiplets."""
    eng = SU2DMRG(model, S2, K, seed=seed); alphas = list(ALPHAS); t0 = time.perf_counter(); rows = []
    alphas = alphas[:sweeps] if sweeps and sweeps <= len(alphas) else alphas + [0.0]*((sweeps or 0) - len(alphas))
    sched = [(max(min(D, 64), D//4), 1e-2, tol, 1e-2), (max(min(D, 64), D//2), 1e-2, tol, 1e-2)] + [(D, al, tol if al > 0 else tol/30, 0.0) for al in alphas]
    for sw, (Dw, al, tl, noise) in enumerate(sched):
        ts = time.perf_counter(); p0 = eng.stats['products']; E = eng.sweep(Dw, al, tl, noise); te = time.perf_counter()
        rows.append(dict(sweep=sw + 1, D=Dw, expansion=al, E=E, seconds=te - ts, elapsed=te - t0, products=eng.stats['products'] - p0,
                         multiplets=max(eng.bond_multiplets()), states=max(eng.bond_states()), discarded=eng.stats['truncation']))
        if log:
            log(f'  sweep {sw+1:2d} D={Dw:5d} exp={al:.0e}: E = {E:.10f} | {te-ts:7.2f} s | products {rows[-1]["products"]:5d} | '
                f'largest bond {rows[-1]["multiplets"]} multiplets = {rows[-1]["states"]} states | discarded {rows[-1]["discarded"]:.1e}')
    return E, eng, rows


if __name__ == '__main__':
    Lx, c, D, S2, K = (int(v) for v in sys.argv[1:6]); tail = len(sys.argv) > 6 and sys.argv[6] == 'tail'
    lat = ed.kagome_cylinder(Lx, c, tail=tail); t0 = time.perf_counter(); model = SU2Model(lat)
    print(f'{lat.name}: N = {lat.N}, S = {S2/2:g}, K = {K}, D = {D} multiplets; layer multiplets {[len(L.mult) for L in model.lay[:2]]}; '
          f'model {time.perf_counter()-t0:.1f} s', flush=True)
    E, eng, rows = ground_state(model, D, S2, K, log=lambda s: print(s, flush=True))
    print(f'E = {E:.10f}')
