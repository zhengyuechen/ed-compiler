#!/usr/bin/env python3
"""DMRG in the ED compiler: the problem is stated in the EDC language, the kernels are generated for it (D28, D29).

    lattice cylinder Lx=8 Ly=6           # square lattice, open along x, periodic along y
    couplings J1=1 J2=0.5                # J2 on the diagonals (square only); Delta=0.8: XXZ anisotropy
    sector Sz=0 k=0                      # k: momentum around the cylinder in units of pi (k = 2 K / Ly, K integer)
    solve method=dmrg D=512              # sweeps=9 (after two warm-up sweeps at D/4, D/2)  labels=sz,k (default) | sz   seed=0
                                         # tol=1e-6: relative residual of the eigensolver on one layer

    lattice cylinder cell=kagome Lx=6 Ly=3   # kagome, YC geometry: Ly unit cells around (circumference 2 Ly sites),
    couplings J1=1 Delta=1                   # Lx chain rings along the cylinder; N = Ly (3 Lx - 1) spins
                                             # ends=cell: one more layer after the last ring, N = 3 Ly Lx
    sector Sz=3 k=2/3                        # any S_z (a magnetisation sector), any momentum 2 K / Ly
                                             # Sz=all: every S_z >= 0 and every momentum (magnetisation curve)
    solve method=dmrg D=1000

Layers.  The cylinder is cut into layers, each closed under the translation around it; a layer is one site of the
chain and every bond carries (number of up spins, momentum) of everything to its left.  Square lattice: every ring is a
layer.  Kagome: chain rings of 2 Ly spins alternate with layers of Ly spins (the third sublattice); every cut between
layers is a straight cut through 2 Ly bonds.  Layers with the same geometry share one "site kind", pairs of
neighbouring layers with the same bonds share one "bond kind".

What is generated.  For every site kind a real basis with the two labels (states invariant under a reflection of the
cylinder times complex conjugation) and the Hamiltonian inside the layer; for every bond kind the pairs of operators
that couple the two layers, sum_q X_q (x) Y_-q, with as few pairs as the geometry allows; all as lists of non-zero
elements in a C library (clang, bound with ctypes, as in edc.py).

One product with the effective Hamiltonian.  The tensor of a layer, theta[a, s, b], is kept in two layouts:
  M layout, one matrix per left label i:   M_i[a, (s, b)]   -- columns are slabs, one per layer state
  U layout, one matrix per right label j:  U_j[(s, a), b]   -- rows are slabs, one per layer state
An operator on the layer moves whole slabs (sparse, like a product of exact diagonalisation); an operator on a bond is
a dense matrix product.  All coupling terms of a bond are stacked, so the left half of the product is ONE matrix
product per left label, GL_i . S_i, and the right half one per right label, S_j . GR_j.  Environments are the same
stacks projected on the new basis; the subspace expansion is the same stack times a random matrix (a sketch).

Conventions.  Label code = m * mod + k.  Term t of a bond: X_t acts on the left layer and raises the label by dq_t,
Y_t on the right layer.  LX_t[j], RY_t[j]: (D_{j+dq} x D_j);  GL_i[:, block t] = cf_t LX_t[i-dq],
GR_j[block t, :] = cf_t RY_t[j-dq]^T."""
import os, sys, math, time, ctypes, hashlib
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
import edc

PI32 = ctypes.POINTER(ctypes.c_int32); PI64 = ctypes.POINTER(ctypes.c_int64)


# ------------------------------------------------------------------ lattices as stacks of layers
class Lattice:
    """c cells around the cylinder.  layers[l]: positions y (Fractions in [0, c)) of the spins of layer l.
    bonds: (layer, site, layer', site', J) with layer' = layer or layer + 1.  Reflection y -> y0 - y."""
    def __init__(self, c, layers, bonds, y0, name=''):
        self.c = c; self.layers = [[Fraction(y) % c for y in ys] for ys in layers]; self.y0 = Fraction(y0); self.name = name
        self.offset = np.concatenate([[0], np.cumsum([len(ys) for ys in self.layers])]).astype(int); self.N = int(self.offset[-1])
        self.T = []; self.P = []
        for ys in self.layers:
            pos = {y: i for i, y in enumerate(ys)}
            self.T.append([pos[(y+1) % c] for y in ys]); self.P.append([pos[(self.y0-y) % c] for y in ys])
        self.intra = [[] for _ in self.layers]; self.inter = [[] for _ in self.layers[:-1]]
        for la, a, lb, b, J in bonds:
            if la == lb:
                self.intra[la].append((min(a, b), max(a, b), float(J)))
            else:
                assert lb == la+1, 'bonds reach the next layer at most'
                self.inter[la].append((a, b, float(J)))
        for l in range(len(self.layers)):                              # translation and reflection must map bonds to bonds
            for perm in (self.T[l], self.P[l]):
                S0 = sorted(self.intra[l]); S1 = sorted((min(perm[a], perm[b]), max(perm[a], perm[b]), J) for a, b, J in self.intra[l])
                assert S0 == S1, 'the layer is not symmetric under translation / reflection'
            if l+1 < len(self.layers):
                for pa, pb in ((self.T[l], self.T[l+1]), (self.P[l], self.P[l+1])):
                    assert sorted(self.inter[l]) == sorted((pa[a], pb[b], J) for a, b, J in self.inter[l]), 'bonds between layers not symmetric'

    def global_bonds(self):
        out = []
        for l, bl in enumerate(self.intra):
            out += [(int(self.offset[l]+a), int(self.offset[l]+b), J) for a, b, J in bl]
        for l, bl in enumerate(self.inter):
            out += [(int(self.offset[l]+a), int(self.offset[l+1]+b), J) for a, b, J in bl]
        return out

    def translation(self):
        return [int(self.offset[l]+t) for l in range(len(self.layers)) for t in self.T[l]]


def square_cylinder(Lx, Ly, J1=1.0, J2=0.0):
    layers = [[Fraction(y) for y in range(Ly)] for _ in range(Lx)]; bonds = []
    for x in range(Lx):
        for y in range(Ly):
            if Ly > 2 or y == 0:
                bonds.append((x, y, x, (y+1) % Ly, J1))
            if x+1 < Lx:
                bonds.append((x, y, x+1, y, J1))
                if J2:
                    bonds.append((x, y, x+1, (y+1) % Ly, J2)); bonds.append((x, y, x+1, (y-1) % Ly, J2))
    return Lattice(Ly, layers, bonds, 0, f'square cylinder {Lx} x {Ly}')


def kagome_cylinder(Lx, c, J1=1.0, tail=False):
    """YC kagome cylinder with c >= 2 unit cells around: Lx chain rings of 2c spins at y = n/2, and between two rings
    a layer of c spins at y = i + 1/4 (after even rings) or i + 3/4 (after odd rings), each bonded to the ring spins at
    y -+ 1/4 on both sides.  N = c (3 Lx - 1).  tail=True adds one more layer of c spins after the last ring, so that
    the cylinder holds Lx whole unit-cell columns: N = 3 c Lx (the plateau fractions 1/9, 1/3, ... then exist)."""
    assert c >= 2; layers = []; bonds = []
    for j in range(Lx):
        lr = len(layers); layers.append([Fraction(n, 2) for n in range(2*c)])
        bonds += [(lr, n, lr, (n+1) % (2*c), J1) for n in range(2*c)]
        if j+1 < Lx or tail:
            off = Fraction(1, 4) if j % 2 == 0 else Fraction(3, 4); layers.append([i+off for i in range(c)])
            for i in range(c):
                for dy in (Fraction(-1, 4), Fraction(1, 4)):
                    n = int(((i+off+dy) % c)*2); bonds.append((lr, n, lr+1, i, J1))
                    if j+1 < Lx:
                        bonds.append((lr+1, i, lr+2, n, J1))
    return Lattice(c, layers, bonds, Fraction(1, 2), f'kagome cylinder YC{2*c}, {Lx} rings' + (' and a last layer' if tail else ''))


def kagome_cylinder_cells(Lx, c, J1=1.0):
    """The same lattice written from the unit cell (three spins per cell, six bonds per cell), as a plain bond list.
    Used only to check kagome_cylinder: numbering s0(i,j) = 3c j + 2i, s1 = 3c j + 2i + 1, s2 = 3c j + 2c + i."""
    s0 = lambda i, j: 3*c*j + 2*(i % c); s1 = lambda i, j: 3*c*j + 2*(i % c) + 1; s2 = lambda i, j: 3*c*j + 2*c + (i % c); b = []
    for j in range(Lx):
        for i in range(c):
            b += [(s0(i, j), s1(i, j)), (s1(i, j), s0(i+1, j))]
            if j+1 < Lx:
                b += [(s0(i, j), s2(i, j)), (s1(i, j), s2(i, j)), (s2(i, j), s0(i, j+1)), (s2(i, j), s1(i-1, j+1))]
    return [(min(a, d), max(a, d), J1) for a, d in b]


def exact_levels(bonds, N, nup, Delta=1.0, tperm=None, c=1, tol=1e-11):
    """Lowest level of an XXZ model on a bond list at fixed number of up spins; with tperm (site -> site, a translation
    of order c) the lowest level of every momentum sector.  Product basis; a reference for tests, small N only."""
    import itertools, scipy.sparse.linalg as sla
    st = np.array([sum(1 << b for b in cb) for cb in itertools.combinations(range(N), nup)], dtype=np.int64); st.sort(); n = len(st)
    diag = np.zeros(n); rows, cols, vals = [], [], []
    for i, j, J in bonds:
        bi, bj = (st >> i) & 1, (st >> j) & 1; diag += J*Delta*np.where(bi == bj, 0.25, -0.25)
        sel = np.nonzero(bi != bj)[0]; rows.append(np.searchsorted(st, st[sel] ^ ((1 << i) | (1 << j)))); cols.append(sel); vals.append(np.full(len(sel), 0.5*J))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) + sp.diags(diag)
    if tperm is None:
        return float(sla.eigsh(H, k=1, which='SA', tol=tol, return_eigenvectors=False)[0]) if n > 600 else float(np.linalg.eigvalsh(H.toarray())[0])
    t = np.zeros(n, dtype=np.int64)
    for a in range(N):
        t |= ((st >> a) & 1) << tperm[a]
    perm = np.searchsorted(st, t); out = {}
    for k in range(c):
        def proj(v):
            acc = v.astype(complex); w = v
            for s in range(1, c):
                u = np.empty_like(w); u[perm] = w; w = u; acc = acc + np.exp(2j*np.pi*k*s/c)*w
            return acc/c
        v0 = proj(np.random.default_rng(k).standard_normal(n).astype(complex))
        if np.linalg.norm(v0) < 1e-12:
            continue
        if n <= 600:
            Hd = H.toarray(); Pm = np.array([proj(e) for e in np.eye(n, dtype=complex)]).T; w, V = np.linalg.eigh(Pm)
            V = V[:, w > 0.5]
            if V.shape[1]:
                out[k] = float(np.linalg.eigvalsh(V.conj().T @ Hd @ V)[0])
        else:
            op = sla.LinearOperator((n, n), matvec=lambda v: proj(H @ proj(v)), dtype=complex)
            out[k] = float(sla.eigsh(op, k=1, which='SA', v0=v0, tol=tol, return_eigenvectors=False)[0])
    return out


# ------------------------------------------------------------------ site kinds and bond kinds
class SiteKind:
    """One kind of layer: n spins, the translation and the reflection as site permutations, its bonds.  Gives the real
    basis with labels (m, k) and operators of the layer as lists of non-zeros in that basis."""
    def __init__(self, ys, c, Tp, Pp, intra, Delta, mod):
        self.n = n = len(ys); self.S = d = 1 << n; self.c = c; codes = np.arange(d, dtype=np.int64)
        assert n <= 10, 'layers of more than 10 spins need a sparse construction of the layer basis (not written)'
        self.cell = [int(math.floor(y)) for y in ys]; self.sub = [y - math.floor(y) for y in ys]

        def lift(perm):
            out = np.zeros(d, dtype=np.int64)
            for a in range(n):
                out |= ((codes >> a) & 1) << perm[a]
            return out
        tmap, refl = lift(Tp), lift(Pp); seen = np.zeros(d, bool); cols = []; ch = []
        for s in range(d):
            if seen[s]:
                continue
            orb = [s]; x = int(tmap[s])
            while x != s:
                orb.append(x); x = int(tmap[x])
            seen[orb] = True; p = len(orb); m = bin(s).count('1')
            for k in range(c):
                if (k*p) % c:
                    continue
                v = np.zeros(d, complex); v[orb] = np.exp(2j*np.pi*k*np.arange(p)/c)/math.sqrt(p)     # T v = exp(-2 pi i k / c) v
                cols.append(v); ch.append((m, k))
        order = sorted(range(d), key=lambda i: ch[i]); U = np.array([cols[i] for i in order]).T; ch = [ch[i] for i in order]
        Ur = np.zeros_like(U); rng = np.random.default_rng(1); self.blocks = {}
        for i, q in enumerate(ch):
            self.blocks.setdefault(q, []).append(i)
        for q, idx in self.blocks.items():                            # every block invariant under reflection x conjugation
            Us = U[:, idx]; PU = np.zeros_like(Us); PU[refl] = Us.conj(); M = Us.conj().T @ PU
            w, O = np.linalg.eigh(M.real + rng.uniform(0.3, 0.7)*M.imag)
            Ur[:, idx] = Us @ (O*np.exp(0.5j*np.angle(np.einsum('ia,ij,ja->a', O, M, O))))
        self.blocks = {q: np.array(v) for q, v in self.blocks.items()}; self.Ur = Ur
        self.sm = np.array([m for m, k in ch], dtype=np.int64); self.sk = np.array([k % mod for m, k in ch], dtype=np.int64)
        self.sectors = sorted(set(zip(self.sm.tolist(), self.sk.tolist())))
        up = [((codes >> a) & 1).astype(bool) for a in range(n)]
        self.sz = [sp.diags(np.where(u, 0.5, -0.5)).tocsr() for u in up]
        self.sp = [sp.csr_matrix((np.ones(int((~u).sum())), (codes[~u] | (1 << a), codes[~u])), shape=(d, d)) for a, u in enumerate(up)]
        self.sm_ = [m.T.tocsr() for m in self.sp]
        H = sp.csr_matrix((d, d))
        for a, b, J in intra:
            H = H + J*(Delta*self.sz[a] @ self.sz[b] + 0.5*(self.sp[a] @ self.sm_[b] + self.sm_[a] @ self.sp[b]))
        o, i, v = self.transform(H, 0, 0)
        assert len(v) == 0 or np.abs(v.imag).max() < 1e-9, 'layer Hamiltonian not real'
        self.H = (o, i, v.real.copy())

    def op(self, comp, a):
        return {'z': self.sz, '+': self.sp, '-': self.sm_}[comp][a]

    def transform(self, O, dm, dq):
        """Product-basis operator -> non-zeros (out, in, value) in the layer basis; it must shift the labels by (dm, dq)."""
        outs, ins, vals = [], [], []
        for (m, k), idx in self.blocks.items():
            tgt = self.blocks.get((m+dm, (k+dq) % self.c)); W = O @ self.Ur[:, idx]
            if tgt is None:
                assert (not W.size) or np.abs(W).max() < 1e-10, 'operator leaves the labelled space'
                continue
            B = self.Ur[:, tgt].conj().T @ W
            assert abs(np.linalg.norm(B)-np.linalg.norm(W)) < 1e-9*max(1.0, np.linalg.norm(W)), 'operator breaks the selection rule'
            r, cc = np.nonzero(np.abs(B) > 1e-12); outs.append(tgt[r]); ins.append(idx[cc]); vals.append(B[r, cc])
        if not outs:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0, complex)
        o, i, v = np.concatenate(outs), np.concatenate(ins), np.concatenate(vals); s = np.lexsort((i, o))
        return o[s], i[s], v[s]


def _realify(v):
    ph = np.exp(1j*np.angle(v[np.argmax(np.abs(v))])); r = v/ph
    assert np.abs(r.imag).max() < 1e-9, 'operator is not real up to a phase in the reflection-invariant basis'
    return r.real.copy(), ph


class BondKind:
    """Coupling of two neighbouring layers as sum_t cf_t X_t (x) Y_t, each term with definite label shifts."""
    def __init__(self, KL, KR, inter, c, Delta, mod):
        subsL, subsR = sorted(set(KL.sub)), sorted(set(KR.sub)); right = len(subsR) <= len(subsL)
        self.cf = []; self.dm = []; self.dk = []; self.X = []; self.Y = []
        for comp, cbar, cf, dm in (('z', 'z', Delta, 0), ('+', '-', 0.5, 1), ('-', '+', 0.5, -1)):
            if cf == 0:
                continue
            for q in range(c):
                for sub in (subsR if right else subsL):
                    X = sp.csr_matrix((KL.S, KL.S), dtype=complex); Y = sp.csr_matrix((KR.S, KR.S), dtype=complex)
                    if right:                                          # spins b of one sublattice of the right layer
                        for b in range(KR.n):
                            if KR.sub[b] != sub:
                                continue
                            f = np.exp(2j*np.pi*q*KR.cell[b]/c)/math.sqrt(c); Y = Y + np.conj(f)*KR.op(cbar, b)
                            for a, b2, J in inter:
                                if b2 == b:
                                    X = X + (f*J)*KL.op(comp, a)
                    else:
                        for a in range(KL.n):
                            if KL.sub[a] != sub:
                                continue
                            f = np.exp(2j*np.pi*q*KL.cell[a]/c)/math.sqrt(c); X = X + f*KL.op(comp, a)
                            for a2, b, J in inter:
                                if a2 == a:
                                    Y = Y + (np.conj(f)*J)*KR.op(cbar, b)
                    xo, xi, xv = KL.transform(X, dm, q); yo, yi, yv = KR.transform(Y, -dm, -q)
                    if len(xv) == 0 or len(yv) == 0:
                        continue
                    xr, px = _realify(xv); yr, py = _realify(yv); f = cf*px*py
                    assert abs(f.imag) < 1e-9, 'coupling not real'
                    self.cf.append(float(f.real)); self.dm.append(dm); self.dk.append(q % mod); self.X.append((xo, xi, xr)); self.Y.append((yo, yi, yr))
        self.NT = len(self.cf); self.dm = np.array(self.dm, dtype=np.int64); self.dk = np.array(self.dk, dtype=np.int64)


class Model:
    def __init__(self, lat, Delta=1.0, momentum=True):
        self.lat = lat; self.c = c = lat.c; self.mod = c if momentum else 1; self.Delta = Delta; self.n = len(lat.layers)
        self.kinds = []; self.bkinds = []; self.site_kind = []; self.bond_kind = []; sk = {}; bk = {}
        for l, ys in enumerate(lat.layers):
            key = (tuple(ys), tuple(sorted(lat.intra[l])))
            if key not in sk:
                sk[key] = len(self.kinds); self.kinds.append(SiteKind(ys, c, lat.T[l], lat.P[l], lat.intra[l], Delta, self.mod))
            self.site_kind.append(sk[key])
        for l in range(self.n-1):
            key = (self.site_kind[l], self.site_kind[l+1], tuple(sorted(lat.inter[l])))
            if key not in bk:
                bk[key] = len(self.bkinds)
                self.bkinds.append(BondKind(self.kinds[key[0]], self.kinds[key[1]], lat.inter[l], c, Delta, self.mod))
            self.bond_kind.append(bk[key])


# ------------------------------------------------------------------ the generated library
C_BODY = r'''
typedef struct { int nL, nR, S, sk, bl, br, NTL, NTR; const int *DL, *DR, *cst, *rst, *cw, *rh, *ul, *C, *R; const int64_t *Moff, *Uoff;
                 const int *srcL, *wstL, *WL, *srcR, *wstR, *WR; const int64_t *LHoff, *GLoff, *RHoff, *GRoff; } plan_t;
typedef void (*dgemm_t)(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*,
                        const double*, const int*, const double*, double*, const int*);
static dgemm_t dgemm_f; static double n_gemm = 0.0, n_slab = 0.0;
void set_blas(void *p) { dgemm_f = (dgemm_t)p; }
void set_threads(int n) { omp_set_num_threads(n); }
void counts(double *c, int reset) { c[0] = n_gemm; c[1] = n_slab; if (reset) { n_gemm = 0.0; n_slab = 0.0; } }

/* row-major C (m x n) = alpha op(A) op(B) + beta C, through the column-major routine: C^T = op(B)^T op(A)^T */
static void gemm(int ta, int tb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb,
                 double beta, double *C, int ldc) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0) { if (beta == 0.0) for (int i = 0; i < m; i++) memset(C + (int64_t)i*ldc, 0, sizeof(double)*n); return; }
  #pragma omp atomic
  n_gemm += (double)m*n*k;
  dgemm_f(tb ? "T" : "N", ta ? "T" : "N", &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

/* S^L of target left label it: (WL x C_it); block t = M_src K_t (the operator Y_t of the left bond moves column slabs) */
static void sl_build(const plan_t *p, int it, const double *M, double *Sb, int scale) {
  int S = p->S, NT = p->NTL, Ct = p->C[it], W = p->WL[it];
  const int *YP = YPb[p->bl], *YO = YOb[p->bl], *YI = YIb[p->bl]; const double *YV = YVb[p->bl], *CF = CFb[p->bl];
  memset(Sb, 0, sizeof(double)*(size_t)W*Ct);
  for (int t = 0; t < NT; t++) {
    int is = p->srcL[it*NT+t]; if (is < 0) continue;
    int Dis = p->DL[is]; if (!Dis) continue;
    int w0 = p->wstL[it*NT+t], Cs = p->C[is]; const double *Ms = M + p->Moff[is]; double f = scale ? CF[t] : 1.0;
    for (int e = YP[t]; e < YP[t+1]; e++) {
      int cs = p->cst[is*S+YI[e]], ct = p->cst[it*S+YO[e]];
      if (cs < 0 || ct < 0) continue;
      int w = p->cw[is*S+YI[e]]; double v = f*YV[e];
      for (int a = 0; a < Dis; a++) {
        double *d = Sb + (int64_t)(w0+a)*Ct + ct; const double *s = Ms + (int64_t)a*Cs + cs;
        for (int b = 0; b < w; b++) d[b] += v*s[b];
      }
      #pragma omp atomic
      n_slab += (double)Dis*w;
    }
  }
}

/* S^R of target right label jt: (R_jt x WR); block t = K_t U_src (the operator X_t of the right bond moves row slabs) */
static void sr_build(const plan_t *p, int jt, const double *U, double *Sb, int scale) {
  int S = p->S, NT = p->NTR, Rt = p->R[jt], W = p->WR[jt];
  const int *XP = XPb[p->br], *XO = XOb[p->br], *XI = XIb[p->br]; const double *XV = XVb[p->br], *CF = CFb[p->br];
  memset(Sb, 0, sizeof(double)*(size_t)Rt*W);
  for (int t = 0; t < NT; t++) {
    int js = p->srcR[jt*NT+t]; if (js < 0) continue;
    int Djs = p->DR[js]; if (!Djs) continue;
    int w0 = p->wstR[jt*NT+t]; const double *Us = U + p->Uoff[js]; double f = scale ? CF[t] : 1.0;
    for (int e = XP[t]; e < XP[t+1]; e++) {
      int rs = p->rst[js*S+XI[e]], rt = p->rst[jt*S+XO[e]];
      if (rs < 0 || rt < 0) continue;
      int h = p->rh[js*S+XI[e]]; double v = f*XV[e];
      for (int a = 0; a < h; a++) {
        double *d = Sb + (int64_t)(rt+a)*W + w0; const double *s = Us + (int64_t)(rs+a)*Djs;
        for (int b = 0; b < Djs; b++) d[b] += v*s[b];
      }
      #pragma omp atomic
      n_slab += (double)h*Djs;
    }
  }
}

static size_t max_sl(const plan_t *p) { size_t m = 1; for (int i = 0; i < p->nL; i++) { size_t x = (size_t)p->WL[i]*p->C[i]; if (x > m) m = x; } return m; }
static size_t max_sr(const plan_t *p) { size_t m = 1; for (int j = 0; j < p->nR; j++) { size_t x = (size_t)p->R[j]*p->WR[j]; if (x > m) m = x; } return m; }

void m2u(const plan_t *p, const double *M, double *U) {
  int S = p->S;
  for (int j = 0; j < p->nR; j++) { int Dj = p->DR[j]; if (!Dj) continue;
    for (int s = 0; s < S; s++) { int r0 = p->rst[j*S+s]; if (r0 < 0) continue;
      int i = p->ul[j*S+s], Di = p->DL[i], c0 = p->cst[i*S+s];
      for (int a = 0; a < Di; a++) memcpy(U + p->Uoff[j] + (int64_t)(r0+a)*Dj, M + p->Moff[i] + (int64_t)a*p->C[i] + c0, sizeof(double)*Dj); } }
}
void u2m(const plan_t *p, const double *U, double *M, int add) {
  int S = p->S;
  for (int j = 0; j < p->nR; j++) { int Dj = p->DR[j]; if (!Dj) continue;
    for (int s = 0; s < S; s++) { int r0 = p->rst[j*S+s]; if (r0 < 0) continue;
      int i = p->ul[j*S+s], Di = p->DL[i], c0 = p->cst[i*S+s];
      for (int a = 0; a < Di; a++) { const double *u = U + p->Uoff[j] + (int64_t)(r0+a)*Dj; double *m = M + p->Moff[i] + (int64_t)a*p->C[i] + c0;
        if (add) for (int b = 0; b < Dj; b++) m[b] += u[b]; else memcpy(m, u, sizeof(double)*Dj); } } }
}

/* out = (H of the left part) M + sum_t cf_t (LX_t x Y_t) M, in the M layout */
void left_half(const plan_t *p, const double *LH, const double *GL, const double *M, double *out) {
  size_t msz = max_sl(p);
  #pragma omp parallel
  { double *Sb = (double*)malloc(sizeof(double)*msz);
  #pragma omp for schedule(dynamic)
  for (int i = 0; i < p->nL; i++) { int Di = p->DL[i], Ci = p->C[i], W = p->WL[i]; if (!Di || !Ci) continue;
    gemm(0, 0, Di, Ci, Di, 1.0, LH + p->LHoff[i], Di, M + p->Moff[i], Ci, 0.0, out + p->Moff[i], Ci);
    if (W) { sl_build(p, i, M, Sb, 0); gemm(0, 0, Di, Ci, W, 1.0, GL + p->GLoff[i], W, Sb, Ci, 1.0, out + p->Moff[i], Ci); } }
  free(Sb); }
}
/* out = U (H of the right part) + sum_t cf_t (X_t x RY_t) U, in the U layout */
void right_half(const plan_t *p, const double *RH, const double *GR, const double *U, double *out) {
  size_t msz = max_sr(p);
  #pragma omp parallel
  { double *Sb = (double*)malloc(sizeof(double)*msz);
  #pragma omp for schedule(dynamic)
  for (int j = 0; j < p->nR; j++) { int Dj = p->DR[j], Rj = p->R[j], W = p->WR[j]; if (!Dj || !Rj) continue;
    gemm(0, 0, Rj, Dj, Dj, 1.0, U + p->Uoff[j], Dj, RH + p->RHoff[j], Dj, 0.0, out + p->Uoff[j], Dj);
    if (W) { sr_build(p, j, U, Sb, 0); gemm(0, 0, Rj, Dj, W, 1.0, Sb, W, GR + p->GRoff[j], Dj, 1.0, out + p->Uoff[j], Dj); } }
  free(Sb); }
}
/* out += (Hamiltonian inside the layer) U, in the U layout */
void hr_u(const plan_t *p, const double *U, double *out) {
  int S = p->S, NH = NHk[p->sk]; const int *HO = HOk[p->sk], *HI = HIk[p->sk]; const double *HV = HVk[p->sk];
  #pragma omp parallel for schedule(dynamic)
  for (int j = 0; j < p->nR; j++) { int Dj = p->DR[j]; if (!Dj) continue;
    for (int e = 0; e < NH; e++) { int rs = p->rst[j*S+HI[e]], rt = p->rst[j*S+HO[e]]; if (rs < 0 || rt < 0) continue;
      int64_t n = (int64_t)p->rh[j*S+HI[e]]*Dj; const double *s = U + p->Uoff[j] + (int64_t)rs*Dj; double *d = out + p->Uoff[j] + (int64_t)rt*Dj, v = HV[e];
      for (int64_t x = 0; x < n; x++) d[x] += v*s[x];
      #pragma omp atomic
      n_slab += (double)n; } }
}
void heff(const plan_t *p, const double *LH, const double *GL, const double *RH, const double *GR, const double *M, double *out,
          double *U, double *oU) {
  left_half(p, LH, GL, M, out); m2u(p, M, U); right_half(p, RH, GR, U, oU); hr_u(p, U, oU); u2m(p, oU, out, 1);
}
/* out_i (w_i x C_i) = Mat_i (w_i x WL_i) S^L_i  -- with random Mat: the sketch of the expansion for a move to the left */
void sl_apply(const plan_t *p, const double *Mat, const int64_t *moff, const int *wv, const double *M, double *out, const int64_t *ooff) {
  size_t msz = max_sl(p);
  #pragma omp parallel
  { double *Sb = (double*)malloc(sizeof(double)*msz);
  #pragma omp for schedule(dynamic)
  for (int i = 0; i < p->nL; i++) { int w = wv[i], Ci = p->C[i], W = p->WL[i]; if (!w || !Ci) continue;
    if (W) sl_build(p, i, M, Sb, 0);
    gemm(0, 0, w, Ci, W, 1.0, Mat + moff[i], W, Sb, Ci, 0.0, out + ooff[i], Ci); }
  free(Sb); }
}
/* out_j (R_j x w_j) = S^R_j Mat_j (WR_j x w_j) */
void sr_apply(const plan_t *p, const double *Mat, const int64_t *moff, const int *wv, const double *U, double *out, const int64_t *ooff) {
  size_t msz = max_sr(p);
  #pragma omp parallel
  { double *Sb = (double*)malloc(sizeof(double)*msz);
  #pragma omp for schedule(dynamic)
  for (int j = 0; j < p->nR; j++) { int w = wv[j], Rj = p->R[j], W = p->WR[j]; if (!w || !Rj) continue;
    if (W) sr_build(p, j, U, Sb, 0);
    gemm(0, 0, Rj, w, W, 1.0, Sb, W, Mat + moff[j], w, 0.0, out + ooff[j], w); }
  free(Sb); }
}
/* new right environment on the left bond: GR_i (WL_i x D_i) = cf S^L_i M_i^T; RH_i = phi_i M_i^T */
void grow_right(const plan_t *p, const double *M, const double *phi, double *GR, const int64_t *goff, double *RH, const int64_t *hoff) {
  size_t msz = max_sl(p);
  #pragma omp parallel
  { double *Sb = (double*)malloc(sizeof(double)*msz);
  #pragma omp for schedule(dynamic)
  for (int i = 0; i < p->nL; i++) { int Di = p->DL[i], Ci = p->C[i], W = p->WL[i]; if (!Di) continue;
    double *H = RH + hoff[i];
    gemm(0, 1, Di, Di, Ci, 1.0, phi + p->Moff[i], Ci, M + p->Moff[i], Ci, 0.0, H, Di);
    for (int a = 0; a < Di; a++) for (int b = 0; b < a; b++) { double x = 0.5*(H[a*Di+b] + H[b*Di+a]); H[a*Di+b] = x; H[b*Di+a] = x; }
    if (!W) continue;
    if (Ci) sl_build(p, i, M, Sb, 1);
    gemm(0, 1, W, Di, Ci, 1.0, Sb, Ci, M + p->Moff[i], Ci, 0.0, GR + goff[i], Di); }
  free(Sb); }
}
/* new left environment on the right bond: GL_j (D_j x WR_j) = cf U_j^T S^R_j; LH_j = U_j^T phi_j */
void grow_left(const plan_t *p, const double *U, const double *phi, double *GL, const int64_t *goff, double *LH, const int64_t *hoff) {
  size_t msz = max_sr(p);
  #pragma omp parallel
  { double *Sb = (double*)malloc(sizeof(double)*msz);
  #pragma omp for schedule(dynamic)
  for (int j = 0; j < p->nR; j++) { int Dj = p->DR[j], Rj = p->R[j], W = p->WR[j]; if (!Dj) continue;
    double *H = LH + hoff[j];
    gemm(1, 0, Dj, Dj, Rj, 1.0, U + p->Uoff[j], Dj, phi + p->Uoff[j], Dj, 0.0, H, Dj);
    for (int a = 0; a < Dj; a++) for (int b = 0; b < a; b++) { double x = 0.5*(H[a*Dj+b] + H[b*Dj+a]); H[a*Dj+b] = x; H[b*Dj+a] = x; }
    if (!W) continue;
    if (Rj) sr_build(p, j, U, Sb, 1);
    gemm(1, 0, Dj, W, Rj, 1.0, U + p->Uoff[j], Dj, Sb, W, 0.0, GL + goff[j], W); }
  free(Sb); }
}
'''


def emit_c(model):
    fd = lambda v: repr(float(v)); fi = lambda v: str(int(v))

    def table(name, ctype, vals, fmt):
        vals = list(vals)
        return f'static const {ctype} {name}[] = {{' + ','.join(fmt(v) for v in (vals or [0])) + '};\n'
    o = ['#include <stdlib.h>\n#include <string.h>\n#include <stdint.h>\n#include <omp.h>\n']
    for s, kd in enumerate(model.kinds):
        o += [table(f'H{s}O', 'int', kd.H[0], fi), table(f'H{s}I', 'int', kd.H[1], fi), table(f'H{s}V', 'double', kd.H[2], fd)]
    nk = range(len(model.kinds)); nb = range(len(model.bkinds))
    o.append(table('NHk', 'int', [len(kd.H[0]) for kd in model.kinds], fi))
    for nm, ct in (('O', 'int'), ('I', 'int'), ('V', 'double')):
        o.append(f'static const {ct} *H{nm}k[] = {{' + ','.join(f'H{s}{nm}' for s in nk) + '};\n')
    cat = lambda ops, i: np.concatenate([a[i] for a in ops]) if ops else []
    for b, bk in enumerate(model.bkinds):
        for nm, ops in (('Y', bk.Y), ('X', bk.X)):
            ptr = np.concatenate([[0], np.cumsum([len(a[0]) for a in ops])])
            o += [table(f'{nm}{b}P', 'int', ptr, fi), table(f'{nm}{b}O', 'int', cat(ops, 0), fi),
                  table(f'{nm}{b}I', 'int', cat(ops, 1), fi), table(f'{nm}{b}V', 'double', cat(ops, 2), fd)]
        o.append(table(f'CF{b}', 'double', bk.cf, fd))
    for nm in ('Y', 'X'):
        for suf, ct in (('P', 'int'), ('O', 'int'), ('I', 'int'), ('V', 'double')):
            o.append(f'static const {ct} *{nm}{suf}b[] = {{' + (','.join(f'{nm}{b}{suf}' for b in nb) or '0') + '};\n')
    o.append('static const double *CFb[] = {' + (','.join(f'CF{b}' for b in nb) or '0') + '};\n')
    return ''.join(o) + C_BODY


class PlanC(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ('nL', 'nR', 'S', 'sk', 'bl', 'br', 'NTL', 'NTR')] + \
               [(n, PI32) for n in ('DL', 'DR', 'cst', 'rst', 'cw', 'rh', 'ul', 'C', 'R')] + [('Moff', PI64), ('Uoff', PI64)] + \
               [(n, PI32) for n in ('srcL', 'wstL', 'WL', 'srcR', 'wstR', 'WR')] + [(n, PI64) for n in ('LHoff', 'GLoff', 'RHoff', 'GRoff')]


def _blas_pointer():
    import scipy.linalg.cython_blas as cb
    cap = cb.__pyx_capi__['dgemm']; api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p; api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p; api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))


_LIBS = {}


def library(model):
    src = emit_c(model); key = hashlib.sha1(src.encode()).hexdigest()
    if key not in _LIBS:
        lib, dt, so = edc.compile_source(src, 'dmrg')
        V = ctypes.c_void_p; PP = ctypes.POINTER(PlanC)
        for name, args in (('set_blas', [V]), ('set_threads', [ctypes.c_int]), ('counts', [V, ctypes.c_int]), ('m2u', [PP, V, V]),
                           ('u2m', [PP, V, V, ctypes.c_int]), ('left_half', [PP, V, V, V, V]), ('right_half', [PP, V, V, V, V]), ('hr_u', [PP, V, V]),
                           ('heff', [PP]+[V]*8), ('sl_apply', [PP]+[V]*6), ('sr_apply', [PP]+[V]*6),
                           ('grow_right', [PP]+[V]*6), ('grow_left', [PP]+[V]*6)):
            getattr(lib, name).argtypes = args; getattr(lib, name).restype = None
        lib.set_blas(_blas_pointer()); lib.set_threads(int(os.environ.get('EDC_DMRG_THREADS', 1))); _LIBS[key] = (lib, dt)
    return _LIBS[key]


# ------------------------------------------------------------------ bonds and the layout of one layer tensor
class Bond:
    def __init__(self, labels, mod, mmax):
        lab = sorted(labels); self.n = len(lab); self.mod = mod; self.mmax = mmax
        self.m = np.array([q[0] for q in lab], dtype=np.int64); self.k = np.array([q[1] for q in lab], dtype=np.int64)
        self.table = np.full((mmax+1)*mod, -1, dtype=np.int32); self.table[self.m*mod+self.k] = np.arange(self.n)
        self.dims = np.zeros(self.n, dtype=np.int32)

    def lookup(self, m, k):
        ok = (m >= 0) & (m <= self.mmax)
        return np.where(ok, self.table[np.where(ok, m, 0)*self.mod+k], -1).astype(np.int32)

    def stack(self, bk):
        """For every label of this bond and every term of its bond kind: source label, start of its block, total."""
        if bk is None or bk.NT == 0:
            return np.full((self.n, 1), -1, dtype=np.int32), np.zeros((self.n, 1), dtype=np.int32), np.zeros(self.n, dtype=np.int32)
        src = self.lookup(self.m[:, None]-bk.dm[None, :], (self.k[:, None]-bk.dk[None, :]) % self.mod)
        d = np.where(src >= 0, self.dims[np.maximum(src, 0)], 0).astype(np.int32)
        return np.ascontiguousarray(src), np.ascontiguousarray(np.cumsum(d, axis=1)-d, dtype=np.int32), d.sum(1).astype(np.int32)


def _off(sizes):
    return np.concatenate([[0], np.cumsum(sizes.astype(np.int64))]).astype(np.int64)


class Plan:
    """Every offset of one layer tensor and its environments, from the sector sizes of its two bonds (no loops)."""
    def __init__(self, model, r, bl, br):
        kd = model.kinds[model.site_kind[r]]; mod = model.mod
        ibl = model.bond_kind[r-1] if r > 0 else -1; ibr = model.bond_kind[r] if r < model.n-1 else -1
        mr = br.lookup(bl.m[:, None]+kd.sm[None, :], (bl.k[:, None]+kd.sk[None, :]) % mod)
        cw = np.where(mr >= 0, br.dims[np.maximum(mr, 0)], 0).astype(np.int32); cst = (np.cumsum(cw, axis=1)-cw).astype(np.int32); cst[cw == 0] = -1
        ul = bl.lookup(br.m[:, None]-kd.sm[None, :], (br.k[:, None]-kd.sk[None, :]) % mod)
        rh = np.where(ul >= 0, bl.dims[np.maximum(ul, 0)], 0).astype(np.int32); rst = (np.cumsum(rh, axis=1)-rh).astype(np.int32); rst[rh == 0] = -1
        self.DL = bl.dims.copy(); self.DR = br.dims.copy(); self.C = cw.sum(1).astype(np.int32); self.R = rh.sum(1).astype(np.int32)
        self.cst, self.rst, self.cw, self.rh, self.ul = (np.ascontiguousarray(a, dtype=np.int32) for a in (cst, rst, cw, rh, ul))
        self.Moff = _off(self.DL.astype(np.int64)*self.C); self.Uoff = _off(self.R.astype(np.int64)*self.DR)
        self.srcL, self.wstL, self.WL = bl.stack(model.bkinds[ibl] if ibl >= 0 else None)
        self.srcR, self.wstR, self.WR = br.stack(model.bkinds[ibr] if ibr >= 0 else None)
        i64 = lambda a: a.astype(np.int64)
        self.LHoff = _off(i64(self.DL)**2); self.GLoff = _off(i64(self.DL)*self.WL)
        self.RHoff = _off(i64(self.DR)**2); self.GRoff = _off(i64(self.WR)*self.DR)
        self.size = int(self.Moff[-1]); assert self.size == int(self.Uoff[-1])
        self.nL, self.nR = bl.n, br.n
        c = PlanC(); c.nL, c.nR, c.S, c.sk, c.bl, c.br = bl.n, br.n, kd.S, model.site_kind[r], ibl, ibr
        c.NTL = model.bkinds[ibl].NT if ibl >= 0 else 0; c.NTR = model.bkinds[ibr].NT if ibr >= 0 else 0
        for name, typ in PlanC._fields_[8:]:
            setattr(c, name, getattr(self, name).ctypes.data_as(typ))
        self.c = c; self.ref = ctypes.byref(c)


def _p(a):
    return a.ctypes.data


class DMRG:
    def __init__(self, model, nup, K=0, D0=2, seed=0):
        self.model = model; self.n = n = model.n; self.lib, self.compile_time = library(model); self.rng = rng = np.random.default_rng(seed)
        self.sketch = 0.2                                              # sketch columns per kept state once the bond is full
        self.noise = 0.0                                               # random part added to the start of every layer solve
        self.random = 0.03                                             # random directions in the expansion, relative to the sketch:
                                                                       # an exact eigenstate that the coupling terms annihilate (a magnon
                                                                       # of the kagome flat band inside one ring) is otherwise a fixed point
        mod = model.mod; add = lambda a, q: (a[0]+q[0], (a[1]+q[1]) % mod); sub = lambda a, q: (a[0]-q[0], (a[1]-q[1]) % mod)
        kinds = [model.kinds[s] for s in model.site_kind]; target = (nup, K % mod); left = [{(0, 0)}]
        for r in range(n):
            left.append({add(a, q) for a in left[-1] for q in kinds[r].sectors})
        right = [None]*(n+1); right[n] = {target}
        for r in range(n-1, -1, -1):
            right[r] = {sub(b, q) for b in right[r+1] for q in kinds[r].sectors}
        assert target in left[n], 'this sector does not exist'
        self.bond = [Bond(left[r] & right[r], mod, model.lat.N) for r in range(n+1)]
        self.bond[n].dims[:] = 1; self.bond[0].dims[:] = 1
        self.M = [None]*n; self.plan = [None]*n
        self.LH = [None]*(n+1); self.GL = [None]*(n+1); self.RH = [None]*(n+1); self.GR = [None]*(n+1)
        for r in range(n-1, 0, -1):                                    # random right-orthonormal start
            C = Plan(model, r, self.bond[r], self.bond[r+1]).C
            self.bond[r].dims[:] = np.minimum(D0, C); p = self.plan[r] = Plan(model, r, self.bond[r], self.bond[r+1])
            self.M[r] = np.zeros(p.size)
            for i in np.nonzero(p.DL)[0]:
                self.M[r][p.Moff[i]:p.Moff[i+1]] = np.linalg.qr(rng.standard_normal((p.C[i], p.DL[i])))[0].T.reshape(-1)
        p = self.plan[0] = Plan(model, 0, self.bond[0], self.bond[1]); v = rng.standard_normal(p.size); self.M[0] = v/np.linalg.norm(v)
        self.LH[0] = np.zeros(1); self.GL[0] = np.zeros(1); self.RH[n] = np.zeros(1); self.GR[n] = np.zeros(1)
        for r in range(n-1, 0, -1):
            self._grow_right(r)
        self.stats = dict(products=0, solves=0, energies=[], truncation=0.0, t_products=0.0, t_svd=0.0, t_env=0.0)

    # ---- environments
    def _grow_right(self, r):                                          # needs M[r] right-orthonormal, RH/GR[r+1]
        p = self.plan[r]; M = self.M[r]; lib = self.lib; U = np.empty(p.size); oU = np.empty(p.size); phi = np.empty(p.size)
        lib.m2u(p.ref, _p(M), _p(U)); lib.right_half(p.ref, _p(self.RH[r+1]), _p(self.GR[r+1]), _p(U), _p(oU)); lib.hr_u(p.ref, _p(U), _p(oU))
        lib.u2m(p.ref, _p(oU), _p(phi), 0)
        goff = _off(p.WL.astype(np.int64)*p.DL); self.GR[r] = np.zeros(max(1, int(goff[-1]))); self.RH[r] = np.zeros(max(1, int(p.LHoff[-1])))
        lib.grow_right(p.ref, _p(M), _p(phi), _p(self.GR[r]), _p(goff), _p(self.RH[r]), _p(p.LHoff))

    def _grow_left(self, r):                                           # needs M[r] left-orthonormal, LH/GL[r]
        p = self.plan[r]; M = self.M[r]; lib = self.lib; U = np.empty(p.size); phi = np.zeros(p.size); pU = np.empty(p.size)
        lib.left_half(p.ref, _p(self.LH[r]), _p(self.GL[r]), _p(M), _p(phi)); lib.m2u(p.ref, _p(M), _p(U)); lib.m2u(p.ref, _p(phi), _p(pU))
        lib.hr_u(p.ref, _p(U), _p(pU))
        goff = _off(p.DR.astype(np.int64)*p.WR); self.GL[r+1] = np.zeros(max(1, int(goff[-1]))); self.LH[r+1] = np.zeros(max(1, int(p.RHoff[-1])))
        lib.grow_left(p.ref, _p(U), _p(pU), _p(self.GL[r+1]), _p(goff), _p(self.LH[r+1]), _p(p.RHoff))

    # ---- lowest state of the effective Hamiltonian of layer r
    def solve(self, r, tol=1e-6, maxit=40):
        p = self.plan[r]; lib = self.lib; n = p.size; U = np.empty(n); oU = np.empty(n)
        LH, GL, RH, GR = (_p(a) for a in (self.LH[r], self.GL[r], self.RH[r+1], self.GR[r+1]))

        def mv(v):
            out = np.zeros(n); lib.heff(p.ref, LH, GL, RH, GR, _p(v), _p(out), _p(U), _p(oU)); self.stats['products'] += 1
            return out
        t0 = time.perf_counter(); v = self.M[r]/np.linalg.norm(self.M[r])
        if self.noise > 0:                                            # a start that is an exact eigenvector cannot leave its symmetry sector
            xi = self.rng.standard_normal(n); v = v + (self.noise/np.linalg.norm(xi))*xi; v /= np.linalg.norm(v)
        V = [v]; al = []; be = []; E = None; y = np.array([1.0])
        for it in range(min(maxit, n)):
            w = mv(V[-1]); a = float(V[-1] @ w); al.append(a)
            w -= a*V[-1]
            if be:
                w -= be[-1]*V[-2]
            for u in V:
                w -= (u @ w)*u
            b = float(np.linalg.norm(w))
            ev, Sv = np.linalg.eigh(np.diag(al) + np.diag(be, 1) + np.diag(be, -1)); E = float(ev[0]); y = Sv[:, 0]
            if b*abs(y[-1]) < tol*max(1.0, abs(E)) or b < 1e-13 or len(V) == n:
                break
            be.append(b); V.append(w/b)
        x = sum(c*u for c, u in zip(y, V)); self.M[r] = x/np.linalg.norm(x)
        self.stats['solves'] += 1; self.stats['t_products'] += time.perf_counter()-t0
        return E

    # ---- moves with a sketched subspace expansion
    def _sketch(self, D, W, room, alpha, grow):
        if alpha <= 0:
            return np.zeros_like(D)
        k = np.maximum(4, D) if grow else np.maximum(2, np.ceil(self.sketch*D).astype(np.int64))
        return np.minimum(k, np.maximum(room, 0)).astype(np.int32)     # also where no coupling term reaches (W = 0): random columns

    @staticmethod
    def _keep(svals, Dmax, cutoff):
        allS = np.sort(np.concatenate([s for s in svals.values()] or [np.zeros(1)]))[::-1]
        thr = max(allS[Dmax] if len(allS) > Dmax else 0.0, cutoff*allS[0])
        return {q: int(np.sum(s > thr)) for q, s in svals.items()}

    def move_right(self, r, Dmax, alpha, cutoff=1e-12):
        p = self.plan[r]; lib = self.lib; md = self.model; t0 = time.perf_counter(); U = np.empty(p.size); lib.m2u(p.ref, _p(self.M[r]), _p(U))
        kv = self._sketch(p.DR, p.WR, p.R-p.DR, alpha, int(p.DR.sum()) < Dmax)
        eoff = _off(p.R.astype(np.int64)*kv); ooff = _off(p.WR.astype(np.int64)*kv)
        Om = self.rng.standard_normal(max(1, int(ooff[-1]))); E = np.zeros(max(1, int(eoff[-1])))
        if kv.any():
            lib.sr_apply(p.ref, _p(Om), _p(ooff), _p(kv), _p(U), _p(E), _p(eoff))
        res = {}
        for j in np.nonzero((p.R > 0) & (p.DR+kv > 0))[0]:
            A = U[p.Uoff[j]:p.Uoff[j+1]].reshape(p.R[j], p.DR[j])
            if kv[j]:
                B = E[eoff[j]:eoff[j+1]].reshape(p.R[j], kv[j])/math.sqrt(kv[j]) + (self.random/math.sqrt(p.R[j]))*self.rng.standard_normal((p.R[j], kv[j]))
                A = np.concatenate([A, alpha*B], axis=1)
            res[j] = np.linalg.svd(A, full_matrices=False)
        keep = self._keep({j: s for j, (_, s, _) in res.items()}, Dmax, cutoff)
        pn = self.plan[r+1]; Mn = self.M[r+1]; old = p.DR.copy(); new = np.zeros_like(old)
        for j, k in keep.items():
            new[j] = k
        self.bond[r+1].dims[:] = new; p2 = Plan(md, r, self.bond[r], self.bond[r+1]); pn2 = Plan(md, r+1, self.bond[r+1], self.bond[r+2])
        U2 = np.zeros(p2.size); Mn2 = np.zeros(pn2.size); kept = 0.0
        for j, (Us, s, Vh) in res.items():
            k = new[j]
            if not k:
                continue
            U2[p2.Uoff[j]:p2.Uoff[j+1]] = Us[:, :k].reshape(-1)
            if old[j] and pn.C[j]:
                Cm = s[:k, None]*Vh[:k, :old[j]]; kept += float(np.sum(Cm*Cm))
                Mn2[pn2.Moff[j]:pn2.Moff[j+1]] = (Cm @ Mn[pn.Moff[j]:pn.Moff[j+1]].reshape(old[j], pn.C[j])).reshape(-1)
        self.stats['truncation'] += max(0.0, 1.0-kept); self.stats['t_svd'] += time.perf_counter()-t0
        M2 = np.zeros(p2.size); lib.u2m(p2.ref, _p(U2), _p(M2), 0)
        self.plan[r] = p2; self.M[r] = M2; self.plan[r+1] = pn2; self.M[r+1] = Mn2/np.linalg.norm(Mn2)
        t0 = time.perf_counter(); self._grow_left(r); self.stats['t_env'] += time.perf_counter()-t0

    def move_left(self, r, Dmax, alpha, cutoff=1e-12):
        p = self.plan[r]; lib = self.lib; md = self.model; t0 = time.perf_counter(); M = self.M[r]
        kv = self._sketch(p.DL, p.WL, p.C-p.DL, alpha, int(p.DL.sum()) < Dmax)
        eoff = _off(kv.astype(np.int64)*p.C); ooff = _off(kv.astype(np.int64)*p.WL)
        Om = self.rng.standard_normal(max(1, int(ooff[-1]))); E = np.zeros(max(1, int(eoff[-1])))
        if kv.any():
            lib.sl_apply(p.ref, _p(Om), _p(ooff), _p(kv), _p(M), _p(E), _p(eoff))
        res = {}
        for i in np.nonzero((p.C > 0) & (p.DL+kv > 0))[0]:
            A = M[p.Moff[i]:p.Moff[i+1]].reshape(p.DL[i], p.C[i])
            if kv[i]:
                B = E[eoff[i]:eoff[i+1]].reshape(kv[i], p.C[i])/math.sqrt(kv[i]) + (self.random/math.sqrt(p.C[i]))*self.rng.standard_normal((kv[i], p.C[i]))
                A = np.concatenate([A, alpha*B], axis=0)
            res[i] = np.linalg.svd(A, full_matrices=False)
        keep = self._keep({i: s for i, (_, s, _) in res.items()}, Dmax, cutoff)
        pp = self.plan[r-1]; Up = np.empty(pp.size); lib.m2u(pp.ref, _p(self.M[r-1]), _p(Up)); old = p.DL.copy(); new = np.zeros_like(old)
        for i, k in keep.items():
            new[i] = k
        self.bond[r].dims[:] = new; p2 = Plan(md, r, self.bond[r], self.bond[r+1]); pp2 = Plan(md, r-1, self.bond[r-1], self.bond[r])
        M2 = np.zeros(p2.size); Up2 = np.zeros(pp2.size); kept = 0.0
        for i, (Us, s, Vh) in res.items():
            k = new[i]
            if not k:
                continue
            M2[p2.Moff[i]:p2.Moff[i+1]] = Vh[:k].reshape(-1)
            if old[i] and pp.R[i]:
                Cm = Us[:old[i], :k]*s[:k]; kept += float(np.sum(Cm*Cm))
                Up2[pp2.Uoff[i]:pp2.Uoff[i+1]] = (Up[pp.Uoff[i]:pp.Uoff[i+1]].reshape(pp.R[i], old[i]) @ Cm).reshape(-1)
        self.stats['truncation'] += max(0.0, 1.0-kept); self.stats['t_svd'] += time.perf_counter()-t0
        Mp2 = np.zeros(pp2.size); lib.u2m(pp2.ref, _p(Up2), _p(Mp2), 0)
        self.plan[r] = p2; self.M[r] = M2; self.plan[r-1] = pp2; self.M[r-1] = Mp2/np.linalg.norm(Mp2)
        t0 = time.perf_counter(); self._grow_right(r); self.stats['t_env'] += time.perf_counter()-t0

    def sweep(self, Dmax, alpha, tol=1e-6, noise=0.0):
        self.stats['truncation'] = 0.0; E = None; self.noise = noise   # random start: only in the warm-up sweeps at small bonds
        for r in range(self.n-1):
            E = self.solve(r, tol); self.move_right(r, Dmax, alpha)
        for r in range(self.n-1, 0, -1):
            E = self.solve(r, tol); self.move_left(r, Dmax, alpha)
        self.noise = 0.0; E = self.solve(0, tol); self.stats['energies'].append(E)
        return E

    def bond_sizes(self):
        return [int(b.dims.sum()) for b in self.bond]


ALPHAS = [1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-5, 1e-6, 0.0, 0.0]


def ground_state(model, D, nup=None, K=0, sweeps=None, alphas=None, tol=1e-6, seed=0, log=None):
    """Lowest state with nup up spins (default: half) and momentum 2 pi K / c around the cylinder."""
    nup = model.lat.N//2 if nup is None else nup
    eng = DMRG(model, nup, K, seed=seed); c = (ctypes.c_double*2)(); eng.lib.counts(c, 1)
    alphas = list(alphas or ALPHAS); alphas = alphas[:sweeps] if sweeps and sweeps <= len(alphas) else alphas + [0.0]*((sweeps or 0)-len(alphas))
    t0 = time.perf_counter(); E = None
    for Dw in (max(min(D, 64), D//4), max(min(D, 64), D//2)):          # two warm-up sweeps at small bonds, with a random start
        eng.sweep(Dw, 1e-2, tol, noise=1e-2); eng.stats['energies'].pop()
    for sw, al in enumerate(alphas):
        p0 = eng.stats['products']; E = eng.sweep(D, al, tol)
        if log:
            eng.lib.counts(c, 0)
            log(f'  sweep {sw+1} (expansion {al:.0e}): E = {E:.10f} | largest bond {max(eng.bond_sizes())} | products {eng.stats["products"]-p0} | '
                f'discarded {eng.stats["truncation"]:.1e} | {time.perf_counter()-t0:.1f} s (products {eng.stats["t_products"]:.1f}, '
                f'SVD {eng.stats["t_svd"]:.1f}, environments {eng.stats["t_env"]:.1f}) | multiply-adds so far: dense {c[0]:.2e}, slabs {c[1]:.2e}')
    eng.lib.counts(c, 0)
    info = dict(time=time.perf_counter()-t0, compile=eng.compile_time, products=eng.stats['products'], bonds=eng.bond_sizes(),
                dense_multiply_adds=c[0], slab_multiply_adds=c[1], energies=eng.stats['energies'], discarded=eng.stats['truncation'],
                t_products=eng.stats['t_products'], t_svd=eng.stats['t_svd'], t_env=eng.stats['t_env'])
    return E, eng, info


def magnetisation(model, D, sweeps=None, tol=1e-6, seed=0, log=None, sz2=None):
    """Lowest level of every S_z >= 0 sector, in every momentum sector K = 0 .. c/2 (K and -K have the same levels).
    Returns rows (2 S_z, {K: E}); the field at which the magnetisation steps from S_z to S_z + 1 is E(S_z+1) - E(S_z)."""
    N = model.lat.N; c = model.mod; rows = []
    for tz in (sz2 if sz2 is not None else range(N % 2, N+1, 2)):
        nup = (N+tz)//2; lev = {}
        for K in range(c//2+1):
            try:
                lev[K] = ground_state(model, D, nup=nup, K=K, sweeps=sweeps, tol=tol, seed=seed)[0]
            except AssertionError as e:                                 # a sector that does not exist (e.g. full polarisation, K != 0)
                if 'does not exist' not in str(e):
                    raise
        rows.append((tz, lev))
        if log:
            log(f'  2 S_z = {tz}: ' + ', '.join(f'K={K}: {e:.8f}' for K, e in lev.items()))
    return rows


def lattice_from_spec(kv, cp):
    cell = kv.get('cell', 'square'); Lx, Ly = int(kv['Lx']), int(kv['Ly'])
    if cell == 'square':
        return square_cylinder(Lx, Ly, cp.get('J1', 1.0), cp.get('J2', 0.0))
    assert cell == 'kagome', 'lattice cylinder: cell=square | kagome'
    assert not cp.get('J2'), 'kagome cylinder: nearest-neighbour bonds only so far'
    return kagome_cylinder(Lx, Ly, cp.get('J1', 1.0), tail=kv.get('ends', 'ring') == 'cell')


def run_spec(spec):
    """Entry from edc_cli.py: `lattice cylinder` with `solve method=dmrg`."""
    kind, kv = spec['lattice']; assert kind == 'cylinder', 'method=dmrg: lattice cylinder [cell=kagome] Lx= Ly='
    cp = spec['couplings']; sec, sol = spec['sector'], spec['solve']
    assert set(cp) <= {'J1', 'J2', 'Delta'}, 'dmrg on a cylinder: couplings J1, J2 (square), Delta'
    lat = lattice_from_spec(kv, cp); momentum = sol.get('labels', 'sz,k') != 'sz'
    if sec.get('Sz') == 'all':                                        # the whole magnetisation curve, every momentum
        model = Model(lat, Delta=cp.get('Delta', 1.0), momentum=True); D = int(sol.get('D', 256)); t0 = time.perf_counter()
        rows = magnetisation(model, D, sweeps=int(sol['sweeps']) if 'sweeps' in sol else None, tol=float(sol.get('tol', 1e-6)))
        low = [min(lev.values()) for tz, lev in rows]
        return dict(lattice=lat.name, N=lat.N, model='heisenberg' if model.Delta == 1 else 'xxz', method='dmrg', D=D,
                    Sz=[tz/2 for tz, lev in rows], energy=low, momentum_of_lowest=[min(lev, key=lev.get) for tz, lev in rows],
                    levels_by_momentum=[{str(K): e for K, e in lev.items()} for tz, lev in rows],
                    step_fields=[low[i+1]-low[i] for i in range(len(low)-1)], seconds=round(time.perf_counter()-t0, 1))
    twosz = Fraction(sec.get('Sz', '0'))*2; nup2 = lat.N + twosz; assert nup2.denominator == 1 and nup2 % 2 == 0, f'Sz does not fit {lat.N} spins'
    kf = Fraction(sec.get('k', '0')); K = kf*lat.c/2; assert K.denominator == 1, f'k = 2 K / Ly in units of pi, K integer (Ly = {lat.c})'
    model = Model(lat, Delta=cp.get('Delta', 1.0), momentum=momentum); D = int(sol.get('D', 256))
    E, eng, info = ground_state(model, D, nup=int(nup2)//2, K=int(K), sweeps=int(sol['sweeps']) if 'sweeps' in sol else None,
                                tol=float(sol.get('tol', 1e-6)), seed=int(sol.get('seed', 0)))
    return dict(lattice=lat.name, N=lat.N, model='heisenberg' if model.Delta == 1 else 'xxz', method='dmrg', layers=model.n,
                labels='sz,k' if momentum else 'sz', D=D, Sz=float(twosz/2), k=float(kf), energy=E, energies_by_sweep=info['energies'],
                bond_sizes=info['bonds'], products=info['products'], discarded_weight=info['discarded'], seconds=round(info['time'], 3),
                compile_seconds=round(info['compile'], 3), dense_multiply_adds=info['dense_multiply_adds'], slab_multiply_adds=info['slab_multiply_adds'])


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '-v']                      # [kagome] Lx Ly D1,D2,... [K] [J2 | Delta for kagome] [nup]
    kag = args[0] == 'kagome'; args = args[1:] if kag else args
    Lx, Ly = int(args[0]), int(args[1]); Ds = [int(v) for v in args[2].split(',')]
    K = int(args[3]) if len(args) > 3 else 0; x = float(args[4]) if len(args) > 4 else (1.0 if kag else 0.0); nup = int(args[5]) if len(args) > 5 else None
    model = Model(kagome_cylinder(Lx, Ly), Delta=x) if kag else Model(square_cylinder(Lx, Ly, J2=x))
    for D in Ds:
        E, eng, info = ground_state(model, D, nup=nup, K=K, log=print if '-v' in sys.argv else None)
        print(f'{model.lat.name}, N = {model.lat.N}, K = {K}, D = {D}: E = {E:.10f} | {info["time"]:.1f} s (compile {info["compile"]:.1f} s) | products '
              f'{info["products"]} | bonds {info["bonds"]} | multiply-adds: dense {info["dense_multiply_adds"]:.2e}, slabs {info["slab_multiply_adds"]:.2e}', flush=True)
