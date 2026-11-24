#!/usr/bin/env python3
"""(j, k) layer DMRG on one GPU: the product with the effective Hamiltonian of a layer on the device.

The centre tensor of su2_dmrg (blocks C[la, q, lb] of shape (D_la, n_q, D_lb)) is kept in two layouts, as in
edc_dmrg_gpu:
    M layout, one matrix per left label la:   M_la[a, (q, lb, s, b)]    columns: blocks (q, lb) side by side
    U layout, one matrix per right label lb:  U_lb[(la, q, a, s), b]    rows: blocks (la, q) one under the other
and the product is
    out_la  = HL_la M_la + GL_la S^L_la            S^L_la rows (t, la1, a1): column slabs of M_la1 moved by Y_t
    oU_lb   = U_lb HR_lb + S^R_lb GR_lb            S^R_lb columns (t, lb1, b1): row slabs of U_lb1 moved by X_t
    out    += (H of the layer) M  ;  out += u2m(oU)
with the recoupling coefficients (su2_layer.w_left / w_right) folded into the slab weights.  Every slab move is one
"block-pair gather": for a pair of blocks and a dense weight matrix W (n2 x n1),
    y[dst + s2*dss + row*ldd + col] (+)= sum_s1 W[s2, s1] x[src + s1*sss + row*lds + col],
one CUDA thread per output number; every output number belongs to exactly one group, so no atomics.  The matrix
products are one cuBLAS call per label.  The gather plan of a centre is built on the host from the block lists."""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
EMULATE = os.environ.get('SU2GPU_EMULATE') == '1'                     # the same plans executed by numpy, for checks without a GPU
if EMULATE:
    class _Dev:
        def synchronize(self):
            pass

    class _Cuda:
        Device = _Dev

    class _CP:
        cuda = _Cuda

        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def asnumpy(a):
            return np.asarray(a)
    cp = _CP()
else:
    import cupy as cp
import su2_dmrg as sd
from su2_layer import w_left, w_right

_SRC = r'''
typedef long long i64;
extern "C" __global__ void bgather(const i64* g, const i64* tg, const i64* ts, i64 ntiles, const double* W, const double* x, double* y, int add) {
  i64 b = blockIdx.x; if (b >= ntiles) return;
  const i64* r = g + 12*tg[b]; i64 p = ts[b] + threadIdx.x;
  i64 h = r[8], w = r[9], hw = h*w; if (p >= r[3]*hw) return;
  i64 s2 = p / hw, q = p % hw, row = q / w, col = q % w, n1 = r[7];
  const double* Wr = W + r[10] + s2*n1; const double* xs = x + r[4] + row*r[5] + col; double acc = 0.0;
  for (i64 s1 = 0; s1 < n1; s1++) acc += Wr[s1] * xs[s1*r[6]];
  double* o = y + r[0] + s2*r[2] + row*r[1] + col;
  if (add) *o += acc; else *o = acc;
}
'''
TPB = 128
_K = None


def _kernel():
    global _K
    if _K is None:
        _K = cp.RawKernel(_SRC, 'bgather')
    return _K


class Plan:
    """A list of block-pair gathers: group = (dst, ldd, dss, n2, src, lds, sss, n1, h, w, woff, 0).  Groups that write
    the same destination block (several source sectors of one target) go into successive waves: the first wave writes,
    the later ones add, so no two threads of one launch write the same number."""
    def __init__(self):
        self.g = []; self.W = []; self.nw = 0; self.seen = {}

    def add(self, dst, ldd, dss, src, lds, sss, h, w, Wm):
        Wm = np.asarray(Wm, dtype=np.float64)
        if h <= 0 or w <= 0 or Wm.size == 0:
            return
        n2, n1 = Wm.shape; wave = self.seen.get(dst, 0); self.seen[dst] = wave + 1
        self.g.append((dst, ldd, dss, n2, src, lds, sss, n1, h, w, self.nw, wave)); self.W.append(Wm.ravel()); self.nw += Wm.size

    def upload(self):
        self.waves = []; self.wave_groups = []
        if not self.g:
            return self
        G = np.array(self.g, dtype=np.int64); self.Wh = np.concatenate(self.W); self.dW = cp.asarray(self.Wh)
        self.wave_groups = [[tuple(int(v) for v in row) for row in G[G[:, 11] == wv]] for wv in range(int(G[:, 11].max()) + 1)]
        if EMULATE:
            return self
        for wv in range(int(G[:, 11].max()) + 1):
            g = G[G[:, 11] == wv]; tot = g[:, 3]*g[:, 8]*g[:, 9]; cnt = (tot + TPB - 1)//TPB
            tg = np.repeat(np.arange(len(g), dtype=np.int64), cnt); first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            ts = (np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(first, cnt))*TPB
            self.waves.append((cp.asarray(g.ravel()), cp.asarray(tg), cp.asarray(ts), len(tg)))
        return self

    def run(self, x, y, add):
        if EMULATE:
            for i, g in enumerate(self.wave_groups):
                for (dst, ldd, dss, n2, src, lds, sss, n1, h, w, woff, _) in g:
                    R = np.arange(h)[:, None]; Cc = np.arange(w)[None, :]; Wm = self.Wh[woff:woff + n2*n1].reshape(n2, n1)
                    for s2 in range(n2):
                        acc = sum(Wm[s2, s1]*x[src + s1*sss + R*lds + Cc] for s1 in range(n1))
                        if add or i:
                            y[dst + s2*dss + R*ldd + Cc] += acc
                        else:
                            y[dst + s2*dss + R*ldd + Cc] = acc
            return
        for i, (dg, dtg, dts, n) in enumerate(self.waves):
            if n:
                _kernel()((n,), (TPB,), (dg, dtg, dts, np.int64(n), self.dW, x, y, np.int32(1 if (add or i) else 0)))


class Layout:
    """Offsets of the centre of layer r in both layouts, from the block list of su2_dmrg."""
    def __init__(self, eng, r):
        m = eng.m; self.r = r; ns = m.sectors[r]; D = eng.D
        self.L = [la for la in eng.labels[r] if D[r][la]]; self.R = [lb for lb in eng.labels[r+1] if D[r+1][lb]]
        self.cols = {la: eng._cols(r, la) for la in self.L}; self.rows = {lb: eng._rows(r, lb) for lb in self.R}
        self.Moff = {}; self.coff = {}; self.C = {}; o = 0
        for la in self.L:
            c = 0
            for q, lb in self.cols[la]:
                self.coff[(la, q, lb)] = c; c += ns[q]*D[r+1][lb]
            self.C[la] = c; self.Moff[la] = o; o += D[r][la]*c
        self.n = o; self.Uoff = {}; self.roff = {}; self.Rr = {}; o = 0
        for lb in self.R:
            rr = 0
            for la, q in self.rows[lb]:
                self.roff[(la, q, lb)] = rr; rr += D[r][la]*ns[q]
            self.Rr[lb] = rr; self.Uoff[lb] = o; o += rr*D[r+1][lb]
        assert o == self.n
        self.keys = [(la, q, lb) for la in self.L for q, lb in self.cols[la]]

    def to_M(self, eng, C):                                              # dict of blocks -> flat M layout (host)
        v = np.zeros(self.n); D = eng.D; r = self.r
        for la in self.L:
            Mx = v[self.Moff[la]:self.Moff[la] + D[r][la]*self.C[la]].reshape(D[r][la], self.C[la])
            for q, lb in self.cols[la]:
                c0 = self.coff[(la, q, lb)]; x = C[(la, q, lb)]; Mx[:, c0:c0 + x.shape[1]*x.shape[2]] = x.reshape(x.shape[0], -1)
        return v

    def from_M(self, eng, v):
        D = eng.D; r = self.r; out = {}
        for la in self.L:
            Mx = v[self.Moff[la]:self.Moff[la] + D[r][la]*self.C[la]].reshape(D[r][la], self.C[la])
            for q, lb in self.cols[la]:
                c0 = self.coff[(la, q, lb)]; nq = eng.m.sectors[r][q]; out[(la, q, lb)] = np.ascontiguousarray(Mx[:, c0:c0 + nq*D[r+1][lb]]).reshape(D[r][la], nq, D[r+1][lb])
        return out


def _bytgt(ops):
    d = {}
    for (q2, q1), M in ops.items():
        d.setdefault(q2, []).append((q1, M))
    return d


def build_SL(eng, r, L, entries):
    """Stacks of column-slab moves in the M layout.  entries: {la2: [(t, la1, ops by target sector, coef)]} with
    coef(la1, q1, lb, la2, q2) -> float.  Row block (t, la1) of target la2 holds D_la1 rows; columns are those of M_la2.
    Returns (plan, {la2: (offset, rows, [(t, la1, row offset)])}, size)."""
    ns = eng.m.sectors[r]; Dl, Dr = eng.D[r], eng.D[r+1]; plan = Plan(); index = {}; size = 0
    for la2, blocks in entries.items():
        rows = []; ro = 0
        for t, la1, ops, coef in blocks:
            used = False
            for q2, lb in L.cols[la2]:
                for q1, Ym in ops.get(q2, ()):
                    if (la1, q1, lb) in L.coff:
                        w = coef(la1, q1, lb, la2, q2)
                        if w:
                            plan.add(size + ro*L.C[la2] + L.coff[(la2, q2, lb)], L.C[la2], Dr[lb], L.Moff[la1] + L.coff[(la1, q1, lb)], L.C[la1], Dr[lb], Dl[la1], Dr[lb], w*Ym)
                            used = True
            if used:
                rows.append((t, la1, ro)); ro += Dl[la1]
        if rows:
            index[la2] = (size, ro, rows); size += ro*L.C[la2]
    return plan, index, size


def build_SR(eng, r, L, entries):
    """Stacks of row-slab moves in the U layout.  entries: {lb2: [(t, lb1, ops by target sector, coef)]} with
    coef(la, q1, lb1, q2, lb2) -> float.  Column block (t, lb1) of target lb2 holds D_lb1 columns; rows are those of
    U_lb2.  Returns (plan, {lb2: (offset, columns, [(t, lb1, column offset)])}, size)."""
    ns = eng.m.sectors[r]; Dl, Dr = eng.D[r], eng.D[r+1]; plan = Plan(); index = {}; size = 0
    for lb2, blocks in entries.items():
        cand = []; co = 0
        for t, lb1, ops, coef in blocks:
            groups = []
            for la, q2 in L.rows[lb2]:
                for q1, Xm in ops.get(q2, ()):
                    if (la, q1, lb1) in L.roff:
                        w = coef(la, q1, lb1, q2, lb2)
                        if w:
                            groups.append((la, q1, q2, w, Xm))
            if groups:
                cand.append((t, lb1, co, groups)); co += Dr[lb1]
        if not cand:
            continue
        Wd = co
        for t, lb1, co, groups in cand:
            for la, q1, q2, w, Xm in groups:
                plan.add(size + L.roff[(la, q2, lb2)]*Wd + co, ns[q2]*Wd, Wd, L.Uoff[lb1] + L.roff[(la, q1, lb1)]*Dr[lb1], ns[q1]*Dr[lb1], Dr[lb1], Dl[la], Dr[lb1], w*Xm)
        index[lb2] = (size, Wd, [(t, lb1, co) for t, lb1, co, _ in cand]); size += L.Rr[lb2]*Wd
    return plan, index, size


class Product:
    """The effective Hamiltonian of the centre r on the device, with the environments of the engine (host dicts)."""
    def __init__(self, eng, r, prof=None, parts=('left', 'right')):
        t0 = time.perf_counter(); m = eng.m; D = eng.D; ns = m.sectors[r]; self.prof = prof if prof is not None else {}; self.parts = parts
        L = self.lay = Layout(eng, r); n = L.n; self.n = n; jj = lambda lab: lab[0]
        Dl, Dr = D[r], D[r+1]
        # copies between the layouts (whole blocks, weight 1)
        self.m2u = Plan(); self.u2m = Plan(); one = np.ones((1, 1))
        for (la, q, lb) in L.keys:
            nb = ns[q]*Dr[lb]
            self.m2u.add(L.Uoff[lb] + L.roff[(la, q, lb)]*Dr[lb], nb, 0, L.Moff[la] + L.coff[(la, q, lb)], L.C[la], 0, Dl[la], nb, one)
            self.u2m.add(L.Moff[la] + L.coff[(la, q, lb)], L.C[la], 0, L.Uoff[lb] + L.roff[(la, q, lb)]*Dr[lb], nb, 0, Dl[la], nb, one)
        # the layer Hamiltonian on the M layout
        self.hs = Plan()
        for (la, q, lb) in L.keys:
            o = L.Moff[la] + L.coff[(la, q, lb)]; self.hs.add(o, L.C[la], Dr[lb], o, L.C[la], Dr[lb], Dl[la], Dr[lb], m.Hs[r][q])
        # left couplings: S^L_la2 rows (t, la1, a1), and GL_la2 = [O_t(la2 <- la1)] side by side
        self.left = []; self.GL = {}; entries = {}
        if r > 0 and 'left' in parts:
            for t, (cf, dk, X, Y) in enumerate(m.terms[r-1]):
                Yt = _bytgt(Y); cw = (lambda cf: lambda la1, q1, lb, la2, q2: cf*w_left(la1[0], q1[0], lb[0], la2[0], q2[0]))(cf)
                for (la2, la1), O in eng.OL[r][t].items():
                    if Dl.get(la2, 0) and Dl.get(la1, 0):
                        entries.setdefault(la2, []).append((t, la1, Yt, cw))
        self.sl, self.SLidx, SLsize = build_SL(eng, r, L, entries)
        for la2, (off, Wd, rows) in self.SLidx.items():
            self.GL[la2] = cp.asarray(np.hstack([eng.OL[r][t][(la2, la1)] for t, la1, _ in rows])); self.left.append((la2, Wd))
        self.SL = cp.zeros(max(1, SLsize))
        # right couplings: S^R_lb2 columns (t, lb1, b1), and GR_lb2 = [O_t(lb2 <- lb1)^T] one under the other
        self.right = []; self.GR = {}; entries = {}
        if r < eng.n - 1 and 'right' in parts:
            for t, (cf, dk, X, Y) in enumerate(m.terms[r]):
                Xt = _bytgt(X); cw = (lambda cf: lambda la, q1, lb1, q2, lb2: cf*w_right(la[0], q1[0], lb1[0], q2[0], lb2[0]))(cf)
                for (lb2, lb1), O in eng.OR[r+1][t].items():
                    if Dr.get(lb2, 0) and Dr.get(lb1, 0):
                        entries.setdefault(lb2, []).append((t, lb1, Xt, cw))
        self.sr, self.SRidx, SRsize = build_SR(eng, r, L, entries)
        for lb2, (off, Wd, cols) in self.SRidx.items():
            self.GR[lb2] = cp.asarray(np.vstack([eng.OR[r+1][t][(lb2, lb1)].T for t, lb1, _ in cols])); self.right.append((lb2, Wd))
        self.SR = cp.zeros(max(1, SRsize))
        for p in (self.m2u, self.u2m, self.hs, self.sl, self.sr):
            p.upload()
        self.HL = {la: cp.asarray(eng.HL[r][la]) for la in L.L} if (r > 0 and 'left' in parts) else None
        self.HR = {lb: cp.asarray(eng.HR[r+1][lb]) for lb in L.R} if 'right' in parts else None
        self.U = cp.empty(n); self.oU = cp.empty(n); self.out = cp.empty(n)
        self._sync('plan', t0)

    def _sync(self, key, t0):
        cp.cuda.Device().synchronize(); t1 = time.perf_counter(); self.prof[key] = self.prof.get(key, 0.0) + t1 - t0
        return t1

    def apply_left(self, x):
        """(H of the left block + left couplings + H of the layer) x, M layout."""
        L = self.lay; out = cp.zeros(self.n); self.sl.run(x, self.SL, False)
        for la in L.L:
            o, C = L.Moff[la], L.C[la]
            if self.HL is not None:
                Da = self.HL[la].shape[0]; cp.matmul(self.HL[la], x[o:o + Da*C].reshape(Da, C), out=out[o:o + Da*C].reshape(Da, C))
        for la2, Wd in self.left:
            o, C = L.Moff[la2], L.C[la2]; Da = self.GL[la2].shape[0]; so = self.SLidx[la2][0]
            out[o:o + Da*C].reshape(Da, C)[...] += self.GL[la2] @ self.SL[so:so + Wd*C].reshape(Wd, C)
        self.hs.run(x, out, True)
        return out

    def apply_right(self, x):
        """(H of the right block + right couplings + H of the layer) x, M layout."""
        L = self.lay; U = cp.empty(self.n); oU = cp.empty(self.n); out = cp.zeros(self.n); self.m2u.run(x, U, False); self.sr.run(U, self.SR, False)
        for lb in L.R:
            o = L.Uoff[lb]; Rr = L.Rr[lb]; Db = self.HR[lb].shape[0]; cp.matmul(U[o:o + Rr*Db].reshape(Rr, Db), self.HR[lb], out=oU[o:o + Rr*Db].reshape(Rr, Db))
        for lb2, Wd in self.right:
            o = L.Uoff[lb2]; Rr = L.Rr[lb2]; Db = self.HR[lb2].shape[0]; so = self.SRidx[lb2][0]
            oU[o:o + Rr*Db].reshape(Rr, Db)[...] += self.SR[so:so + Rr*Wd].reshape(Rr, Wd) @ self.GR[lb2]
        self.u2m.run(oU, out, True); self.hs.run(x, out, True)
        return out

    def apply(self, x):
        L = self.lay; D = None; t = time.perf_counter(); out = self.out; out.fill(0.0)
        Dl = {la: (L.Moff[la], L.C[la]) for la in L.L}
        self.sl.run(x, self.SL, False); t = self._sync('gather', t)
        for la in L.L:
            o, C = Dl[la]; d = (self.HL[la].shape[0] if self.HL is not None else None)
            if self.HL is not None:
                Da = self.HL[la].shape[0]; cp.matmul(self.HL[la], x[o:o + Da*C].reshape(Da, C), out=out[o:o + Da*C].reshape(Da, C))
        for la2, Wd in self.left:
            o, C = Dl[la2]; Da = self.GL[la2].shape[0]; so = self.SLidx[la2][0]; S = self.SL[so:so + Wd*C].reshape(Wd, C)
            out[o:o + Da*C].reshape(Da, C)[...] += self.GL[la2] @ S
        t = self._sync('gemm', t)
        self.hs.run(x, out, True); self.m2u.run(x, self.U, False); self.sr.run(self.U, self.SR, False); t = self._sync('gather', t)
        oU = self.oU
        for lb in L.R:
            o = L.Uoff[lb]; Rr = L.Rr[lb]; Db = self.HR[lb].shape[0]
            cp.matmul(self.U[o:o + Rr*Db].reshape(Rr, Db), self.HR[lb], out=oU[o:o + Rr*Db].reshape(Rr, Db))
        for lb2, Wd in self.right:
            o = L.Uoff[lb2]; Rr = L.Rr[lb2]; Db = self.HR[lb2].shape[0]; so = self.SRidx[lb2][0]; S = self.SR[so:so + Rr*Wd].reshape(Rr, Wd)
            oU[o:o + Rr*Db].reshape(Rr, Db)[...] += S @ self.GR[lb2]
        t = self._sync('gemm', t)
        self.u2m.run(oU, out, True); self._sync('gather', t)
        return out


def grow_left_dev(eng, r, prof=None):
    """Left environment of bond r + 1 from the left-canonical tensor of layer r (eng.T[r]), on the device.  Returns
    (HL {label: D x D}, OL [{(L2, L1): D2 x D1} per term of the bond r, r + 1]) as host arrays."""
    m = eng.m; P = Product(eng, r, prof, parts=('left',)); L = P.lay; x = cp.asarray(L.to_M(eng, eng.T[r])); tmp = P.apply_left(x)
    xU = cp.empty(L.n); tU = cp.empty(L.n); P.m2u.run(x, xU, False); P.m2u.run(tmp, tU, False); Dr = eng.D[r+1]; HL = {}
    blk = lambda V, lb: V[L.Uoff[lb]:L.Uoff[lb] + L.Rr[lb]*Dr[lb]].reshape(L.Rr[lb], Dr[lb])
    for lb in L.R:
        H = blk(xU, lb).T @ blk(tU, lb); HL[lb] = cp.asnumpy(0.5*(H + H.T))
    OL = [{} for _ in (m.terms[r] if r < eng.n - 1 else [])]
    if OL:
        entries = {}
        for t, (cf, dk, X, Y) in enumerate(m.terms[r]):
            Xt = _bytgt(X); co = lambda la, q1, lb1, q2, lb2: sd.t_second(la[0], q1[0], lb1[0], q2[0], lb2[0])
            for lb2 in L.R:
                entries.setdefault(lb2, []).extend((t, lb1, Xt, co) for lb1 in L.R)
        plan, idx, size = build_SR(eng, r, L, entries); plan.upload(); S = cp.zeros(max(1, size)); plan.run(xU, S, False)
        for lb2, (off, Wd, cols) in idx.items():
            F = cp.asnumpy(blk(xU, lb2).T @ S[off:off + L.Rr[lb2]*Wd].reshape(L.Rr[lb2], Wd))
            for t, lb1, co_ in cols:
                OL[t][(lb2, lb1)] = F[:, co_:co_ + Dr[lb1]]
    return HL, OL


def grow_right_dev(eng, r, prof=None):
    """Right environment of bond r from the right-canonical tensor of layer r (eng.T[r]), on the device.  Returns
    (HR {label: D x D}, OR [{(L2, L1): D2 x D1} per term of the bond r - 1, r]) as host arrays."""
    m = eng.m; P = Product(eng, r, prof, parts=('right',)); L = P.lay; x = cp.asarray(L.to_M(eng, eng.T[r])); tmp = P.apply_right(x)
    Dl = eng.D[r]; HR = {}
    blk = lambda V, la: V[L.Moff[la]:L.Moff[la] + Dl[la]*L.C[la]].reshape(Dl[la], L.C[la])
    for la in L.L:
        H = blk(x, la) @ blk(tmp, la).T; HR[la] = cp.asnumpy(0.5*(H + H.T))
    OR = [{} for _ in (m.terms[r-1] if r > 0 else [])]
    if OR:
        entries = {}
        for t, (cf, dk, X, Y) in enumerate(m.terms[r-1]):
            Yt = _bytgt(Y); co = lambda la1, q1, lb, la2, q2: sd.t_first(q1[0], lb[0], la1[0], q2[0], la2[0])
            for la2 in L.L:
                entries.setdefault(la2, []).extend((t, la1, Yt, co) for la1 in L.L)
        plan, idx, size = build_SL(eng, r, L, entries); plan.upload(); S = cp.zeros(max(1, size)); plan.run(x, S, False)
        for la2, (off, Wd, rows) in idx.items():
            C = L.C[la2]; F = cp.asnumpy(blk(x, la2) @ S[off:off + Wd*C].reshape(Wd, C).T)
            for t, la1, ro in rows:
                OR[t][(la2, la1)] = F[:, ro:ro + Dl[la1]]
    return HR, OR


class SU2GPU(sd.SU2DMRG):
    """su2_dmrg with the layer solves on the device (version 0 of the (j, k) GPU line); moves and environments are
    those of the numpy reference."""
    version = 'su2-gpu-v0'

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.prof = {}; self.peak = 0

    def solve(self, r, tol=1e-6, maxit=40):
        t00 = time.perf_counter(); H = Product(self, r, self.prof); n = H.n; L = H.lay
        v = cp.asarray(L.to_M(self, self.T[r])); v /= cp.linalg.norm(v)
        if self.noise > 0:
            xi = cp.asarray(self.rng.standard_normal(n)); v = v + (self.noise/float(cp.linalg.norm(xi)))*xi; v /= cp.linalg.norm(v)
        m = min(maxit, n); V = cp.empty((m, n)); V[0] = v; al = []; be = []; E = None; y = np.array([1.0]); k = 1
        for it in range(m):
            w = H.apply(V[it]).copy(); self.stats['products'] += 1; t0 = time.perf_counter()
            a = float(V[it] @ w); al.append(a); w -= a*V[it]
            if be:
                w -= be[-1]*V[it-1]
            w -= V[:k].T @ (V[:k] @ w); b = float(cp.linalg.norm(w))
            ev, Sv = np.linalg.eigh(np.diag(al) + np.diag(be, 1) + np.diag(be, -1)); E = float(ev[0]); y = Sv[:, 0]
            stop = b*abs(y[-1]) < tol*max(1.0, abs(E)) or b < 1e-13 or k == n or k == m
            if not stop:
                be.append(b); V[k] = w/b; k += 1
            H._sync('lanczos', t0)
            if stop:
                break
        x = cp.asarray(y[:k]) @ V[:k]; x /= cp.linalg.norm(x); self.T[r] = L.from_M(self, cp.asnumpy(x))
        self.peak = max(self.peak, cp.get_default_memory_pool().total_bytes()); del H, V
        self.stats['t_solve'] += time.perf_counter() - t00
        return E


class SU2GPU1(SU2GPU):
    """Version 1: the environments also grow on the device (host copies kept for the numpy moves)."""
    version = 'su2-gpu-v1'

    def _grow_left(self, r):
        if not hasattr(self, 'prof'):
            return super()._grow_left(r)
        t0 = time.perf_counter(); HL, OL = grow_left_dev(self, r, self.prof)
        self.HL[r+1] = HL; self.OL[r+1] = OL; self.OLsrc[r+1] = [self._src(O) for O in OL]; self.prof['env'] = self.prof.get('env', 0.0) + time.perf_counter() - t0

    def _grow_right(self, r):
        if not hasattr(self, 'prof'):
            return super()._grow_right(r)
        t0 = time.perf_counter(); HR, OR = grow_right_dev(self, r, self.prof)
        self.HR[r] = HR; self.OR[r] = OR; self.ORsrc[r] = [self._src(O) for O in OR]; self.prof['env'] = self.prof.get('env', 0.0) + time.perf_counter() - t0


def check_env(Lx, c, tail, D, S2, K, sweeps=2, seed=3):
    """Environments grown on the device against the numpy reference on the same tensors."""
    import edc_dmrg as ed
    lat = ed.kagome_cylinder(Lx, c, tail=tail); model = sd.SU2Model(lat); eng = sd.SU2DMRG(model, S2, K, seed=seed)
    for _ in range(sweeps):
        eng.sweep(D, 1e-2, 1e-6)
    worst = 0.0

    def cmp(A, B):
        keys = set(A) | set(B); d = 0.0; sc = max([float(np.abs(v).max()) for v in B.values() if np.size(v)] + [1e-300])
        for k in keys:
            a = A.get(k); b = B.get(k)
            a = np.zeros_like(b) if a is None else a; b = np.zeros_like(a) if b is None else b
            d = max(d, float(np.abs(np.asarray(a) - np.asarray(b)).max()) if np.size(a) else 0.0)
        return d/sc
    for r in range(eng.n - 1, 0, -1):
        HR, OR = grow_right_dev(eng, r); d = max([cmp(HR, eng.HR[r])] + [cmp(O, P) for O, P in zip(OR, eng.OR[r])]); worst = max(worst, d)
        print(f'  right environment of bond {r:2d}: device - numpy {d:.1e}', flush=True)
    for r in range(eng.n - 1):
        eng.solve(r, 1e-6); eng.move_right(r, D, 0.0)
        HL, OL = grow_left_dev(eng, r); d = max([cmp(HL, eng.HL[r+1])] + [cmp(O, P) for O, P in zip(OL, eng.OL[r+1])]); worst = max(worst, d)
        print(f'  left environment of bond {r+1:2d}: device - numpy {d:.1e}', flush=True)
    print(f'{lat.name}, S = {S2/2:g}, K = {K}, D = {D}: largest relative difference of the environments {worst:.1e}')
    return worst


def check_product(Lx, c, tail, D, S2, K, sweeps=2, seed=3):
    """Device product against the numpy reference on random vectors, at every layer of a state after a few sweeps."""
    import edc_dmrg as ed
    lat = ed.kagome_cylinder(Lx, c, tail=tail); model = sd.SU2Model(lat); eng = sd.SU2DMRG(model, S2, K, seed=seed)
    for _ in range(sweeps):
        eng.sweep(D, 1e-2, 1e-6)
    rng = np.random.default_rng(seed); worst = 0.0
    for r in range(eng.n):
        H = Product(eng, r); L = H.lay; C = {k: rng.standard_normal(x.shape) for k, x in eng.T[r].items()}
        ref = L.to_M(eng, eng.heff(r, C)); got = cp.asnumpy(H.apply(cp.asarray(L.to_M(eng, C))))
        d = float(np.abs(got - ref).max()/np.abs(ref).max()); worst = max(worst, d)
        print(f'  layer {r:2d}: size {H.n:8d}, groups {len(H.sl.g)} + {len(H.sr.g)} in {len(H.sl.waves)} + {len(H.sr.waves)} waves, device - numpy {d:.1e}', flush=True)
        if r < eng.n - 1:
            eng.move_right(r, D, 0.0)
    print(f'{lat.name}, S = {S2/2:g}, K = {K}, D = {D}: largest relative difference {worst:.1e}')
    return worst


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        w = max(check_product(2, 3, True, 64, 0, 0), check_product(2, 3, True, 64, 2, 1), check_product(4, 3, False, 60, 1, 0))
        w = max(w, check_env(2, 3, True, 64, 0, 0), check_env(2, 3, True, 64, 2, 1), check_env(4, 3, False, 60, 1, 0))
        assert w < 1e-12
    else:
        import edc_dmrg as ed
        Lx, c, D, S2, K = (int(v) for v in sys.argv[1:6]); tail = len(sys.argv) > 6 and sys.argv[6] == 'tail'
        lat = ed.kagome_cylinder(Lx, c, tail=tail); model = sd.SU2Model(lat)
        cls = SU2GPU1 if os.environ.get('SU2GPU_VERSION', '1') == '1' else SU2GPU; eng = cls(model, S2, K); print('version', cls.version, flush=True)
        t0 = time.perf_counter(); alphas = list(sd.ALPHAS)
        sched = [(max(min(D, 64), D//4), 1e-2, 3e-5, 1e-2), (max(min(D, 64), D//2), 1e-2, 3e-5, 1e-2)] + [(D, al, 3e-5 if al > 0 else 1e-6, 0.0) for al in alphas]
        for sw, (Dw, al, tl, noise) in enumerate(sched):
            ts = time.perf_counter(); p0 = dict(eng.prof); s0 = dict(eng.stats); E = eng.sweep(Dw, al, tl, noise); te = time.perf_counter()
            print(f'  sweep {sw+1:2d} D={Dw:5d}: E = {E:.10f} | {te-ts:7.2f} s (elapsed {te-t0:8.2f}) | products {eng.stats["products"]-s0["products"]} | solve {eng.stats["t_solve"]-s0["t_solve"]:.1f} '
                  f'(device: ' + ', '.join(f'{k} {eng.prof[k]-p0.get(k, 0):.1f}' for k in sorted(eng.prof)) + f'), move {eng.stats["t_move"]-s0["t_move"]:.1f}, env {eng.stats["t_env"]-s0["t_env"]:.1f} | '
                  f'bond {max(eng.bond_multiplets())} multiplets', flush=True)
