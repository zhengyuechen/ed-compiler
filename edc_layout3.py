#!/usr/bin/env python3
"""One H x with three layouts of the state vector (kagome, nearest-neighbour Heisenberg).

H = H_AB + H_BC + H_CA, where A, B, C are the three kagome sublattices: every nearest-neighbour bond joins two of them and
leaves the third untouched. On a torus whose symmetry operations map every sublattice onto itself (translations,
inversion, global spin flip) the vector is repacked for each of the three terms: the word of the untouched sublattice
(the spectator) is an orbit representative, the other two words run over all configurations of the right magnetisation.
In such a layout the same operator on 2 N/3 spins acts on every spectator column, with no symmetry search inside it and
with the columns adjacent in memory. One product is, three times: pack (gather from the compact vector), pair operator,
unpack (gather back and add).

Layout3      conversions through stored index tables; numpy product; the reference for small clusters and for tests.
Layout3Fly   conversions computed inside the CUDA kernels from tables of 2^(N/3) entries; columns whose spectator word has
             a stabiliser carry a bitmap with rank counters and one list of canonical words.
Layout3Engine  what the model language uses (`solve engine=layout3`): ground_energy, time_product.

Covers: kagome clusters whose little group at k = 0 is {E, inversion}; nearest-neighbour coupling J1; sector S_z = 0,
k = 0, trivial irrep, spin-flip even; double precision; one GPU (device=gpu) or numpy for small clusters (device=cpu).
Measured (one H100, 3 x 4 cells, 36 sites, 189,090,447 states): 0.060 s per product against 0.138 s for the sub engine
computing matrix elements on the fly and 0.036 s for its stored connectivity (41.8 GiB there, 7.2 GiB here).
"""
import time, math
import numpy as np
import edc_lattices as L

KERNEL = r'''
extern "C" __global__ void pair(const double* X, double* Y, const unsigned int* W, const int* OFF, const int* IDX,
                                const unsigned int* MASK, int nb, int M, long long n, int Nc, double J) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    long long r = i / M; int col = (int)(i - r * M);
    unsigned int w = W[r], lom = (1u << Nc) - 1u; double acc = 0.0; int anti = 0;
    for (int b = 0; b < nb; b++) {
        unsigned int m = MASK[b], t = w & m;
        if (t && t != m) { anti++; unsigned int v = w ^ m; long long rr = (long long)OFF[v >> Nc] + IDX[v & lom]; acc += X[rr * M + col]; }
    }
    Y[i] = 0.25 * J * (nb - 2 * anti) * X[i] + 0.5 * J * acc;
}
'''

class Layout3:
    def __init__(self, A1, A2, J=1.0, log=lambda *a: None, tables=True):
        t0 = time.perf_counter()
        cl = L.LatticeCluster('kagome', A1, A2); self.cl, self.J = cl, J
        N, Nc = cl.N, cl.Nc; self.N, self.Nc = N, Nc; full = (1 << Nc) - 1; self.full = full; half = N//2
        perms, chars, ops = cl.group((0.0, 0.0), 'X0'); nG = len(perms); self.nG = nG
        sub = np.arange(N)//Nc
        if not all(sub[p[t]] == sub[t] for p in perms for t in range(N)):
            raise ValueError('layout3: a symmetry operation of this cluster exchanges the sublattices (use a cluster whose point group is inversion only)')
        words = np.arange(1 << Nc, dtype=np.int64)
        GT = np.zeros((3, nG, 1 << Nc), np.int64)          # action of group element g on the word of sublattice S
        for S in range(3):
            for g, p in enumerate(perms):
                for c in range(Nc):
                    GT[S, g] |= ((words >> c) & 1) << (p[S*Nc+c]-S*Nc)
        pc = np.zeros(1 << Nc, np.int64)
        for c in range(Nc): pc += (words >> c) & 1
        IDX = np.zeros(1 << Nc, np.int64); LOWS = []
        for n in range(Nc+1):
            m = pc == n; IDX[m] = np.arange(m.sum()); LOWS.append(words[m])
        comb = np.array([math.comb(Nc, n) for n in range(Nc+1)], np.int64)
        P2 = 2*Nc+1
        OFFT = np.zeros((P2, 1 << Nc), np.int64); NROW = np.zeros(P2, np.int64); W = []
        for p in range(P2):
            nlo = p-pc; ok = (nlo >= 0) & (nlo <= Nc); c = np.where(ok, comb[np.clip(nlo, 0, Nc)], 0)
            OFFT[p] = np.concatenate([[0], np.cumsum(c)[:-1]]); NROW[p] = c.sum()
            W.append(np.concatenate([LOWS[p-pc[h]] | (h << Nc) for h in range(1 << Nc) if ok[h]]) if NROW[p] else np.zeros(0, np.int64))
        self.GT, self.pc, self.IDX, self.OFFT, self.NROW, self.W, self.P2, self.half = GT, pc, IDX, OFFT, NROW, W, P2, half
        # spectator words: representative, a group element that reaches it, columns, stabilisers
        self.REP, self.GI, self.GF, self.COL, self.M, self.BASE, self.cols, self.stab = [], [], [], [], [], [], [], []
        for S in range(3):
            best = np.full(1 << Nc, 1 << 40, np.int64); bg = np.zeros(1 << Nc, np.int64); bf = np.zeros(1 << Nc, np.int64)
            for g in range(nG):
                for f in (0, 1):
                    cand = GT[S, g] ^ (full*f); m = cand < best; best[m] = cand[m]; bg[m] = g; bf[m] = f
            reps = np.unique(best); COL = np.zeros(1 << Nc, np.int64); M = np.zeros(P2, np.int64); cols = []
            for p in range(P2):
                cp_ = reps[pc[reps] == half-p]; cols.append(cp_); M[p] = len(cp_); COL[cp_] = np.arange(len(cp_))
            self.REP.append(best); self.GI.append(bg); self.GF.append(bf); self.COL.append(COL); self.M.append(M); self.cols.append(cols)
            self.BASE.append(np.concatenate([[0], np.cumsum(NROW*M)]))
            self.stab.append({int(r): [(g, f) for g in range(nG) for f in (0, 1) if (int(GT[S, g, r]) ^ (full*f)) == int(r)] for r in reps})
        self.nexp = [int(b[-1]) for b in self.BASE]
        log(f'tables {time.perf_counter()-t0:.1f} s; expanded entries per layout {self.nexp}')
        self._masks()
        if not tables:
            self.setup_seconds = time.perf_counter()-t0; return
        # compact basis = canonical entries of layout C (spectator 2, active words A, B)
        S = 2; n = self.nexp[S]; canon = np.zeros(n, bool); ecan = np.zeros(n, np.int32); ssz = np.ones(n, np.uint8)
        for p in range(P2):
            nrow, M = int(NROW[p]), int(self.M[S][p])
            if not nrow or not M: continue
            base = int(self.BASE[S][p]); w = W[p]; w1, w2 = w & full, w >> Nc; rank = np.arange(nrow, dtype=np.int64)
            for col, c in enumerate(self.cols[S][p]):
                st = self.stab[S][int(c)]; rmin = rank; cnt = np.ones(nrow, np.int64)
                if len(st) > 1:
                    rmin = rank.copy(); cnt = np.zeros(nrow, np.int64)
                    for g, f in st:
                        rh = OFFT[p][GT[1, g][w2] ^ (full*f)] + IDX[GT[0, g][w1] ^ (full*f)]
                        rmin = np.minimum(rmin, rh); cnt += rh == rank
                sl = slice(base+col, base+nrow*M, M)
                canon[sl] = rmin == rank; ecan[sl] = base+rmin*M+col; ssz[sl] = cnt
        self.CE = np.flatnonzero(canon).astype(np.int32); self.dim = len(self.CE)
        cid = np.cumsum(canon, dtype=np.int64)-1; self.cidx_entry = cid[ecan].astype(np.int32); del cid, ecan, canon
        self.sqrtn = np.sqrt(2.0*nG/ssz[self.CE]); del ssz
        log(f'compact basis {time.perf_counter()-t0:.1f} s; dimension {self.dim}')
        # pack tables P[S] (layout entry -> compact index) and unpack tables U[S] (compact index -> layout entry)
        self.P, self.U = [None]*3, [None]*3
        self.P[2] = self.cidx_entry; self.U[2] = self.CE
        wABC_c = self._decode(2, self.CE)                       # words (A, B, C) of the compact representatives
        for S in (0, 1):
            P = np.zeros(self.nexp[S], np.int32)
            for p in range(P2):
                nrow, M = int(NROW[p]), int(self.M[S][p])
                if not nrow or not M: continue
                base = int(self.BASE[S][p]); e = np.arange(base, base+nrow*M, dtype=np.int64)
                P[base:base+nrow*M] = self.cidx_entry[self._entry(2, *self._decode(S, e))]
            self.P[S] = P; self.U[S] = self._entry(S, *wABC_c).astype(np.int32)
        self.rep_words = wABC_c
        log(f'pack/unpack tables {time.perf_counter()-t0:.1f} s'); self.setup_seconds = time.perf_counter()-t0

    def _masks(self):
        """Pair operators: bonds between the two active sublattices of each layout, as masks on (lo | hi << Nc)."""
        cl, Nc = self.cl, self.Nc
        self.MASK = []
        for S in range(3):
            S1, S2 = (S+1) % 3, (S+2) % 3; ms = []
            for a, b in cl.bonds(1):
                sa, sb = a//Nc, b//Nc
                if {sa, sb} == {S1, S2}:
                    i1, i2 = (a, b) if sa == S1 else (b, a)
                    ms.append((1 << (i1-S1*Nc)) | (1 << (Nc+i2-S2*Nc)))
            self.MASK.append(np.array(ms, np.int64))
        assert sum(len(m) for m in self.MASK) == len(cl.bonds(1))

    def _entry(self, S, wA, wB, wC):
        """Layout-S entry index of the states with sublattice words (wA, wB, wC): bring the spectator word to its representative."""
        ws = (wA, wB, wC); S1, S2 = (S+1) % 3, (S+2) % 3; full = self.full
        g, f, r = self.GI[S][ws[S]], self.GF[S][ws[S]], self.REP[S][ws[S]]
        v1 = self.GT[S1][g, ws[S1]] ^ (full*f); v2 = self.GT[S2][g, ws[S2]] ^ (full*f); p = self.half-self.pc[r]
        return self.BASE[S][p] + (self.OFFT[p, v2]+self.IDX[v1])*self.M[S][p] + self.COL[S][r]

    def _decode(self, S, e):
        """Sublattice words (wA, wB, wC) of layout-S entries e."""
        e = np.asarray(e, np.int64); p = np.searchsorted(self.BASE[S], e, side='right')-1
        out = [np.zeros(len(e), np.int64) for _ in range(3)]; S1, S2 = (S+1) % 3, (S+2) % 3
        for q in np.unique(p):
            m = p == q; M = int(self.M[S][q]); loc = e[m]-int(self.BASE[S][q]); w = self.W[q][loc//M]
            out[S1][m] = w & self.full; out[S2][m] = w >> self.Nc; out[S][m] = self.cols[S][q][loc % M]
        return out

    # ------------------------------------------------------------------ numpy product (reference, small clusters)
    def pair_np(self, S, X):
        Y = np.zeros_like(X); Nc, full = self.Nc, self.full
        for p in range(self.P2):
            nrow, M = int(self.NROW[p]), int(self.M[S][p])
            if not nrow or not M: continue
            b0 = int(self.BASE[S][p]); x = X[b0:b0+nrow*M].reshape(nrow, M); y = np.zeros_like(x); w = self.W[p]
            anti_n = np.zeros(nrow, np.int64)
            for m in self.MASK[S]:
                t = w & m; a = (t != 0) & (t != m); v = w[a] ^ m
                y[a] += 0.5*self.J*x[self.OFFT[p][v >> Nc]+self.IDX[v & full]]; anti_n += a
            y += (0.25*self.J*(len(self.MASK[S])-2*anti_n))[:, None]*x
            Y[b0:b0+nrow*M] = y.ravel()
        return Y

    def matvec_np(self, x):
        xs = x/self.sqrtn; acc = np.zeros_like(x)
        for S in range(3):
            acc += self.pair_np(S, xs[self.P[S]])[self.U[S]]
        return acc*self.sqrtn

    def matvec_full(self, x):
        """The same product through the full S_z = 0 space (no layouts): for checking."""
        N, Nc, full = self.N, self.Nc, self.full
        s = np.arange(1 << N, dtype=np.int64); cnt = np.zeros(1 << N, np.int8)
        for i in range(N): cnt += ((s >> i) & 1).astype(np.int8)
        s = s[cnt == N//2]; del cnt
        psi = (x/self.sqrtn)[self.cidx_entry[self._entry(2, s & full, (s >> Nc) & full, s >> (2*Nc))]]
        out = np.zeros_like(psi)
        for a, b in self.cl.bonds(1):
            anti = ((s >> a) ^ (s >> b)) & 1 == 1
            out += np.where(anti, -0.25, 0.25)*self.J*psi
            out[anti] += 0.5*self.J*psi[np.searchsorted(s, s[anti] ^ ((1 << a) | (1 << b)))]
        wA, wB, wC = self.rep_words; sr = wA | (wB << Nc) | (wC << (2*Nc))
        return out[np.searchsorted(s, sr)]*self.sqrtn

    # ------------------------------------------------------------------ GPU product
    def to_gpu(self):
        import cupy as cp
        self.cp = cp; self.kern = cp.RawKernel(KERNEL, 'pair')
        self.gP = [cp.asarray(p) for p in self.P]; self.gU = [cp.asarray(u) for u in self.U]
        self.gsq = cp.asarray(self.sqrtn); self.gisq = 1.0/self.gsq
        self.gW = [cp.asarray(w.astype(np.uint32)) for w in self.W]; self.gOFF = [cp.asarray(o.astype(np.int32)) for o in self.OFFT]
        self.gIDX = cp.asarray(self.IDX.astype(np.int32)); self.gMASK = [cp.asarray(m.astype(np.uint32)) for m in self.MASK]
        n = max(self.nexp); self.gX = cp.empty(n, cp.float64); self.gY = cp.empty(n, cp.float64)
        self.gxs = cp.empty(self.dim, cp.float64); self.times = dict(pack=0.0, pair=0.0, unpack=0.0, scale=0.0)

    def matvec_gpu(self, x, y, timed=False):
        cp = self.cp; sync = cp.cuda.Stream.null.synchronize; T = self.times
        def tick(key, t):
            if timed: sync(); T[key] += time.perf_counter()-t
            return time.perf_counter()
        t = time.perf_counter(); cp.multiply(x, self.gisq, out=self.gxs); y.fill(0.0); t = tick('scale', t)
        for S in range(3):
            n = self.nexp[S]; X, Y = self.gX[:n], self.gY[:n]
            cp.take(self.gxs, self.gP[S], out=X); t = tick('pack', t)
            for p in range(self.P2):
                nrow, M = int(self.NROW[p]), int(self.M[S][p])
                if not nrow or not M: continue
                b0 = int(self.BASE[S][p]); cnt = nrow*M
                self.kern(((cnt+255)//256,), (256,), (X[b0:b0+cnt], Y[b0:b0+cnt], self.gW[p], self.gOFF[p], self.gIDX, self.gMASK[S],
                          np.int32(len(self.MASK[S])), np.int32(M), np.int64(cnt), np.int32(self.Nc), np.float64(self.J)))
            t = tick('pair', t)
            y += cp.take(Y, self.gU[S]); t = tick('unpack', t)
        y *= self.gsq; tick('scale', t)

def lanczos(mv, dim, cp, tol=1e-12, maxit=1000, seed=42):
    """Lowest level by the three-term recurrence; cp is numpy or cupy; mv(x, y) writes H x into y."""
    v = cp.asarray(np.random.default_rng(seed).standard_normal(dim)); v /= cp.linalg.norm(v); vp = cp.zeros_like(v); w = cp.empty_like(v)
    al, be, E, beta = [], [], None, 0.0
    for it in range(1, maxit+1):
        mv(v, w); a = float(cp.dot(v, w)); w -= a*v; w -= beta*vp; beta = float(cp.linalg.norm(w)); al.append(a)
        e = float(np.linalg.eigvalsh(np.diag(al)+np.diag(be, 1)+np.diag(be, -1))[0]) if it > 1 else a
        if E is not None and abs(e-E) < tol*abs(e): return e, it
        E = e; be.append(beta); vp, v = v, w/beta; w = cp.empty_like(v)
    return E, maxit


KERNELS = r'''
struct Tab { const int* GT; const int* REP; const int* GIF; const int* COL; const int* OFFT; const int* IDX; };

extern "C" __global__ void pack(const double* x, double* X, const unsigned int* W, const int* COLS,
        const int* GT, const int* REP, const int* GIF, const int* COL, const int* OFFT, const int* IDX,
        const int* STN, const int* STOFF, const int* STLIST, const long long* CB, const long long* BOFF, const int* TRIV,
        const unsigned long long* BITS, const long long* RANKC,
        int S, int M, long long n, int Nc, int nG, int half, int maxM) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    long long r = i / M; int col = (int)(i - r * M);
    const int NW = 1 << Nc; const unsigned int full = NW - 1;
    unsigned int w = W[r], a[3]; int S1 = (S + 1) % 3, S2 = (S + 2) % 3;
    a[S] = (unsigned int)COLS[col]; a[S1] = w & full; a[S2] = w >> Nc;
    unsigned int c = a[2]; int rep = REP[2 * NW + c], gf = GIF[2 * NW + c], g = gf & 255; unsigned int f = (gf >> 8) ? full : 0u;
    unsigned int v1 = (unsigned int)GT[(0 * nG + g) * NW + a[0]] ^ f, v2 = (unsigned int)GT[(1 * nG + g) * NW + a[1]] ^ f;
    int p = half - __popc((unsigned int)rep); const int* off = OFFT + (long long)p * NW;
    long long rank = (long long)off[v2] + IDX[v1], rmin = rank; int ns = STN[rep], ssz = 1;
    if (ns > 1) {
        ssz = 0; const int* sl = STLIST + STOFF[rep];
        for (int k = 0; k < ns; k++) {
            int h = sl[k] & 255; unsigned int hf = (sl[k] >> 8) ? full : 0u;
            unsigned int u1 = (unsigned int)GT[(0 * nG + h) * NW + v1] ^ hf, u2 = (unsigned int)GT[(1 * nG + h) * NW + v2] ^ hf;
            long long rh = (long long)off[u2] + IDX[u1];
            if (rh < rmin) rmin = rh;
            if (rh == rank) ssz++;
        }
    }
    int cc = p * maxM + COL[2 * NW + rep]; long long j;
    if (TRIV[cc]) j = CB[cc] + rmin;
    else { long long b = BOFF[cc] + (rmin >> 6); j = CB[cc] + RANKC[b] + __popcll(BITS[b] & ((1ull << (rmin & 63)) - 1ull)); }
    X[i] = x[j] * sqrt((double)ssz);
}

extern "C" __global__ void unpack(const double* Y, double* y, const unsigned int* WCAT, const long long* WOFF,
        const int* GT, const int* REP, const int* GIF, const int* COL, const int* OFFT, const int* IDX,
        const long long* CSTART, const int* BP, const int* BREP, const int* BTRIV, const long long* BSEL,
        const int* SEL, const unsigned char* SELSZ, const long long* BASE, const int* MS,
        int S, long long n, int nblocks, int Nc, int nG, int half) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return;
    int lo = 0, hi = nblocks - 1;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (CSTART[mid] <= r) lo = mid; else hi = mid - 1; }
    long long loc = r - CSTART[lo]; int p0 = BP[lo]; long long rank; double sc = 1.0;
    if (BTRIV[lo]) rank = loc; else { rank = SEL[BSEL[lo] + loc]; sc = rsqrt((double)SELSZ[BSEL[lo] + loc]); }
    const int NW = 1 << Nc; const unsigned int full = NW - 1;
    unsigned int w = WCAT[WOFF[p0] + rank], a[3]; a[0] = w & full; a[1] = w >> Nc; a[2] = (unsigned int)BREP[lo];
    int S1 = (S + 1) % 3, S2 = (S + 2) % 3;
    unsigned int c = a[S]; int rep = REP[S * NW + c], gf = GIF[S * NW + c], g = gf & 255; unsigned int f = (gf >> 8) ? full : 0u;
    unsigned int v1 = (unsigned int)GT[(S1 * nG + g) * NW + a[S1]] ^ f, v2 = (unsigned int)GT[(S2 * nG + g) * NW + a[S2]] ^ f;
    int q = half - __popc((unsigned int)rep);
    long long e = BASE[q] + ((long long)OFFT[(long long)q * NW + v2] + IDX[v1]) * MS[q] + COL[S * NW + rep];
    y[r] += Y[e] * sc;
}
'''

class Layout3Fly(Layout3):
    def __init__(self, A1, A2, J=1.0, log=lambda *a: None):
        super().__init__(A1, A2, J, log, tables=False)
        t0 = time.perf_counter(); Nc, full, P2 = self.Nc, self.full, self.P2; S = 2
        maxM = int(max(int(m.max()) for m in self.M)); self.maxM = maxM
        CB = np.zeros(P2*maxM, np.int64); BOFF = np.zeros(P2*maxM, np.int64); TRIV = np.ones(P2*maxM, np.int32)
        blocks, SEL, SELSZ, BITS, RANKC = [], [], [], [], []; start = nsel = nwords = 0
        for p in range(P2):
            nrow, M = int(self.NROW[p]), int(self.M[S][p])
            if not nrow or not M: continue
            w = self.W[p]; w1, w2 = w & full, w >> Nc; rank = np.arange(nrow, dtype=np.int64)
            for col, c in enumerate(self.cols[S][p]):
                st = self.stab[S][int(c)]; cc = p*maxM+col; CB[cc] = start
                if len(st) == 1:
                    blocks.append((start, p, int(c), 1, 0)); start += nrow; continue
                rmin = rank.copy(); cnt = np.zeros(nrow, np.int64)
                for g, f in st:
                    rh = self.OFFT[p][self.GT[1, g][w2] ^ (full*f)] + self.IDX[self.GT[0, g][w1] ^ (full*f)]
                    rmin = np.minimum(rmin, rh); cnt += rh == rank
                canon = rmin == rank; sel = np.flatnonzero(canon)
                nw = (nrow+63)//64; pad = np.zeros(nw*64, bool); pad[:nrow] = canon
                BITS.append(np.packbits(pad, bitorder='little').view('<u8')); cs = np.cumsum(pad, dtype=np.int64)
                RANKC.append(np.concatenate([[0], cs[63::64]])[:nw]); TRIV[cc] = 0; BOFF[cc] = nwords; nwords += nw
                SEL.append(sel.astype(np.int32)); SELSZ.append(cnt[sel].astype(np.uint8))
                blocks.append((start, p, int(c), 0, nsel)); start += len(sel); nsel += len(sel)
        self.dim = start; self.blocks = blocks
        cat = lambda L_, dt: np.concatenate(L_).astype(dt) if L_ else np.zeros(1, dt)
        self.t = dict(CB=CB, BOFF=BOFF, TRIV=TRIV, BITS=cat(BITS, np.uint64), RANKC=cat(RANKC, np.int64), SEL=cat(SEL, np.int32), SELSZ=cat(SELSZ, np.uint8),
                      CSTART=np.array([b[0] for b in blocks]+[start], np.int64), BP=np.array([b[1] for b in blocks], np.int32),
                      BREP=np.array([b[2] for b in blocks], np.int32), BTRIV=np.array([b[3] for b in blocks], np.int32),
                      BSEL=np.array([b[4] for b in blocks], np.int64))
        NW = 1 << Nc
        STN = np.zeros(NW, np.int32); STOFF = np.zeros(NW, np.int32); STL = []
        for r, st in self.stab[S].items():
            STN[r] = len(st); STOFF[r] = len(STL); STL += [g | (f << 8) for g, f in st]
        self.t.update(STN=STN, STOFF=STOFF, STLIST=np.array(STL, np.int32), GT=self.GT.astype(np.int32).ravel(),
                      REP=np.concatenate(self.REP).astype(np.int32), GIF=np.concatenate([g | (f << 8) for g, f in zip(self.GI, self.GF)]).astype(np.int32),
                      COL=np.concatenate(self.COL).astype(np.int32), OFFT=self.OFFT.astype(np.int32).ravel(), IDX=self.IDX.astype(np.int32),
                      WCAT=np.concatenate(self.W).astype(np.uint32), WOFF=np.concatenate([[0], np.cumsum([len(w) for w in self.W])]).astype(np.int64))
        self.nsel = nsel; self.setup_seconds += time.perf_counter()-t0
        log(f'compact basis by columns {time.perf_counter()-t0:.1f} s; dimension {self.dim}; rows in columns with a stabiliser {nsel}')

    def rep_states(self):
        """Sublattice words (A, B, C) of all compact rows, in the order of this class (numpy; for checks)."""
        out = [np.zeros(self.dim, np.int64) for _ in range(3)]; t = self.t
        for b, (start, p, c, triv, so) in enumerate(self.blocks):
            n = int(t['CSTART'][b+1]-start); rank = np.arange(n) if triv else t['SEL'][so:so+n]
            w = self.W[p][rank]; out[0][start:start+n] = w & self.full; out[1][start:start+n] = w >> self.Nc; out[2][start:start+n] = c
        return out

    def to_gpu(self):
        import cupy as cp
        self.cp = cp; mod = cp.RawModule(code=KERNEL + KERNELS, options=('--std=c++11',))
        self.kern, self.kpack, self.kunpack = mod.get_function('pair'), mod.get_function('pack'), mod.get_function('unpack')
        self.g = {k: cp.asarray(v) for k, v in self.t.items()}
        self.table_bytes = int(sum(v.nbytes for v in self.t.values()))
        self.gW = [cp.asarray(w.astype(np.uint32)) for w in self.W]; self.gOFF = [cp.asarray(o.astype(np.int32)) for o in self.OFFT]
        self.gIDX = self.g['IDX']; self.gMASK = [cp.asarray(m.astype(np.uint32)) for m in self.MASK]
        self.gCOLS = [[cp.asarray(c.astype(np.int32)) if len(c) else None for c in cs] for cs in self.cols]
        self.gBASE = [cp.asarray(b.astype(np.int64)) for b in self.BASE]; self.gM = [cp.asarray(m.astype(np.int32)) for m in self.M]
        n = max(self.nexp); self.gX = cp.empty(n, cp.float64); self.gY = cp.empty(n, cp.float64)
        self.times = dict(pack=0.0, pair=0.0, unpack=0.0, scale=0.0)

    def matvec_gpu(self, x, y, timed=False):
        cp = self.cp; sync = cp.cuda.Stream.null.synchronize; T = self.times; g = self.g; Nc = np.int32(self.Nc); nG = np.int32(self.nG); half = np.int32(self.half)
        def tick(key, t):
            if timed: sync(); T[key] += time.perf_counter()-t
            return time.perf_counter()
        t = time.perf_counter(); y.fill(0.0); t = tick('scale', t)
        for S in range(3):
            n = self.nexp[S]; X, Y = self.gX[:n], self.gY[:n]
            for p in range(self.P2):
                nrow, M = int(self.NROW[p]), int(self.M[S][p])
                if not nrow or not M: continue
                b0 = int(self.BASE[S][p]); cnt = nrow*M
                self.kpack(((cnt+255)//256,), (256,), (x, X[b0:b0+cnt], self.gW[p], self.gCOLS[S][p], g['GT'], g['REP'], g['GIF'], g['COL'], g['OFFT'], g['IDX'],
                           g['STN'], g['STOFF'], g['STLIST'], g['CB'], g['BOFF'], g['TRIV'], g['BITS'], g['RANKC'],
                           np.int32(S), np.int32(M), np.int64(cnt), Nc, nG, half, np.int32(self.maxM)))
            t = tick('pack', t)
            for p in range(self.P2):
                nrow, M = int(self.NROW[p]), int(self.M[S][p])
                if not nrow or not M: continue
                b0 = int(self.BASE[S][p]); cnt = nrow*M
                self.kern(((cnt+255)//256,), (256,), (X[b0:b0+cnt], Y[b0:b0+cnt], self.gW[p], self.gOFF[p], self.gIDX, self.gMASK[S],
                          np.int32(len(self.MASK[S])), np.int32(M), np.int64(cnt), Nc, np.float64(self.J)))
            t = tick('pair', t)
            self.kunpack(((self.dim+255)//256,), (256,), (Y, y, g['WCAT'], g['WOFF'], g['GT'], g['REP'], g['GIF'], g['COL'], g['OFFT'], g['IDX'],
                         g['CSTART'], g['BP'], g['BREP'], g['BTRIV'], g['BSEL'], g['SEL'], g['SELSZ'], self.gBASE[S], self.gM[S],
                         np.int32(S), np.int64(self.dim), np.int32(len(self.blocks)), Nc, nG, half))
            t = tick('unpack', t)


class Layout3Engine:
    """The three-layout product as an engine of one sector (see the module text for what it covers)."""
    CPU_MAX_DIM = 5_000_000

    def __init__(self, cl, J=1.0, device='gpu', log=lambda *a: None):
        if cl.kind != 'kagome':
            raise ValueError('layout3: kagome clusters only')
        A1, A2 = (int(cl.M[0, 0]), int(cl.M[1, 0])), (int(cl.M[0, 1]), int(cl.M[1, 1]))
        self.device = device; self.real = True
        if device == 'gpu':
            self.lay = Layout3Fly(A1, A2, J, log); self.lay.to_gpu(); self.xp = self.lay.cp
        else:
            self.lay = Layout3(A1, A2, J, log, tables=False)           # dimension first: the numpy reference is for small clusters
            if max(self.lay.nexp) > 3*self.CPU_MAX_DIM:
                raise ValueError('layout3: device=cpu is the reference for small clusters; use device=gpu')
            self.lay = Layout3(A1, A2, J, log); self.xp = np
        self.dim = self.lay.dim; self.N = self.lay.N; self.group_order = 2*self.lay.nG; self.expanded = max(self.lay.nexp)
        self.setup_seconds = self.lay.setup_seconds

    def matvec(self, x, y):
        if self.device == 'gpu':
            self.lay.matvec_gpu(x, y)
        else:
            y[:] = self.lay.matvec_np(x)

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42):
        """(lowest level, products, seconds)."""
        t = time.perf_counter(); E, it = lanczos(self.matvec, self.dim, self.xp, tol, maxit, seed)
        return E, it, time.perf_counter()-t

    def time_product(self, reps=3):
        xp = self.xp; x = xp.asarray(np.random.default_rng(0).standard_normal(self.dim)); y = xp.empty_like(x); best = math.inf
        sync = (lambda: xp.cuda.Stream.null.synchronize()) if self.device == 'gpu' else (lambda: None)
        for _ in range(reps+1):
            sync(); t = time.perf_counter(); self.matvec(x, y); sync(); best = min(best, time.perf_counter()-t)
        return best
