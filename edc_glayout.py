#!/usr/bin/env python3
"""Layout engine: H x of one symmetry block through layouts of the vector in which the same operator acts on columns that
are adjacent in memory (spin-1/2 models with two-site couplings, S_z conserved; CPU C/OpenMP or CUDA).

Construction.  The sites are split into p parts of equal size that the group G of the block permutes (the three
sublattices of the kagome and of the triangular lattice, the four sublattices of a 2 x 2 cell on the square lattice).
A bond touches one or two parts, so it lies inside a pair of parts and leaves the other p - 2 untouched.  With H_q the
bonds of the pair q, K_q the operations that map q onto itself and d_t one operation that brings the pair t of the
G-orbit of q onto q, a vector of the block (psi(g s) = chi(g)^* psi(s)) obeys
        (H psi)(r) = sum over orbit representatives q, sum over t in the orbit of q:  chi(d_t) (H_q psi)(d_t r).
The layout of q holds psi on the states whose spectator word (the p - 2 other parts) is the minimum of its K_q-orbit,
with every word of the two parts of q.  H_q is the same operator on every spectator column, and columns are adjacent in
memory, so it runs at the streaming rate of the device.  One product:
    expand   every layout entry reads its compact row (one gather, one phase);
    pair     the pair operator, class by class of the number of up spins in the pair;
    fold     every compact row reads one layout entry per pair of the orbit (a gather and a phase each).
The conversions are stored as index tables (4 bytes and a phase byte per entry; about 30 to 45 bytes per row in all),
built on the device from tables of 2^(spectator bits) entries.  The three-layout product of edc_layout3.py is the case
of three parts, k = 0, a group that keeps each part.

Interface: the subset of edc_sub.SubEngine that the solvers use (dim, real, xp, matvec, ground_energy, time_matvec,
batched, retune, operator), so edc_hubbard_sym.lanczos, edc_levels and edc_measure run on it unchanged.  The compact
basis is its own (rows are ordered as the home layout), so vectors cannot be exchanged with a SubEngine.
The fold has three forms (`fold=`): 'table' as above; 'fly' (GPU) computes the entry of a row in the kernel, so the fold
tables go; 'scatter' (GPU) adds every entry to its row with atomic additions, class by class, so both buffers shrink to
one class.  `expand='fly'` (GPU, with fold='scatter') stores no table per entry at all: the row and the code of every
entry of a class are computed in the kernel (representative search, then a rank in a bitmap of the representatives)
and used for the expansion and the fold of that class.  On a GPU the engine takes the fastest combination that fits in
the free device memory beside `reserve_vectors` vectors.  With the scatter fold a product is not bit-reproducible (the
order of the atomic additions varies), so a replayed Lanczos run can differ from the run;
edc_hubbard_sym.ground_state_vector checks the residual of the replayed vector and repairs it.  Limits: double precision; one device; fields are not written; one vector per pass; M-th roots of unity with
M <= 256 as characters; layouts of fewer than 2^32 entries.

Measured (one H100, kagome 36 sites (2,2),(-2,4)): 3 to 10 times the on-the-fly product of SubEngine, equal to
its stored connectivity where that fits whole, with a fifth of the memory.
"""
import time, math, ctypes
import numpy as np
import edc

CUDA = r'''
#include <cupy/complex.cuh>
typedef TYPE T;
extern "C" __global__ void pair(const T* X, T* Y, const unsigned int* W, const int* OFF, const int* IDX,
        const unsigned int* MASK, const double* JXY, const double* JZ, int nb, int M, long long n, int w, double dsum) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    long long r = i / M; int col = (int)(i - r * M);
    unsigned int wd = W[r], lom = (1u << w) - 1u; T acc = 0.0; double d = dsum;
    for (int b = 0; b < nb; b++) {
        unsigned int m = MASK[b], t = wd & m;
        if (t && t != m) { d -= 0.5 * JZ[b]; unsigned int v = wd ^ m; long long rr = (long long)OFF[v >> w] + IDX[v & lom]; acc += JXY[b] * X[rr * M + col]; }
    }
    Y[i] = d * X[i] + 0.5 * acc;
}
extern "C" __global__ void expand(const T* x, T* X, const unsigned int* P, const CODE_T* PC, const T* TE, long long n, long long dim) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    long long j = P[i];
    if (j < dim) X[i] = TE[PC[i]] * x[j]; else X[i] = 0.0;
}
extern "C" __global__ void fold(const T* Y, T* y, const unsigned int* F, const CODE_T* FC, const T* TF, long long n) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return;
    y[r] += TF[FC[r]] * Y[F[r]];
}
// rows and codes of the entries of one class without a stored table: the representative of the state's orbit is the
// smallest home entry among its images (one per pair of the home orbit, more where the spectator has a stabiliser);
// its row is a rank in the bitmap of the representatives
extern "C" __global__ void canon(unsigned int* P, CODE_T* PC, const unsigned int* W, const unsigned int* COLSQ, int M_, long long n,
        const int* WTT, const int* SRC, const int* MCODE, const int* KELH, const int* COLH, const int* STABNH, const int* STROWH, const int* STABLH,
        const long long* BASEH, const long long* MCH, const int* OFFT, const int* IDX, const unsigned long long* BITS, const unsigned int* RANK,
        const unsigned short* SG, const int* SIDX, const int* PAR) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const int p = PAR[0], w = PAR[1], nf = PAR[2], nup = PAR[3], M = PAR[4], tH = PAR[6], smax = PAR[7], ns = p - 2;
    const int* H = PAR + 8; const int* Q = PAR + 24; const int* DH = PAR + 40;
    const unsigned int wm = (1u << w) - 1u; const long long NW = 1ll << w;
    long long r = i / M_; int col = (int)(i - r * M_);
    unsigned int pw = W[r], spec = COLSQ[col], a[8], u[8], v[8], vv[8];
    a[Q[0]] = pw & wm; a[Q[1]] = pw >> w;
    for (int j = 0; j < ns; j++) a[Q[2 + j]] = (spec >> (j * w)) & wm;
    long long best = 0x7fffffffffffffffLL; int bcode = 0;
    for (int c = 0; c < tH; c++) {
        int d = DH[c], g0 = d / nf; unsigned int fl = (d % nf) ? wm : 0u;
        for (int t = 0; t < p; t++) u[t] = (unsigned int)WTT[((long long)g0 * p + t) * NW + a[SRC[g0 * p + t]]] ^ fl;
        unsigned int s = 0; for (int j = 0; j < ns; j++) s |= u[H[2 + j]] << (j * w);
        int kel = KELH[s]; g0 = kel / nf; fl = (kel % nf) ? wm : 0u;
        for (int t = 0; t < p; t++) v[t] = (unsigned int)WTT[((long long)g0 * p + t) * NW + u[SRC[g0 * p + t]]] ^ fl;
        s = 0; for (int j = 0; j < ns; j++) s |= v[H[2 + j]] << (j * w);
        int q2 = nup - __popc(s); long long mc = MCH[q2], base = BASEH[q2] + COLH[s]; const int* off = OFFT + (long long)q2 * NW;
        long long e = base + ((long long)off[v[H[1]]] + IDX[v[H[0]]]) * mc; int code = (MCODE[d] + MCODE[kel]) % M;
        if (e < best) { best = e; bcode = code; }
        int nst = STABNH[s];
        if (nst > 1) {
            const int* sl = STABLH + (long long)STROWH[s] * smax;
            for (int k = 1; k < nst; k++) {
                int h = sl[k]; g0 = h / nf; fl = (h % nf) ? wm : 0u;
                for (int t = 0; t < p; t++) vv[t] = (unsigned int)WTT[((long long)g0 * p + t) * NW + v[SRC[g0 * p + t]]] ^ fl;
                long long e2 = base + ((long long)off[vv[H[1]]] + IDX[vv[H[0]]]) * mc;
                if (e2 < best) { best = e2; bcode = (code + MCODE[h]) % M; }
            }
        }
    }
    unsigned long long word = BITS[best >> 6]; int bit = (int)(best & 63);
    if (!((word >> bit) & 1ull)) { P[i] = 0xFFFFFFFFu; PC[i] = 0; return; }
    unsigned int row = RANK[best >> 6] + (unsigned int)__popcll(word & ((1ull << bit) - 1ull));
    P[i] = row; PC[i] = (CODE_T)(bcode + M * SIDX[SG[row]]);
}
// the fold from the layout side, class by class: every entry adds its value to its row.  Several entries reach the
// same row (one per pair of the orbit, more where the spectator word has a stabiliser), hence atomic additions; the
// weight is sqrt(stabiliser of the row) / (stabiliser of the spectator word in K), with the conjugate phase.
extern "C" __global__ void scatter(const T* Yc, T* y, const unsigned int* P, const CODE_T* PC, const T* TE, const double* CW,
        int M, long long n, long long dim) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    long long j = P[i]; if (j >= dim) return;
    T v = (CONJ(TE[PC[i]]) * Yc[i]) * CW[i % M];
    ATOMIC_ADD(y + j, v);
}
// the fold without a table: the row's state from its home entry, the operation d, the spectator brought to its
// representative by one element of K (any K-equivalent entry holds the same value up to the phase), the entry index
extern "C" __global__ void foldfly(const T* Y, T* y, const unsigned int* CE, const unsigned short* SG, long long n,
        const long long* HBASE, const long long* HMC, const unsigned int* WCAT, const long long* WOFF, const unsigned int* HCOLS, const long long* HCOFF,
        const int* WTT, const int* SRC, const int* KEL, const int* COL, const long long* BASE, const long long* MC,
        const int* OFFT, const int* IDX, const int* MCODE, const T* PH, const int* PAR) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return;
    const int p = PAR[0], w = PAR[1], nf = PAR[2], nup = PAR[3], M = PAR[4], P2 = PAR[5], d = PAR[6], ns = p - 2;
    const int* H = PAR + 8; const int* Q = PAR + 24;
    const unsigned int wm = (1u << w) - 1u; const long long NW = 1ll << w;
    long long e = CE[r]; int n2 = P2 - 1; while (HBASE[n2] > e) n2--;
    long long loc = e - HBASE[n2], mc = HMC[n2], row = loc / mc, col = loc - row * mc;
    unsigned int pw = WCAT[WOFF[n2] + row], spec = HCOLS[HCOFF[n2] + col];
    unsigned int a[8], u[8];
    a[H[0]] = pw & wm; a[H[1]] = pw >> w;
    for (int j = 0; j < ns; j++) a[H[2 + j]] = (spec >> (j * w)) & wm;
    int g0 = d / nf; unsigned int fl = (d % nf) ? wm : 0u;
    for (int t = 0; t < p; t++) u[t] = (unsigned int)WTT[((long long)g0 * p + t) * NW + a[SRC[g0 * p + t]]] ^ fl;
    unsigned int s = 0; for (int j = 0; j < ns; j++) s |= u[Q[2 + j]] << (j * w);
    int kel = KEL[s]; g0 = kel / nf; fl = (kel % nf) ? wm : 0u;
    for (int t = 0; t < p; t++) a[t] = (unsigned int)WTT[((long long)g0 * p + t) * NW + u[SRC[g0 * p + t]]] ^ fl;
    s = 0; for (int j = 0; j < ns; j++) s |= a[Q[2 + j]] << (j * w);
    int q2 = nup - __popc(s);
    long long f = BASE[q2] + ((long long)OFFT[(long long)q2 * NW + a[Q[1]]] + IDX[a[Q[0]]]) * MC[q2] + COL[s];
    y[r] += (PH[(MCODE[d] + MCODE[kel]) % M] * Y[f]) * rsqrt((double)SG[r]);
}
'''

CSRC = r'''
#include <stdint.h>
#include <math.h>
#include <complex.h>
typedef TYPE T;
void pair(const T* X, T* Y, const uint32_t* W, const int32_t* OFF, const int32_t* IDX, const uint32_t* MASK,
          const double* JXY, const double* JZ, int nb, int64_t M, int64_t nrow, int w, double dsum) {
    const uint32_t lom = (1u << w) - 1u;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < nrow; r++) {
        uint32_t wd = W[r]; double d = dsum; T* y = Y + r * M; const T* x = X + r * M;
        for (int64_t c = 0; c < M; c++) y[c] = 0;
        for (int b = 0; b < nb; b++) {
            uint32_t m = MASK[b], t = wd & m;
            if (t && t != m) {
                d -= 0.5 * JZ[b]; uint32_t v = wd ^ m; const T* xs = X + ((int64_t)OFF[v >> w] + IDX[v & lom]) * M; const double c2 = 0.5 * JXY[b];
                for (int64_t c = 0; c < M; c++) y[c] += c2 * xs[c];
            }
        }
        for (int64_t c = 0; c < M; c++) y[c] += d * x[c];
    }
}
void expand(const T* x, T* X, const uint32_t* P, const CODE_T* PC, const T* TE, int64_t n, int64_t dim) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) { int64_t j = P[i]; if (j < dim) X[i] = TE[PC[i]] * x[j]; else X[i] = 0; }
}
void fold(const T* Y, T* y, const uint32_t* F, const CODE_T* FC, const T* TF, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n; r++) y[r] += TF[FC[r]] * Y[F[r]];
}
'''


def candidate_parts(cl, max_spectator_bits=24):
    """Partitions of the sites into parts of equal size: the basis sublattices times classes of cells (one class, the
    three classes of (x + y) or (x - y) mod 3, the four classes of a 2 x 2 cell).  Fewest parts first; at least three."""
    N = cl.N; nb = getattr(cl, 'nb', 1)
    pos = [cl.where(t) if hasattr(cl, 'where') else (0,)+tuple(cl.coord[t]) for t in range(N)]
    fns = [lambda x, y: 0, lambda x, y: (x+y) % 3, lambda x, y: (x+2*y) % 3, lambda x, y: (x % 2)+2*(y % 2)]
    out = []
    for f in fns:
        keys = sorted({(s, f(x, y)) for s, x, y in pos}); parts = [[t for t in range(N) if (pos[t][0], f(pos[t][1], pos[t][2])) == key] for key in keys]
        p = len(parts)
        if p >= 3 and len({len(q) for q in parts}) == 1 and (p-2)*len(parts[0]) <= max_spectator_bits and 2*len(parts[0]) <= 30:
            out.append(parts)
    return sorted(out, key=len)


def valid_parts(parts, perms):
    part_of = {t: s for s, q in enumerate(parts) for t in q}
    return all(len({part_of[p[t]] for t in q}) == 1 for p in perms for q in parts)


class _Lay:
    pass


class LayoutEngine:
    def __init__(self, model, nup, cl, k, irrep, flip=0, device='cpu', point_ops=None, parts=None, chunk=None, log=None, dry=False, fold=None, reserve_vectors=4, expand=None):
        t0 = time.perf_counter(); log = log or (lambda *a: None); self._log = log
        N = model.n; assert flip in (0, 1, -1) and (flip == 0 or 2*nup == N)
        assert not np.any(model.h), 'layout engine: fields are not written'
        if point_ops is None:
            perms, chars, ops = cl.group(k, irrep)
        elif hasattr(cl, 'irrep_table'):
            perms, chars, ops = cl.group(k, irrep, allowed=point_ops)
        else:
            import edc_lattice as el
            perms, chars, ops = cl.group(k, irrep, [o for o in cl.little_group(k) if o in el.IRREPS[irrep] and o in point_ops])
        perms = np.array(perms, np.int64); chars = np.array(chars, complex); ng0 = len(perms)
        assert (perms[0] == np.arange(N)).all()
        if parts is None:
            parts = next((q for q in candidate_parts(cl) if valid_parts(q, perms)), None)
            if parts is None:
                raise ValueError('layout engine: no partition of the sites into three or more parts that this group permutes')
        else:
            assert valid_parts(parts, perms), 'the group does not permute these parts'
        p, w = len(parts), len(parts[0]); self.p, self.w, self.N, self.nup, self.flip = p, w, N, nup, flip
        self.device = device; self.reserve_vectors = reserve_vectors
        self.expand = expand or 'table'; self.fold = fold or ('scatter' if self.expand == 'fly' else 'table')
        self._auto = fold is None and expand is None and device == 'gpu'
        assert self.expand in ('table', 'fly') and not (self.expand == 'fly' and (device != 'gpu' or self.fold != 'scatter')), 'expand=fly: GPU, fold=scatter'
        assert self.fold in ('fly', 'table', 'scatter') and not (self.fold != 'table' and device != 'gpu'), 'fold=fly or scatter: GPU only'
        assert p <= 8
        if device == 'gpu':
            import cupy as xp
        else:
            xp = np
        self.xp = xp; self.chunk = chunk or (1 << 24 if device == 'gpu' else 1 << 20)
        NW = 1 << w; wm = NW-1; self.NW = NW; nf = 2 if flip else 1; nG = ng0*nf; self.nf, self.nG, self.group_order = nf, nG, nG
        ch = np.array([chars[g//nf]*(float(flip) if (g % nf) else 1.0) for g in range(nG)])
        M = next((m for m in range(1, 257) if np.abs(ch**m-1).max() < 1e-8), None)
        assert M is not None, 'characters are not roots of unity of order <= 256'
        self.Mroot = M; mcode = np.round(np.angle(ch)/(2*np.pi)*M).astype(np.int64) % M
        assert np.abs(np.exp(2j*np.pi*mcode/M)-ch).max() < 1e-8
        self.real = M <= 2; self.dtype = np.float64 if self.real else np.complex128
        PH = np.exp(2j*np.pi*np.arange(M)/M); self._PH = (PH.real.copy() if self.real else PH).astype(self.dtype)
        self._mcode = mcode; self.mcode = xp.asarray(mcode)
        part_of = np.zeros(N, np.int64); bit_of = np.zeros(N, np.int64)
        for s, q in enumerate(parts):
            for b, t in enumerate(q):
                part_of[t] = s; bit_of[t] = b
        self._part_of, self._bit_of, self.parts = part_of, bit_of, [list(q) for q in parts]
        # action of the operations on the words of the parts
        words = np.arange(NW, dtype=np.int64)
        PI = np.array([[part_of[perms[g, parts[s][0]]] for s in range(p)] for g in range(ng0)])
        SRC = np.zeros((ng0, p), np.int64); WTT = np.zeros((ng0, p, NW), np.int64)
        for g in range(ng0):
            for s in range(p):
                T = PI[g, s]; SRC[g, T] = s; img = np.zeros(NW, np.int64)
                for c in range(w):
                    img |= ((words >> c) & 1) << bit_of[perms[g, parts[s][c]]]
                WTT[g, T] = img
        self._PI, self._SRC, self._WTT = PI, SRC, WTT
        self.SRC = xp.asarray(SRC); self.WTT = xp.asarray(WTT.reshape(-1))
        # ranks of the pair words
        pc = np.zeros(NW, np.int64)
        for c in range(w):
            pc += (words >> c) & 1
        IDX = np.zeros(NW, np.int64); LOWS = []
        for n in range(w+1):
            m = pc == n; IDX[m] = np.arange(int(m.sum())); LOWS.append(words[m])
        comb = np.array([math.comb(w, n) for n in range(w+1)], np.int64); P2 = 2*w+1; self.P2 = P2
        OFFT = np.zeros((P2, NW), np.int64); NROW = np.zeros(P2, np.int64); W = []
        for n2 in range(P2):
            nlo = n2-pc; ok = (nlo >= 0) & (nlo <= w); c = np.where(ok, comb[np.clip(nlo, 0, w)], 0)
            OFFT[n2] = np.concatenate([[0], np.cumsum(c)[:-1]]); NROW[n2] = c.sum()
            W.append(np.concatenate([LOWS[n2-pc[h]] | (h << w) for h in range(NW) if ok[h]]) if NROW[n2] else np.zeros(0, np.int64))
        self._pc, self._NROW = pc, NROW
        self.IDX = xp.asarray(IDX); self.OFFT = xp.asarray(OFFT.reshape(-1))
        self.WCAT = xp.asarray(np.concatenate(W)); self.WOFF = xp.asarray(np.concatenate([[0], np.cumsum([len(x) for x in W])]))
        self._kW = [self._dev(x.astype(np.uint32)) for x in W]; self._kOFF = [self._dev(o.astype(np.int32)) for o in OFFT]
        self._kIDX = self._dev(IDX.astype(np.int32)); self._kPH = self._dev(self._PH)
        # pairs of parts and their orbits
        pairs = [(a, b) for a in range(p) for b in range(a+1, p)]; self._pair_rep, self._orbit = {}, {}
        for q in pairs:
            if q in self._pair_rep:
                continue
            orb = sorted({tuple(sorted((int(PI[g, q[0]]), int(PI[g, q[1]])))) for g in range(ng0)})
            for t in orb:
                self._pair_rep[t] = orb[0]
            self._orbit[orb[0]] = orb
        self._lays = {}; self._buf = None; self._OFFT_np = OFFT
        self._bs = self._bondsets(model)
        if not self._bs:
            raise ValueError('layout engine: the model has no couplings')
        self._home = self._tables(sorted(self._bs)[0]); H = self._home
        if dry:                                                  # sizes only: the layouts of the model, no basis
            for q in sorted(self._bs):
                self._tables(q)
            self.dim = None; return
        # compact basis: of the home entries of a G-orbit the one with the smallest index; orbits with a non-trivial
        # character on their stabiliser are left out
        E0 = H.nexp; chunk = self.chunk
        if self._auto:                                           # what fits on the device decides how much is stored
            xp.get_default_memory_pool().free_all_blocks(); self.expand, self.fold = self._choose_mode()
        if self.expand == 'fly':                                 # no table per entry: a bitmap of the representatives and its ranks
            bits, rank, sgl = [], [], []; dim = 0; ch = max(64, (chunk//64)*64); sh = xp.arange(64, dtype=np.uint64)
            for a in range(0, E0, ch):
                e = xp.arange(a, min(a+ch, E0), dtype=np.int64); m, code, cnt, bad = self._canon(self._decode(H, e)); ok = (m == e) & ~bad
                n64 = (len(e)+63)//64; pad = xp.zeros(n64*64, np.uint64); pad[:len(e)] = ok; pad = pad.reshape(n64, 64)
                per = pad.sum(axis=1, dtype=np.int64); c = xp.cumsum(per)
                bits.append((pad << sh).sum(axis=1, dtype=np.uint64)); rank.append((dim+c-per).astype(np.uint32))
                sgl.append(cnt[ok].astype(np.uint16)); dim += int(c[-1])
            assert dim < 2**32-1, 'more than 2^32 rows'
            self.dim = dim; self._BITS = xp.concatenate(bits); self._RANK = xp.concatenate(rank); self._SG = xp.concatenate(sgl)
            self._CE = xp.zeros(0, np.uint32); del bits, rank, sgl; c0 = None
        else:
            m0 = xp.zeros(E0, np.uint32); c0 = xp.zeros(E0, np.uint8); ok = xp.zeros(E0, bool); sg = xp.zeros(E0, np.uint16)
            for a in range(0, E0, chunk):
                e = xp.arange(a, min(a+chunk, E0), dtype=np.int64); m, code, cnt, bad = self._canon(self._decode(H, e))
                m0[a:a+len(e)] = m; c0[a:a+len(e)] = code; ok[a:a+len(e)] = (m == e) & ~bad; sg[a:a+len(e)] = cnt
            cid = xp.empty(E0, np.uint32); dim = 0; ce = []                  # 0xFFFFFFFF marks an entry without a row
            for a in range(0, E0, chunk):
                o = ok[a:a+chunk]; c = xp.cumsum(o, dtype=np.int64); cid[a:a+chunk] = xp.where(o, c-1+dim, 0xFFFFFFFF); dim += int(c[-1])
                ce.append((xp.flatnonzero(o)+a).astype(np.uint32))
            assert dim < 2**32-1, 'more than 2^32 rows'
            self.dim = dim; self._CE = xp.concatenate(ce) if ce else xp.zeros(0, np.uint32); del ce, ok
            self._SG = xp.empty(dim, np.uint16); H.P = xp.empty(E0, np.uint32)
            for a in range(0, dim, chunk):
                self._SG[a:a+chunk] = sg[self._CE[a:a+chunk].astype(np.int64)]
            for a in range(0, E0, chunk):
                H.P[a:a+chunk] = cid[m0[a:a+chunk].astype(np.int64)]
            del m0, sg, cid
        # codes: phase index + M * (index of the stabiliser size of the row); the tables hold phase * sqrt(size) for
        # the expansion and phase / sqrt(size) for the fold, so the kernels read no norm factor
        cnts = np.zeros(65536, np.int64)
        for a in range(0, dim, chunk):
            b = xp.bincount(self._SG[a:a+chunk].astype(np.int64), minlength=65536); cnts += b.get() if xp is not np else b
        sizes = np.flatnonzero(cnts); sidx = np.zeros(65536, np.int64); sidx[sizes] = np.arange(len(sizes)); self._SIDX = xp.asarray(sidx)
        self._ctype = np.uint8 if M*max(len(sizes), 1) <= 256 else np.uint16; assert M*max(len(sizes), 1) <= 65536
        ph = np.exp(2j*np.pi*np.arange(M)/M); sq = np.sqrt(sizes.astype(float)) if len(sizes) else np.ones(1)
        te, tf = np.outer(sq, ph).ravel(), np.outer(1.0/sq, ph).ravel()
        self._TE, self._TF = (self._dev((t.real if self.real else t).astype(self.dtype)) for t in (te, tf))
        if self.expand == 'table':
            H.PC = xp.zeros(E0, self._ctype)
            for a in range(0, E0, chunk):
                H.PC[a:a+chunk] = self._code(c0[a:a+chunk], H.P[a:a+chunk])
        del c0
        if device == 'gpu':
            src = CUDA.replace('TYPE', 'double' if self.real else 'complex<double>').replace('CODE_T', 'unsigned char' if self._ctype is np.uint8 else 'unsigned short')
            src = src.replace('CONJ(', '(' if self.real else 'conj(')
            src = src.replace('ATOMIC_ADD(y + j, v);', 'atomicAdd(y + j, v);' if self.real else
                              'atomicAdd((double*)(y + j), v.real()); atomicAdd((double*)(y + j) + 1, v.imag());')
            mod = xp.RawModule(code=src, options=('-std=c++17',))
            self._kp, self._ke, self._kf = mod.get_function('pair'), mod.get_function('expand'), mod.get_function('fold')
            self._kff = mod.get_function('foldfly'); self._ks = mod.get_function('scatter'); self._kc = mod.get_function('canon')
            self._g = dict(WTT=xp.asarray(self._WTT.reshape(-1).astype(np.int32)), SRC=xp.asarray(self._SRC.reshape(-1).astype(np.int32)),
                           MCODE=xp.asarray(self._mcode.astype(np.int32)), WCAT=self.WCAT.astype(np.uint32), WOFF=self.WOFF.astype(np.int64),
                           OFFT=xp.asarray(self._OFFT_np.reshape(-1).astype(np.int32)), PH=self._kPH, SIDX=self._SIDX.astype(np.int32))
        else:
            src = CSRC.replace('TYPE', 'double' if self.real else 'double _Complex').replace('CODE_T', 'uint8_t' if self._ctype is np.uint8 else 'uint16_t')
            self._clib, _, _ = edc.compile_source(src, 'glay')
        log(f'layout engine: {p} parts of {w} sites, group {nG}, dimension {dim:,d}; home layout {H.nexp:,d} entries; {time.perf_counter()-t0:.1f} s')
        for q in sorted(self._bs):
            self._layout(q)
        if device == 'gpu':
            xp.get_default_memory_pool().free_all_blocks()
        self.setup_seconds = time.perf_counter()-t0

    # ------------------------------------------------------------------ helpers
    def _dev(self, a):
        return self.xp.asarray(a)

    def _choose_mode(self):
        """(expand, fold): the fastest combination whose tables and buffers fit in the free device memory beside
        reserve_vectors vectors of the block.  (table, table) is the fastest; (table, fly) has no fold tables; (table,
        scatter) holds both buffers class by class; (fly, scatter) stores no table per entry at all.  Decided before the
        basis is built, from the layout sizes: the dimension is at most about (entries of a layout) / (pairs in its orbit).
        Measured on one H100: fold fly 6 to 13 % slower than table, scatter 20 to 32 %."""
        xp = self.xp; vs = 8 if self.real else 16; cs = 5
        lays = [self._tables(q) for q in sorted(self._bs)]; E0 = self._home.nexp; Et = sum(l.nexp for l in lays)
        dim = min(l.nexp//len(l.D) for l in lays)
        cmax = max(int((self._NROW*l._Mc).max()) for l in lays); ymax = max(l.nexp for l in lays); t = sum(len(l.D) for l in lays)
        free = xp.cuda.Device().mem_info[0] + xp.get_default_memory_pool().free_bytes()
        res = self.reserve_vectors*dim*vs; rows = 6*dim
        need = {('table', 'table'): max(Et*cs + t*dim*cs + (cmax+ymax)*vs + rows + res, 17*E0),
                ('table', 'fly'): max(Et*cs + (cmax+ymax)*vs + rows + res, 17*E0),
                ('table', 'scatter'): max(Et*cs + 2*cmax*vs + rows + res, 17*E0),
                ('fly', 'scatter'): 2*cmax*vs + cmax*cs + E0//5 + 2*dim + res}
        self.mode_need = {'/'.join(k): int(v) for k, v in need.items()}; self.mode_free = int(free)
        return next((k for k in need if need[k] <= 0.92*free), ('fly', 'scatter'))

    def _code(self, phase, rows):
        """Phase codes and compact rows (0xFFFFFFFF = none) -> codes that also carry the stabiliser size of the row."""
        xp = self.xp; r = rows.astype(np.int64); ok = r < self.dim
        sg = self._SG[xp.where(ok, r, 0)].astype(np.int64) if self.dim else xp.zeros(len(r), np.int64)
        return (phase.astype(np.int64) + self.Mroot*xp.where(ok, self._SIDX[sg], 0)).astype(self._ctype)

    @property
    def stored_bytes(self):
        if self.expand == 'fly':
            return int(self._BITS.nbytes + self._RANK.nbytes + self._SG.nbytes)
        return int(self._CE.nbytes + self._SG.nbytes + sum(l.P.nbytes+l.PC.nbytes+sum(f.nbytes for f in l.F)+sum(f.nbytes for f in l.FC)
                                                           for l in self._lays.values() if hasattr(l, 'F')))

    @property
    def expanded(self):
        return {q: l.nexp for q, l in self._lays.items()}

    def _bondsets(self, model):
        """Bonds of the model by pair of parts (a bond inside one part is shared equally among the pairs that contain
        the part); only the orbit representatives are kept, and their images must add up to the whole model."""
        p, w, part_of, bit_of = self.p, self.w, self._part_of, self._bit_of; acc = {}; tot = np.zeros(2)
        for (i, j), (jxy, jz) in model.bonds.items():
            if not (jxy or jz):
                continue
            tot += (jxy, jz); s, t = int(part_of[i]), int(part_of[j])
            targets = [((min(s, t), max(s, t)), 1.0)] if s != t else [((min(s, u), max(s, u)), 1.0/(p-1)) for u in range(p) if u != s]
            for q, wt in targets:
                bit = lambda site: int(bit_of[site]) + (0 if part_of[site] == q[0] else w)
                e = acc.setdefault(q, {}).setdefault((1 << bit(i)) | (1 << bit(j)), [0.0, 0.0]); e[0] += wt*jxy; e[1] += wt*jz
        sets = {}; cov = np.zeros(2)
        for q, d in acc.items():
            if self._pair_rep[q] == q:
                ms = sorted(d); jx = np.array([d[m][0] for m in ms]); jz = np.array([d[m][1] for m in ms])
                sets[q] = dict(MASK=self._dev(np.array(ms, np.uint32)), JXY=self._dev(jx), JZ=self._dev(jz), nb=len(ms), dsum=float(0.25*jz.sum()))
                cov += len(self._orbit[q])*np.array([jx.sum(), jz.sum()])
        assert np.allclose(cov, tot, atol=1e-9*max(1.0, np.abs(tot).max())), 'the model is not invariant under the group of this block'
        return sets

    def _tables(self, q):
        """Layout of the pair q: spectator representatives under K_q, their columns, the bases of the classes."""
        if q in self._lays:
            return self._lays[q]
        xp, p, w, nf, NW, nup = self.xp, self.p, self.w, self.nf, self.NW, self.nup; PI, SRC, WTT = self._PI, self._SRC, self._WTT
        lay = _Lay(); lay.q = q; lay.sp = [t for t in range(p) if t not in q]; ns = len(lay.sp); NS = 1 << (ns*w); fullS = NS-1
        ng0 = len(PI); K = [g for g in range(self.nG) if {int(PI[g//nf, q[0]]), int(PI[g//nf, q[1]])} == set(q)]; assert K[0] == 0
        lay.nK = len(K)
        lay.D = [next(g0*nf for g0 in range(ng0) if {int(PI[g0, t[0]]), int(PI[g0, t[1]])} == set(q)) for t in self._orbit[q]]
        specs = np.arange(NS, dtype=np.int64); pw = [(specs >> (j*w)) & (NW-1) for j in range(ns)]

        def image(g):
            g0, f = divmod(g, nf); img = np.zeros(NS, np.int64)
            for j, T in enumerate(lay.sp):
                img |= WTT[g0, T][pw[lay.sp.index(int(SRC[g0, T]))]] << (j*w)
            return img ^ (fullS*f)
        best = np.full(NS, 1 << 62, np.int64); kel = np.zeros(NS, np.int64)
        for g in K:
            img = image(g); m = img < best; best[m] = img[m]; kel[m] = g
        reps = np.unique(best); fixed = np.zeros((len(K), len(reps)), bool)
        for i, g in enumerate(K):
            fixed[i] = image(g)[reps] == reps
        nst = fixed.sum(axis=0); withst = np.flatnonzero(nst > 1); smax = int(nst.max()) if len(reps) else 1
        STABN = np.ones(NS, np.int64); STROW = np.zeros(NS, np.int64); STABL = np.zeros((max(len(withst), 1), smax), np.int64); Ka = np.array(K)
        for row, j in enumerate(withst):
            st = Ka[fixed[:, j]]; STABN[reps[j]] = len(st); STROW[reps[j]] = row; STABL[row, :len(st)] = st
        pcs = np.zeros(NS, np.int64)
        for c in range(ns*w):
            pcs += (specs >> c) & 1
        COL = np.zeros(NS, np.int64); Mc = np.zeros(self.P2, np.int64); cols = []
        for n2 in range(self.P2):
            c = reps[pcs[reps] == nup-n2]; cols.append(c); Mc[n2] = len(c); COL[c] = np.arange(len(c))
        BASE = np.concatenate([[0], np.cumsum(self._NROW*Mc)]); lay.nexp = int(BASE[-1]); assert lay.nexp < 2**32, 'a layout with more than 2^32 entries'
        lay._Mc, lay._BASE, lay.smax = Mc, BASE, smax
        lay.REP, lay.KEL, lay.STABN, lay.STROW, lay.STABL = (xp.asarray(x) for x in (best, kel, STABN, STROW, STABL))
        lay.PCS, lay.COL, lay.Mc, lay.BASE = (xp.asarray(x) for x in (pcs, COL, Mc, BASE))
        lay.COLS = xp.asarray(np.concatenate(cols) if lay.nexp else np.zeros(1, np.int64))
        lay._COFF = np.concatenate([[0], np.cumsum([len(c) for c in cols])]); lay.COFF = xp.asarray(lay._COFF)
        self._lays[q] = lay
        return lay

    def _layout(self, q):
        """The conversion tables of the layout of q (expand: entry -> row; fold: row -> entry, per pair of the orbit)."""
        lay = self._tables(q)
        if hasattr(lay, 'F'):
            return lay
        xp, chunk, dim, M, H = self.xp, self.chunk, self.dim, self.Mroot, self._home
        if self.expand == 'fly':
            assert len(H.D) <= 8, 'expand=fly: at most eight pairs in the orbit of the home pair'
            lay.F, lay.FC = [], []; st = lay.STABN.astype(np.float64); pad = lambda v, n: list(v) + [0]*(n-len(v))
            lay.CW = [1.0/st[lay.COLS[int(lay._COFF[n2]):int(lay._COFF[n2+1])]] if int(lay._Mc[n2]) else None for n2 in range(self.P2)]
            lay.PARC = xp.asarray(np.array([self.p, self.w, self.nf, self.nup, M, self.P2, len(H.D), H.smax] + pad([H.q[0], H.q[1]] + H.sp, 16)
                                           + pad([lay.q[0], lay.q[1]] + lay.sp, 16) + pad(H.D, 8), np.int32))
            for L_ in (lay, H):
                if not hasattr(L_, 'g'):
                    L_.g = dict(KEL=L_.KEL.astype(np.int32), COL=L_.COL.astype(np.int32), BASE=L_.BASE.astype(np.int64), MC=L_.Mc.astype(np.int64),
                                COLS=L_.COLS.astype(np.uint32), COFF=L_.COFF.astype(np.int64), STABN=L_.STABN.astype(np.int32),
                                STROW=L_.STROW.astype(np.int32), STABL=L_.STABL.reshape(-1).astype(np.int32))
            cmax = int((self._NROW*lay._Mc).max())
            if self._buf is None or self._buf[0].size < cmax:
                self._buf = self._pbuf = None
                self._buf = (xp.empty(cmax, self.dtype), xp.empty(cmax, self.dtype)); self._pbuf = (xp.empty(cmax, np.uint32), xp.empty(cmax, self._ctype))
            return lay
        if lay is not H:
            E = lay.nexp; lay.P = xp.zeros(E, np.uint32); lay.PC = xp.zeros(E, self._ctype)
            for a in range(0, E, chunk):
                e = xp.arange(a, min(a+chunk, E), dtype=np.int64); m, code, cnt, bad = self._canon(self._decode(lay, e))
                rows = H.P[m]; lay.P[a:a+len(e)] = rows; lay.PC[a:a+len(e)] = self._code(code, rows)
        if self.fold == 'table':
            lay.F = [xp.zeros(dim, np.uint32) for _ in lay.D]; lay.FC = [xp.zeros(dim, self._ctype) for _ in lay.D]
            for a in range(0, dim, chunk):
                wd = self._decode(H, self._CE[a:a+chunk].astype(np.int64))
                for i, d in enumerate(lay.D):
                    u = self._apply_el(d, wd); kel = lay.KEL[self._spec(lay, u)]; v = self._apply_var(kel, u)
                    lay.F[i][a:a+chunk] = self._entry(lay, v)
                    lay.FC[i][a:a+chunk] = ((int(self._mcode[d]) + self.mcode[kel]) % M + M*self._SIDX[self._SG[a:a+chunk].astype(np.int64)]).astype(self._ctype)
        elif self.fold == 'scatter':
            lay.F, lay.FC = [], []; st = lay.STABN.astype(np.float64)
            lay.CW = [1.0/st[lay.COLS[int(lay.COFF[n2]):int(lay.COFF[n2+1])]] if int(lay._Mc[n2]) else None for n2 in range(self.P2)]
        else:
            lay.F, lay.FC = [], []; hp = [H.q[0], H.q[1]] + H.sp; qp = [lay.q[0], lay.q[1]] + lay.sp
            lay.PAR = [xp.asarray(np.array([self.p, self.w, self.nf, self.nup, M, self.P2, d, 0] + hp + [0]*(16-len(hp)) + qp + [0]*(16-len(qp)), np.int32))
                       for d in lay.D]
        if self.device == 'gpu':
            for L_ in (lay, H):
                if not hasattr(L_, 'g'):
                    L_.g = dict(KEL=L_.KEL.astype(np.int32), COL=L_.COL.astype(np.int32), BASE=L_.BASE.astype(np.int64), MC=L_.Mc.astype(np.int64),
                                COLS=L_.COLS.astype(np.uint32), COFF=L_.COFF.astype(np.int64))
        # buffers: the result of the pair operator for a whole layout; on a GPU its input class by class
        cmax = int((self._NROW*lay._Mc).max()) if self.device == 'gpu' else lay.nexp
        ymax = cmax if self.fold == 'scatter' else lay.nexp               # scatter: the result too is held class by class
        if self._buf is None or self._buf[1].size < ymax or self._buf[0].size < cmax:
            nx = max(cmax, self._buf[0].size if self._buf else 0); ny = max(ymax, self._buf[1].size if self._buf else 0)
            self._buf = None; self._buf = (xp.empty(nx, self.dtype), xp.empty(ny, self.dtype))
        return lay

    def _apply_el(self, g, wd):
        g0, f = divmod(int(g), self.nf); NW = self.NW; fl = (NW-1)*f
        return [self.WTT[(g0*self.p+T)*NW + wd[int(self._SRC[g0, T])]] ^ fl for T in range(self.p)]

    def _apply_var(self, g, wd):
        xp, p, NW = self.xp, self.p, self.NW; g0 = g//self.nf; fl = (g % self.nf)*(NW-1); ws = xp.stack(wd); ar = xp.arange(len(g))
        return [self.WTT[(g0*p+T)*NW + ws[self.SRC[g0, T], ar]] ^ fl for T in range(p)]

    def _spec(self, lay, v):
        s = v[lay.sp[0]]
        for j in range(1, len(lay.sp)):
            s = s | (v[lay.sp[j]] << (j*self.w))
        return s

    def _entry(self, lay, v):
        spec = self._spec(lay, v); n2 = self.nup-lay.PCS[spec]
        return lay.BASE[n2] + (self.OFFT[n2*self.NW + v[lay.q[1]]] + self.IDX[v[lay.q[0]]])*lay.Mc[n2] + lay.COL[spec]

    def _decode(self, lay, e):
        xp, w = self.xp, self.w; n2 = xp.searchsorted(lay.BASE, e, side='right')-1; loc = e-lay.BASE[n2]; Mc = lay.Mc[n2]; row = loc//Mc
        pw = self.WCAT[self.WOFF[n2]+row]; spec = lay.COLS[lay.COFF[n2]+(loc-row*Mc)]; out = [None]*self.p
        out[lay.q[0]] = pw & (self.NW-1); out[lay.q[1]] = pw >> w
        for j, T in enumerate(lay.sp):
            out[T] = (spec >> (j*w)) & (self.NW-1)
        return out

    def _canon(self, wd):
        """Home entry of the representative of the orbit of each state, the phase code of an operation that reaches it,
        the number of operations that do, and whether two of them differ in character (then the block has no such state)."""
        xp, H, M = self.xp, self._home, self.Mroot; n = len(wd[0])
        best = xp.full(n, 1 << 62, np.int64); bcode = xp.zeros(n, np.int64); cnt = xp.zeros(n, np.int64); bad = xp.zeros(n, bool)

        def upd(idx, e, code):
            b, bc, ct, bd = (best, bcode, cnt, bad) if idx is None else (best[idx], bcode[idx], cnt[idx], bad[idx])
            lt = e < b; eq = e == b
            nb_ = xp.where(lt, e, b); nc = xp.where(lt, code, bc); nt = xp.where(lt, 1, ct+eq); nd = xp.where(lt, False, bd | (eq & (code != bc)))
            if idx is None:
                best[:] = nb_; bcode[:] = nc; cnt[:] = nt; bad[:] = nd
            else:
                best[idx] = nb_; bcode[idx] = nc; cnt[idx] = nt; bad[idx] = nd
        for d in H.D:
            u = self._apply_el(d, wd); su = self._spec(H, u); rep = H.REP[su]; kel = H.KEL[su]; v = self._apply_var(kel, u)
            code = (int(self._mcode[d]) + self.mcode[kel]) % M
            upd(None, self._entry(H, v), code)
            sn = H.STABN[rep]
            for i in range(1, H.smax):
                idx = xp.flatnonzero(sn > i)
                if len(idx) == 0:
                    break
                h = H.STABL[H.STROW[rep[idx]], i]; vv = self._apply_var(h, [x[idx] for x in v])
                upd(idx, self._entry(H, vv), (code[idx] + self.mcode[h]) % M)
        return best, bcode, cnt, bad

    # ------------------------------------------------------------------ product
    def _product(self, bs, x, y, accumulate):
        xp = self.xp; assert x.dtype == self.dtype and y.dtype == self.dtype, 'layout engine: double-precision vectors of the block type'
        if not accumulate:
            y[...] = 0
        if self.dim == 0:
            return y
        dim = self.dim; w = np.int32(self.w)
        for q in sorted(bs):
            lay = self._layout(q); b = bs[q]; n = lay.nexp; sc = self.fold == 'scatter'; Y = self._buf[1] if sc else self._buf[1][:n]
            if self.device == 'gpu':
                for n2 in range(self.P2):
                    nrow, Mc = int(self._NROW[n2]), int(lay._Mc[n2])
                    if nrow and Mc:
                        b0 = int(lay._BASE[n2]); cnt = nrow*Mc; X = self._buf[0][:cnt]; Yc = Y[:cnt] if sc else Y[b0:b0+cnt]
                        if self.expand == 'fly':                 # rows and codes of this class, computed now and used twice
                            g, H = self._g, self._home; Pc, PCc = self._pbuf[0][:cnt], self._pbuf[1][:cnt]
                            self._kc(((cnt+255)//256,), (256,), (Pc, PCc, self._kW[n2], lay.g['COLS'][int(lay._COFF[n2]):int(lay._COFF[n2+1])], np.int32(Mc),
                                     np.int64(cnt), g['WTT'], g['SRC'], g['MCODE'], H.g['KEL'], H.g['COL'], H.g['STABN'], H.g['STROW'], H.g['STABL'],
                                     H.g['BASE'], H.g['MC'], g['OFFT'], self._kIDX, self._BITS, self._RANK, self._SG, g['SIDX'], lay.PARC))
                            self._ke(((cnt+255)//256,), (256,), (x, X, Pc, PCc, self._TE, np.int64(cnt), np.int64(dim)))
                            self._kp(((cnt+255)//256,), (256,), (X, Yc, self._kW[n2], self._kOFF[n2], self._kIDX, b['MASK'], b['JXY'], b['JZ'],
                                     np.int32(b['nb']), np.int32(Mc), np.int64(cnt), w, np.float64(b['dsum'])))
                            self._ks(((cnt+255)//256,), (256,), (Yc, y, Pc, PCc, self._TE, lay.CW[n2], np.int32(Mc), np.int64(cnt), np.int64(dim)))
                            continue
                        self._ke(((cnt+255)//256,), (256,), (x, X, lay.P[b0:b0+cnt], lay.PC[b0:b0+cnt], self._TE, np.int64(cnt), np.int64(dim)))
                        self._kp(((cnt+255)//256,), (256,), (X, Yc, self._kW[n2], self._kOFF[n2], self._kIDX, b['MASK'], b['JXY'], b['JZ'],
                                 np.int32(b['nb']), np.int32(Mc), np.int64(cnt), w, np.float64(b['dsum'])))
                        if sc:
                            self._ks(((cnt+255)//256,), (256,), (Yc, y, lay.P[b0:b0+cnt], lay.PC[b0:b0+cnt], self._TE, lay.CW[n2], np.int32(Mc),
                                     np.int64(cnt), np.int64(dim)))
                for F, FC in zip(lay.F, lay.FC):
                    self._kf(((dim+255)//256,), (256,), (Y, y, F, FC, self._TF, np.int64(dim)))
                if self.fold == 'fly':
                    g, H = self._g, self._home
                    for par in lay.PAR:
                        self._kff(((dim+255)//256,), (256,), (Y, y, self._CE, self._SG, np.int64(dim), H.g['BASE'], H.g['MC'], g['WCAT'], g['WOFF'], H.g['COLS'],
                                  H.g['COFF'], g['WTT'], g['SRC'], lay.g['KEL'], lay.g['COL'], lay.g['BASE'], lay.g['MC'], g['OFFT'], self._kIDX, g['MCODE'],
                                  g['PH'], par))
            else:
                X = self._buf[0][:n]
                lib = self._clib; P_ = lambda a: a.ctypes.data_as(ctypes.c_void_p); I64 = ctypes.c_int64
                lib.expand(P_(x), P_(X), P_(lay.P), P_(lay.PC), P_(self._TE), I64(n), I64(dim))
                for n2 in range(self.P2):
                    nrow, Mc = int(self._NROW[n2]), int(lay._Mc[n2])
                    if nrow and Mc:
                        b0 = int(lay._BASE[n2]); cnt = nrow*Mc; Xc, Yc = X[b0:b0+cnt], Y[b0:b0+cnt]
                        lib.pair(P_(Xc), P_(Yc), P_(self._kW[n2]), P_(self._kOFF[n2]), P_(self._kIDX), P_(b['MASK']), P_(b['JXY']), P_(b['JZ']),
                                 ctypes.c_int(b['nb']), I64(Mc), I64(nrow), ctypes.c_int(self.w), ctypes.c_double(b['dsum']))
                for F, FC in zip(lay.F, lay.FC):
                    lib.fold(P_(Y), P_(y), P_(F), P_(FC), P_(self._TF), I64(dim))
        return y

    def matvec(self, x, y, accumulate=False):
        """y (+)= H x."""
        return self._product(self._bs, x, y, accumulate)

    def retune(self, model):
        """Change the couplings; the layouts and the conversion tables stay."""
        assert not np.any(model.h); self._bs = self._bondsets(model)
        for q in sorted(self._bs):
            self._layout(q)
        return self

    def operator(self, op_model):
        """y (+)= O x for another model O on the same sites that is invariant under the group of this block."""
        assert not np.any(op_model.h); bs = self._bondsets(op_model)
        for q in sorted(bs):
            self._layout(q)
        return lambda x, y, accumulate=False: self._product(bs, x, y, accumulate)

    def batched(self, R):
        """Y (+)= H X for R vectors stored interleaved, shape (dim, R): one vector after the other."""
        xp = self.xp; tx = xp.empty(self.dim, self.dtype); ty = xp.empty(self.dim, self.dtype)

        def apply(X, Y, accumulate=False):
            assert X.shape == (self.dim, R)
            for r in range(R):
                tx[:] = X[:, r]; self.matvec(tx, ty)
                if accumulate:
                    Y[:, r] += ty
                else:
                    Y[:, r] = ty
            return Y
        return apply

    def _relabel(self, model):
        return model                                             # no relabelling: edc_scan reads the coupling table of the model itself

    def scan(self, models, tol=1e-12, warm=False, nev=1):
        """Ground energies along a list of models (retune between points); see edc_sub.SubEngine.scan."""
        import edc_sub
        return edc_sub.SubEngine.scan(self, models, tol=tol, warm=warm, nev=nev)

    def time_matvec(self, reps=3):
        xp = self.xp; x = xp.asarray(np.random.default_rng(0).standard_normal(self.dim).astype(self.dtype)); y = xp.empty_like(x); best = np.inf
        sync = (lambda: xp.cuda.Stream.null.synchronize()) if xp is not np else (lambda: None)
        for _ in range(reps+1):
            sync(); t = time.perf_counter(); self.matvec(x, y); sync(); best = min(best, time.perf_counter()-t)
        return best

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3, v0=None, keep=False, rtol=None, etol=None):
        from edc_hubbard_sym import lanczos
        return lanczos(self, tol, maxit, seed, nev, stable, v0=v0, keep=keep, rtol=rtol, etol=etol)
