#!/usr/bin/env python3
"""Fermi-Hubbard model with lattice symmetries, generated C/OpenMP (CPU) or CUDA (GPU) from one source.

H = sum_<ab>,sigma -t_ab (c+_a,s c_b,s + h.c.) + U sum_a n_a,up n_a,dn,   fixed N_up, N_dn.

Symmetry sector: momentum k, a 1D irrep of the little group of k (translations x point operations of an
edc_lattice.Cluster), and for N_up = N_dn optionally the spin flip F (c_up <-> c_dn) with parity +1 or -1.

Conventions (operator order: all up operators, sites ascending, then all down operators, sites ascending):
  U(g) c+_a,s U(g)^-1 = c+_g(a),s, so U(g)|u, d> = sigma(g, u) sigma(g, d) |g u, g d>, where sigma(g, S) = (-1)^(number
  of pairs a < b in S with g(a) > g(b));  U(F)|u, d> = (-1)^(N_up N_dn) |d, u>.
  Sector projector P = (1/|G|) sum_g chi(g)^* U(g); basis state |r> = P|r> / norm for every representative r (the
  minimum of its orbit, key (u, d) compared u first) whose stabiliser carries chi(g) sigma(g, r) = 1 throughout.
  Pull form of the product:  (H x)_r = sum over hops s of r:  h_sr chi(g) sigma(g, s) sqrt(|Stab r'| / |Stab r|) x_r',
  where g maps s to its representative r'.

Representatives species by species: the up string of the representative is the orbit minimum of the up string
(of the down string for flipped images); one table look-up gives that minimum and one group element g0 that reaches
it; only the stabiliser of the minimum (usually the identity alone) is then searched for the down string.  The
representatives are bits in a bitmap over (up-orbit minimum) x (all down strings), so the bitmap is |G| times smaller
than a bitmap over the whole (N_up, N_dn) space.  Group elements act through byte tables, fermion signs through one
mask per (element, site)."""
import os, time, math, cmath, ctypes, hashlib
import numpy as np
import edc
import edc_lattice as el

_BOUND = {}     # library or module key -> id of the engine whose tables are bound there (identical sources share one
                # compiled library, and with it the global table pointers)


# ------------------------------------------------------------------ model
def hubbard_bonds(cl, t=1.0, tp=0.0):
    """Nearest-neighbour hopping t (and next-nearest t') on a Cluster (edc_lattice) or LatticeCluster (edc_lattices,
    by distance shells); returns {(a, b): t_ab} with a < b."""
    if hasattr(cl, 'bonds'):
        h = {pr: float(t) for pr in cl.bonds(1)}
        if tp:
            for pr in cl.bonds(2):
                h[pr] = h.get(pr, 0.0)+tp
        return h
    h = {}
    for s, (x, y) in enumerate(cl.coord):
        vecs = [((1, 0), t), ((0, 1), t)] + ([((1, 1), tp), ((1, -1), tp)] if tp else [])
        for (dx, dy), tv in vecs:
            o = cl.site(x+dx, y+dy)
            assert o != s, 'bond to itself: cluster too small'
            key = (min(s, o), max(s, o))
            h[key] = h.get(key, 0.0)+tv
    return h


def hop_families(bonds):
    """Split bonds into families in which every site is the first member of at most one bond."""
    fams = []
    for (a, b), tv in sorted(bonds.items()):
        for f in fams:
            if a not in f:
                f[a] = (b, tv); break
        else:
            fams.append({a: (b, tv)})
    return fams


def species_strings(L, n):
    """All L-bit strings with n ones, ascending (= the Lin rank order)."""
    import itertools
    s = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(L), n)], np.int64)
    s.sort()
    return s


def apply_perm(perm, s):
    out = np.zeros_like(s)
    for a, b in enumerate(perm):
        out |= ((s >> a) & 1) << b
    return out


# ------------------------------------------------------------------ code generation
def _hex(v):
    return f'{hex(v)}u'


def source(L, nup, ndn, fams, U, ngr, nch, flip, real, target, fs, sflip, tab=True):
    """Generated C (target 'c') or CUDA (target 'cuda') for one model and sector shape.  Tables are global pointers
    bound at run time, so the code depends only on the model, N_up, N_dn, the group size and real/complex.
    tab: the product works on string ranks with a table APPR[g][rank s] = rank(g s) | sign << 31 (no bit
    permutations and no sign loops in the inner loop); otherwise group action by byte tables and sign loops."""
    LA = L//2
    o = []
    if target == 'cuda':
        o += ['#include <cupy/complex.cuh>', '#define DEV __device__ __forceinline__', '#define GLOB __device__',
              '#define POPC(x) __popc(x)', '#define POPC64(x) __popcll(x)', '#define FFS(x) __ffs(x)', 'typedef unsigned int st_t;', 'typedef long long i64;',
              'typedef unsigned long long u64;']
        o += ['typedef double vt;' if real else 'typedef complex<double> vt;']
    else:
        o += ['#include <stdint.h>', '#include <math.h>', '#include <complex.h>', '#include <omp.h>',
              '#define DEV static inline', '#define GLOB static', '#define POPC(x) __builtin_popcount(x)', '#define POPC64(x) __builtin_popcountll(x)',
              '#define FFS(x) __builtin_ffs(x)', 'typedef uint32_t st_t;', 'typedef int64_t i64;', 'typedef uint64_t u64;']
        o += ['typedef double vt;' if real else 'typedef double _Complex vt;']
    o.append(f'#define NS {L}\n#define NCH {nch}\n#define LOWM {_hex((1 << LA)-1)}\n#define LA {LA}')
    # tables (bound at run time)
    tabs = [('unsigned', 'PT'), ('unsigned', 'MSK'), ('double', 'CHR'), ('double', 'CHI_'), ('long long', 'JAU'),
            ('long long', 'JAD'), ('unsigned', 'JB'), ('int', 'REPU'), ('unsigned short', 'G0'), ('unsigned', 'UREP'),
            ('int', 'SP'), ('unsigned short', 'SI'), ('unsigned', 'DSTR'), ('unsigned long long', 'BITS'),
            ('unsigned long long', 'RANK'), ('unsigned', 'STU'), ('unsigned', 'STD'), ('unsigned char', 'SGA'),
            ('unsigned', 'APPR'), ('unsigned char', 'SIS')]
    for ty, nm in tabs:
        o.append(f'GLOB const {ty}* {nm};')
    o.append('GLOB long long DD;')                      # number of down strings
    if target == 'c':
        o.append('void bind(void** p, long long dd) { ' + ' '.join(f'{nm} = (const {ty}*)p[{i}];' for i, (ty, nm) in enumerate(tabs))
                 + ' DD = dd; }')
    # group action, signs, characters
    o.append('DEV st_t app(int g, st_t s) { const unsigned* t = PT + (long long)g * NCH * 256; return '
             + ' | '.join(f't[{c*256} + ((s >> {8*c}) & 255u)]' for c in range(nch)) + '; }')
    o.append('DEV int par(int g, st_t s) { const unsigned* m = MSK + g * NS; int p = 0; st_t z = s; '
             'while (z) { int a = FFS(z) - 1; z &= z - 1; p ^= POPC(s & m[a]); } return p & 1; }')
    if real:
        o.append('DEV vt chi(int g) { return CHR[g]; }')
    elif target == 'cuda':
        o.append('DEV vt chi(int g) { return vt(CHR[g], CHI_[g]); }')
    else:
        o.append('DEV vt chi(int g) { return CHR[g] + CHI_[g] * I; }')
    o.append('DEV long long ranku(st_t s) { return JAU[s >> LA] + JB[s & LOWM]; }')
    o.append('DEV long long rankd(st_t s) { return JAD[s >> LA] + JB[s & LOWM]; }')
    # one branch of the representative search: the up string of the image comes from `a`, the down string from `b`
    # sign of g0 on the string a is tabulated (SGA); the identity (element 0) needs neither action nor sign
    o.append('DEV void branch(st_t a, st_t b, long long ia, int* q, st_t* best, int* cnt, int* gb, int* hb, int* sg) {'
             ' int qq = REPU[ia]; int g0 = G0[ia]; st_t bb = b; int s0 = 0;'
             ' if (g0) { bb = app(g0, b); s0 = SGA[ia] ^ par(g0, b); }'
             ' int p0 = SP[qq], p1 = SP[qq + 1];'
             ' if (p1 - p0 == 1) { *q = qq; *best = bb; *cnt = 1; *gb = g0; *hb = 0; *sg = s0; return; }'
             ' st_t bm = 0xFFFFFFFFu; int c = 0, h0 = 0;'
             ' for (int p = p0; p < p1; p++) { int h = SI[p]; st_t v = h ? app(h, bb) : bb;'
             ' if (v < bm) { bm = v; c = 1; h0 = h; } else if (v == bm) c++; }'
             ' if (h0) s0 ^= par(h0, UREP[qq]) ^ par(h0, bb);'
             ' *q = qq; *best = bm; *cnt = c; *gb = g0; *hb = h0; *sg = s0; }')
    # representative of (u, d): returns rep (ru, rd), its stabiliser size, and phase chi(g) sigma(g, s) for one g
    rep = ['DEV void rep_of(st_t u, st_t d, st_t* ru, st_t* rd, int* st, vt* ph) {',
           ' int q, c, g0, h0, s; st_t b; branch(u, d, ranku(u), &q, &b, &c, &g0, &h0, &s);']
    if flip:
        rep += [' int q2, c2, g2, h2, s2; st_t b2; branch(d, u, rankd(d), &q2, &b2, &c2, &g2, &h2, &s2);',
                ' if (q2 < q || (q2 == q && b2 < b)) {'
                f' *ru = UREP[q2]; *rd = b2; *st = c2; *ph = chi(h2) * chi(g2) * ({fs!r} * (((s2 ^ {sflip}) & 1) ? -1.0 : 1.0)); return; }}',
                ' if (q2 == q && b2 == b) c += c2;']
    rep += [' *ru = UREP[q]; *rd = b; *st = c; *ph = chi(h0) * chi(g0) * ((s & 1) ? -1.0 : 1.0); }']
    o.append(' '.join(rep))
    if tab:
        # the same search on ranks: ia = rank of the string giving the image's up part, ib = rank of the other one
        o.append('DEV void branch_t(long long ia, long long ib, int* q, long long* best, int* cnt, int* gb, int* hb, int* sg) {'
                 ' int qq = REPU[ia]; int g0 = G0[ia]; long long rb = ib; int s0 = 0;'
                 ' if (g0) { unsigned t = APPR[(i64)g0 * DD + ib]; rb = t & 0x7FFFFFFFu; s0 = SGA[ia] ^ (int)(t >> 31); }'
                 ' int p0 = SP[qq], p1 = SP[qq + 1];'
                 ' if (p1 - p0 == 1) { *q = qq; *best = rb; *cnt = 1; *gb = g0; *hb = 0; *sg = s0; return; }'
                 ' long long bm = 0x7FFFFFFFFFFFLL; int c = 0, h0 = 0, pb = p0, sb = 0;'
                 ' for (int p = p0; p < p1; p++) { int h = SI[p]; long long v = rb; int sv = 0;'
                 '  if (h) { unsigned t = APPR[(i64)h * DD + rb]; v = t & 0x7FFFFFFFu; sv = (int)(t >> 31); }'
                 '  if (v < bm) { bm = v; c = 1; h0 = h; pb = p; sb = sv; } else if (v == bm) c++; }'
                 ' if (h0) s0 ^= SIS[pb] ^ sb;'
                 ' *q = qq; *best = bm; *cnt = c; *gb = g0; *hb = h0; *sg = s0; }')
        rt = ['DEV void rep_t(long long iu, long long id, int* rq, long long* rb, int* st, vt* ph) {']
        if flip:
            # the image's up part is the smaller of the two orbit minima: evaluate only the branch(es) that can win
            rt += [' int qa = REPU[iu], qb = REPU[id];',
                   ' if (qb < qa) { int q2, c2, g2, h2, s2; long long b2; branch_t(id, iu, &q2, &b2, &c2, &g2, &h2, &s2);'
                   f' *rq = q2; *rb = b2; *st = c2; *ph = chi(h2) * chi(g2) * ({fs!r} * (((s2 ^ {sflip}) & 1) ? -1.0 : 1.0)); return; }}']
        rt += [' int q, c, g0, h0, s; long long b; branch_t(iu, id, &q, &b, &c, &g0, &h0, &s);']
        if flip:
            rt += [' if (qb == qa) { int q2, c2, g2, h2, s2; long long b2; branch_t(id, iu, &q2, &b2, &c2, &g2, &h2, &s2);',
                   '  if (b2 < b) {'
                   f' *rq = q2; *rb = b2; *st = c2; *ph = chi(h2) * chi(g2) * ({fs!r} * (((s2 ^ {sflip}) & 1) ? -1.0 : 1.0)); return; }}',
                   '  if (b2 == b) c += c2; }']
        rt += [' *rq = q; *rb = b; *st = c; *ph = chi(h0) * chi(g0) * ((s & 1) ? -1.0 : 1.0); }']
        o.append(' '.join(rt))
    # stabiliser test for a candidate (u = an up-orbit minimum, d): 0 if not a representative of this sector
    chk = ['DEV int stab_ok(int q, st_t u, st_t d) { int st = 0;',
           ' for (int p = SP[q]; p < SP[q + 1]; p++) { int h = SI[p]; st_t v = app(h, d); if (v < d) return 0;',
           '  if (v == d) { st++; vt c = chi(h) * ((par(h, u) ^ par(h, d)) ? -1.0 : 1.0);']
    chk += ['   if (' + ('fabs(c - 1.0) > 1e-9' if real else 'fabs(creal(c) - 1.0) + fabs(cimag(c)) > 1e-9' if target == 'c'
                       else 'fabs(c.real() - 1.0) + fabs(c.imag()) > 1e-9') + ') return 0; } }']
    if flip:
        chk += [' { long long id = rankd(d); int q2 = REPU[id]; if (q2 < q) return 0;',
                '  if (q2 == q) { int g0 = G0[id]; st_t bb = app(g0, u); int s0 = par(g0, d) ^ par(g0, u);',
                '   for (int p = SP[q]; p < SP[q + 1]; p++) { int h = SI[p]; st_t v = app(h, bb); if (v < d) return 0;',
                f'    if (v == d) {{ st++; vt c = chi(h) * chi(g0) * ({fs!r} * (((s0 ^ par(h, u) ^ par(h, bb) ^ {sflip}) & 1) ? -1.0 : 1.0));']
        chk += ['     if (' + ('fabs(c - 1.0) > 1e-9' if real else 'fabs(creal(c) - 1.0) + fabs(cimag(c)) > 1e-9' if target == 'c'
                            else 'fabs(c.real() - 1.0) + fabs(c.imag()) > 1e-9') + ') return 0; } } } }']
    chk += [' return st; }']
    o.append(' '.join(chk))
    # hopping families
    body = []
    for fi, f in enumerate(fams):
        flipm, bet, amp = [0]*L, [0]*L, [0.0]*L
        groups = {}
        for a, (b, tv) in f.items():
            flipm[a] = (1 << a) | (1 << b)
            lo, hi = min(a, b), max(a, b)
            bet[a] = ((1 << hi)-1) & ~((1 << (lo+1))-1)
            amp[a] = -tv
            groups[b-a] = groups.get(b-a, 0) | (1 << a)
        o.append(f'GLOB const st_t FL{fi}[NS] = {{{", ".join(_hex(v) for v in flipm)}}};')
        o.append(f'GLOB const st_t BT{fi}[NS] = {{{", ".join(_hex(v) for v in bet)}}};')
        o.append(f'GLOB const double AM{fi}[NS] = {{{", ".join(repr(float(v)) for v in amp)}}};')
        for sp in ('u', 'd'):
            ex = ' | '.join(f'(({sp} ^ ({sp} >> {dd})) & {_hex(m)})' if dd > 0 else f'(({sp} ^ ({sp} << {-dd})) & {_hex(m)})'
                            for dd, m in sorted(groups.items()))
            tgt = f'{sp} ^ FL{fi}[a]'
            if tab:
                rk = f'(ranku({tgt}), idd' if sp == 'u' else f'(iuu, rankd({tgt})'
                find = (f'int rq, st; long long rb; vt ph; rep_t{rk}, &rq, &rb, &st, &ph); '
                        f'u64 f = (u64)rq * (u64)DD + (u64)rb; u64 w = BITS[f >> 6]; ')
            else:
                arg = f'({tgt}, d' if sp == 'u' else f'(u, {tgt}'
                find = (f'st_t ru, rd; int st; vt ph; rep_of{arg}, &ru, &rd, &st, &ph); '
                        f'u64 f = (u64)REPU[ranku(ru)] * (u64)DD + (u64)rankd(rd); u64 w = BITS[f >> 6]; ')
            body.append(f'{{ st_t z = {ex}; while (z) {{ int a = FFS(z) - 1; z &= z - 1; '
                        f'double h = AM{fi}[a] * ((POPC({sp} & BT{fi}[a]) & 1) ? -1.0 : 1.0); ' + find +
                        f'if (!((w >> (f & 63)) & 1ull)) continue; '
                        f'i64 j = (i64)RANK[f >> 6] + POPC64(w & ((1ull << (f & 63)) - 1)); '
                        f'acc += (h * sqrt((double)st)) * ph * x[j]; }} }}')
    o.append('DEV vt row(i64 i, const vt* x) { st_t u = STU[i], d = STD[i]; vt acc = 0.0;'
             + (' long long iuu = ranku(u), idd = rankd(d);' if tab else ''))
    o += ['  '+b for b in body]
    if tab:
        o.append('  int rq, st; long long rb; vt ph; rep_t(iuu, idd, &rq, &rb, &st, &ph);')
    else:
        o.append('  st_t ru, rd; int st; vt ph; rep_of(u, d, &ru, &rd, &st, &ph);')
    o.append(f'  return ({float(U)!r} * (double)POPC(u & d)) * x[i] + acc / sqrt((double)st); }}')
    # set-up: bitmap of representatives over (up-orbit minimum) x (down strings), then the state list
    o.append('DEV u64 bits_word(i64 w, i64 nbits) { u64 word = 0; for (int b = 0; b < 64; b++) { i64 f = w * 64 + b;'
             ' if (f >= nbits) break; int q = (int)(f / DD); st_t d = DSTR[f - (i64)q * DD]; if (stab_ok(q, UREP[q], d)) word |= 1ull << b; }'
             ' return word; }')
    if target == 'cuda':
        o.append('extern "C" __global__ void mv(const vt* __restrict__ x, vt* __restrict__ y, i64 n, int accumulate) {'
                 ' i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; vt r = row(i, x); y[i] = accumulate ? y[i] + r : r; }')
        o.append('extern "C" __global__ void rep_bits(u64* out, i64 nw, i64 nbits) {'
                 ' i64 w = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (w < nw) out[w] = bits_word(w, nbits); }')
        o.append('extern "C" __global__ void make_reps(unsigned* su, unsigned* sd, i64 nw) {'
                 ' i64 w = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (w >= nw) return; u64 word = BITS[w]; i64 k = (i64)RANK[w];'
                 ' while (word) { int b = __ffsll(word) - 1; word &= word - 1; i64 f = w * 64 + b; int q = (int)(f / DD);'
                 ' su[k] = UREP[q]; sd[k] = DSTR[f - (i64)q * DD]; k++; } }')
    else:
        o.append('void mv(const vt* x, vt* y, i64 n, int accumulate) {\n#pragma omp parallel for schedule(dynamic, 1024)\n'
                 ' for (i64 i = 0; i < n; i++) { vt r = row(i, x); y[i] = accumulate ? y[i] + r : r; } }')
        # threaded vector operations for the Lanczos loop, on the underlying doubles (complex = 2 doubles; real parts of
        # <a|b> and real scalars only, which is all Lanczos needs)
        o.append('double vdot_re(const double* a, const double* b, i64 m) { double s = 0.0;\n#pragma omp parallel for reduction(+:s) schedule(static)\n'
                 ' for (i64 i = 0; i < m; i++) s += a[i] * b[i]; return s; }')
        o.append('void axpy(double c, const double* x, double* y, i64 m) {\n#pragma omp parallel for schedule(static)\n'
                 ' for (i64 i = 0; i < m; i++) y[i] += c * x[i]; }')
        o.append('void scale2(double* u, double cu, double* v, double cv, i64 m) {\n#pragma omp parallel for schedule(static)\n'
                 ' for (i64 i = 0; i < m; i++) { u[i] *= cu; v[i] *= cv; } }')
        o.append('void rep_bits(u64* out, i64 nw, i64 nbits) {\n#pragma omp parallel for schedule(dynamic, 256)\n'
                 ' for (i64 w = 0; w < nw; w++) out[w] = bits_word(w, nbits); }')
        o.append('void make_reps(unsigned* su, unsigned* sd, i64 nw) {\n#pragma omp parallel for schedule(dynamic, 256)\n'
                 ' for (i64 w = 0; w < nw; w++) { u64 word = BITS[w]; i64 k = (i64)RANK[w];'
                 ' while (word) { int b = __builtin_ctzll(word); word &= word - 1; i64 f = w * 64 + b; int q = (int)(f / DD);'
                 ' su[k] = UREP[q]; sd[k] = DSTR[f - (i64)q * DD]; k++; } } }')
    return '\n'.join(o)+'\n'


# ------------------------------------------------------------------ engine
class HubbardSym:
    """Lowest levels of one symmetry sector of the Hubbard model on an edc_lattice.Cluster.
    device: 'cpu' (generated C + OpenMP) or 'gpu' (generated CUDA via CuPy)."""

    def __init__(self, cl, bonds, U, nup, ndn, k=(0.0, 0.0), irrep='A1', flip=0, device='cpu', point_ops=None, tab=None,
                 block=256):
        t0 = time.perf_counter()
        L = cl.N
        assert L <= 32
        assert flip in (0, 1, -1) and (flip == 0 or nup == ndn), 'spin flip needs N_up = N_dn'
        self.cl, self.L, self.nup, self.ndn, self.k, self.irrep, self.flip, self.device = cl, L, nup, ndn, k, irrep, flip, device
        if hasattr(cl, 'irrep_table'):                     # edc_lattices: irreps of the (restricted) little group
            perms, chars, ops = cl.group(k, irrep, allowed=point_ops)
        else:
            ops = [o for o in cl.little_group(k) if o in el.IRREPS[irrep] and (point_ops is None or o in point_ops)]
            perms, chars, ops = cl.group(k, irrep, ops)
        ngr = len(perms)
        assert ngr < 65536
        chars = np.array(chars, complex)
        self.real = bool(np.all(np.abs(chars.imag) < 1e-12))
        self.group_order = ngr*(2 if flip else 1)
        # byte tables for the action of every element, sign masks
        nch = (L+7)//8
        PT = np.zeros((ngr, nch, 256), np.uint32)
        byte = np.arange(256, dtype=np.int64)
        MSK = np.zeros((ngr, L), np.uint32)
        for g, p in enumerate(perms):
            for c in range(nch):
                v = np.zeros(256, np.int64)
                for bi in range(8):
                    a = 8*c+bi
                    if a < L:
                        v |= ((byte >> bi) & 1) << p[a]
                PT[g, c] = v
            for a in range(L):
                MSK[g, a] = sum(1 << b for b in range(a+1, L) if p[b] < p[a])
        # species tables
        su, sd = species_strings(L, nup), species_strings(L, ndn)
        imgs = np.empty((len(su), ngr), np.int64)
        for g, p in enumerate(perms):
            imgs[:, g] = apply_perm(p, su)
        umin = imgs.min(axis=1); g0 = imgs.argmin(axis=1)
        urep = np.unique(umin)
        repu = np.searchsorted(urep, umin).astype(np.int32)
        if nup != ndn:          # the down strings need their own orbit tables only for the flipped branch (unused)
            pass
        stab = [np.flatnonzero(imgs[np.searchsorted(su, r)] == r) for r in urep]
        sp = np.zeros(len(urep)+1, np.int32); sp[1:] = np.cumsum([len(x) for x in stab])
        si = np.concatenate(stab).astype(np.uint16)
        assert np.all(g0[np.searchsorted(su, urep)] == 0)    # a minimum is reached by the identity first
        # parity of g0(u) acting on u itself, one byte per up string
        sga = np.zeros(len(su), np.uint8)
        for g in np.unique(g0):
            sel = np.flatnonzero(g0 == g); S = su[sel]; parity = np.zeros(len(S), np.int64)
            for a in range(L):
                parity += ((S >> a) & 1)*np.bitwise_count(S & np.int64(MSK[g, a]))
            sga[sel] = (parity & 1).astype(np.uint8)
        _, _, jau, jb, du = edc.lin_tables(L, nup)
        _, _, jad, _, dd = edc.lin_tables(L, ndn)
        assert du == len(su) and dd == len(sd)
        self.nq, self.dd = len(urep), dd
        self.host = dict(PT=PT.ravel(), MSK=MSK.ravel(), CHR=chars.real.copy(), CHI_=chars.imag.copy(), JAU=jau, JAD=jad,
                         JB=jb, REPU=repu, G0=g0.astype(np.uint16), UREP=urep.astype(np.uint32), SP=sp, SI=si,
                         DSTR=sd.astype(np.uint32), SGA=sga)
        # rank tables of the group action on the second species: APPR[g][rank s] = rank(g s) | sign(g, s) << 31
        self.tab = tab if tab is not None else ngr*len(sd)*4 <= 4 << 30
        if self.tab:
            appr = np.empty((ngr, len(sd)), np.uint32)
            for g, p in enumerate(perms):
                img = apply_perm(p, sd); parity = np.zeros(len(sd), np.int64)
                for a in range(L):
                    parity += ((sd >> a) & 1)*np.bitwise_count(sd & np.int64(MSK[g, a]))
                appr[g] = np.searchsorted(sd, img).astype(np.uint32) | ((parity & 1).astype(np.uint32) << np.uint32(31))
            sis = np.zeros(len(si), np.uint8)
            for qq in range(len(urep)):
                for pp in range(sp[qq], sp[qq+1]):
                    h = si[pp]; S = int(urep[qq])
                    sis[pp] = sum(bin(S & int(MSK[h, a])).count('1') for a in range(L) if (S >> a) & 1) & 1
        else:
            appr, sis = np.zeros(1, np.uint32), np.zeros(1, np.uint8)
        self.host.update(APPR=appr.ravel(), SIS=sis)
        fams = hop_families(bonds)
        fs = float(flip) if flip else 1.0
        sflip = (nup*ndn) & 1
        target = 'cuda' if device == 'gpu' else 'c'
        self.src = source(L, nup, ndn, fams, U, ngr, nch, flip != 0, self.real, target, fs, sflip, self.tab)
        self.cpu_tables_seconds = time.perf_counter()-t0
        nbits = self.nq*dd
        nw = (nbits+63)//64
        if device == 'gpu':
            self._setup_gpu(nbits, nw, block)
        else:
            self._setup_cpu(nbits, nw)
        self.setup_seconds = time.perf_counter()-t0

    # -------------------------------------------------- GPU
    def _bind_gpu(self):
        import cupy as cp
        for nm, a in self.dev.items():
            gp = self.mod.get_global(nm)
            gp.copy_from_host(np.array([a.data.ptr], np.uint64).ctypes.data, 8)
        gp = self.mod.get_global('DD'); gp.copy_from_host(np.array([self.dd], np.int64).ctypes.data, 8)
        _BOUND[self._bind_key()] = id(self)

    def _setup_gpu(self, nbits, nw, block):
        import cupy as cp
        self.xp = cp
        self.mod = cp.RawModule(code=self.src, options=('-std=c++17',))
        self.dev = {nm: cp.asarray(a) for nm, a in self.host.items()}
        self.dev['BITS'] = cp.zeros(1, cp.uint64); self.dev['RANK'] = cp.zeros(1, cp.uint64)
        self.dev['STU'] = cp.zeros(1, cp.uint32); self.dev['STD'] = cp.zeros(1, cp.uint32)
        bits = cp.empty(nw, cp.uint64)
        self._bind_gpu()
        self.mod.get_function('rep_bits')(((nw+block-1)//block,), (block,), (bits, np.int64(nw), np.int64(nbits)))
        cnt = cp.zeros(nw, cp.uint64)
        cnt[:] = cp.asarray(_popcount64(bits))
        incl = cp.cumsum(cnt)
        self.dim = n = int(incl[-1]) if nw else 0
        rank = incl-cnt; del incl, cnt
        stu, std = cp.empty(max(n, 1), cp.uint32), cp.empty(max(n, 1), cp.uint32)
        self.dev.update(BITS=bits, RANK=rank)
        self._bind_gpu()
        self.mod.get_function('make_reps')(((nw+block-1)//block,), (block,), (stu, std, np.int64(nw)))
        self.dev.update(STU=stu, STD=std)
        self._bind_gpu()
        self.kern = self.mod.get_function('mv')
        self.block, self.grid = block, (max(n, 1)+block-1)//block
        cp.cuda.Stream.null.synchronize()
        self.device_gib = sum(a.nbytes for a in self.dev.values())/2**30

    # -------------------------------------------------- CPU
    def _setup_cpu(self, nbits, nw):
        self.xp = np
        lib, self.compile_seconds, _ = edc.compile_source(self.src, 'hubsym')
        self.lib = lib
        self.arrs = dict(self.host)
        self.arrs.update(BITS=np.zeros(1, np.uint64), RANK=np.zeros(1, np.uint64), STU=np.zeros(1, np.uint32),
                         STD=np.zeros(1, np.uint32))
        bits = np.zeros(nw, np.uint64)
        self._bind_cpu()
        lib.rep_bits(bits.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(nw), ctypes.c_int64(nbits))
        cnt = _popcount64(bits).astype(np.uint64)
        incl = np.cumsum(cnt)
        self.dim = n = int(incl[-1]) if nw else 0
        rank = (incl-cnt).astype(np.uint64)
        self.arrs.update(BITS=bits, RANK=rank)
        stu, std = np.zeros(max(n, 1), np.uint32), np.zeros(max(n, 1), np.uint32)
        self._bind_cpu()
        lib.make_reps(stu.ctypes.data_as(ctypes.c_void_p), std.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(nw))
        self.arrs.update(STU=stu, STD=std)
        self._bind_cpu()
        self.device_gib = sum(a.nbytes for a in self.arrs.values())/2**30

    def _bind_cpu(self):
        order = ['PT', 'MSK', 'CHR', 'CHI_', 'JAU', 'JAD', 'JB', 'REPU', 'G0', 'UREP', 'SP', 'SI', 'DSTR', 'BITS', 'RANK', 'STU', 'STD', 'SGA',
                 'APPR', 'SIS']
        self.arrs = {k: np.ascontiguousarray(v) for k, v in self.arrs.items()}
        ptrs = (ctypes.c_void_p*len(order))(*[self.arrs[k].ctypes.data for k in order])
        self._ptrs = ptrs
        self.lib.bind(ptrs, ctypes.c_longlong(self.dd))
        _BOUND[('cpu', self.lib._name)] = id(self)

    # -------------------------------------------------- products and Lanczos
    def _ensure_bound(self):
        key = self._bind_key()
        if _BOUND.get(key) != id(self):
            self._bind_gpu() if self.device == 'gpu' else self._bind_cpu()

    def _bind_key(self):
        return ('gpu', int(self.mod.get_global('ST' if 'ST' in self.dev else 'STU').ptr)) if self.device == 'gpu' else ('cpu', self.lib._name)

    def matvec(self, x, y, accumulate=False):
        self._ensure_bound()
        if self.device == 'gpu':
            self.kern((self.grid,), (self.block,), (x, y, np.int64(self.dim), np.int32(1 if accumulate else 0)))
        else:
            self.lib.mv(x.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(self.dim),
                        ctypes.c_int(1 if accumulate else 0))
        return y

    def time_matvec(self, reps=3):
        xp = self.xp
        dt = np.float64 if self.real else np.complex128
        x = xp.asarray(np.random.default_rng(0).standard_normal(self.dim).astype(dt)); y = xp.empty_like(x)
        best = np.inf
        for _ in range(reps+1):
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            t = time.perf_counter(); self.matvec(x, y)
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            best = min(best, time.perf_counter()-t)
        return best

    def ground_energy(self, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3):
        return lanczos(self, tol, maxit, seed, nev, stable)


def _popcount64(a):
    a = a.astype(np.uint64) if hasattr(a, 'astype') else a
    xp = __import__('cupy') if type(a).__module__.startswith('cupy') else np
    v = a.copy()
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint64)


_DOT32 = None


def _vec_ops(xp, real, fp32=False):
    """In-place AXPY and dot products without full-size temporaries (CuPy's vdot copies a conjugated complex
    vector; numpy's a -= c*b allocates c*b).  fp32: single-precision vectors, dot products accumulated in double
    (only the real part of <a|b> is returned for complex vectors)."""
    global _DOT32
    if fp32:
        if xp is np:
            from scipy.linalg import blas
            ax = blas.saxpy if real else blas.caxpy
            def axpy(a, x, y):
                ax(x, y, a=a)
            def dot(a, b):
                fa, fb = a.view(np.float32), b.view(np.float32)
                return float(sum(np.dot(fa[i:i+1 << 22].astype(np.float64), fb[i:i+1 << 22].astype(np.float64))
                                 for i in range(0, fa.size, 1 << 22)))
            return axpy, dot
        import cupy as cp
        if _DOT32 is None:
            # two stages: each thread sums 1024 products in double (a warp reads 32 neighbouring entries per step), then the
            # partial sums are added.  A library reduction to one number runs on a single block of 512 threads.
            _DOT32 = cp.RawKernel(r'''
extern "C" __global__ void dot32(const float* __restrict__ x, const float* __restrict__ y, double* __restrict__ part, long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long f = (t >> 5) * 32768 + (t & 31); double s = 0.0;
    for (int k = 0; k < 1024 && f < n; k++, f += 32) s += (double)x[f] * (double)y[f];
    part[t] = s;
}''', 'dot32')
        kr = cp.ElementwiseKernel('float32 x, float64 a', 'float32 y', 'y = (float)((double)y + a * (double)x)', 'axpy32')
        kc = cp.ElementwiseKernel('complex64 x, float32 a', 'complex64 y', 'y = y + a * x', 'axpy64c')
        def axpy(a, x, y):
            (kr(x, a, y) if real else kc(x, np.float32(a), y))
        def dot(a, b):
            fa, fb = a.view(cp.float32), b.view(cp.float32); nt = ((fa.size+32767)//32768)*32
            part = cp.empty(((nt+255)//256)*256, cp.float64)
            _DOT32((part.size//256,), (256,), (fa, fb, part, np.int64(fa.size)))
            return float(part.sum())
        return axpy, dot
    if xp is np:
        from scipy.linalg import blas
        ax = blas.daxpy if real else blas.zaxpy
        def axpy(a, x, y):                                    # y += a x
            ax(x, y, a=a)
        dot = (lambda a, b: float(np.dot(a, b))) if real else (lambda a, b: complex(np.vdot(a, b)))
        return axpy, dot
    import cupy as cp
    from edc_gpu import _dotc
    k = cp.ElementwiseKernel('T x, float64 a', 'T y', 'y = y + a * x', 'hubsym_axpy')
    def axpy(a, x, y):
        k(x, a, y)
    from edc_gpu import _ddot
    dot = _ddot if real else (lambda a, b: complex(_dotc(a, b)))
    return axpy, dot


def _random_vector(xp, n, real, seed, fp32=False):
    dt = (np.float32 if real else np.complex64) if fp32 else (np.float64 if real else np.complex128)
    v = xp.empty(n, dt)
    rng = (np.random.default_rng(seed) if xp is np else xp.random.default_rng(seed))
    ch = 1 << 26
    for i in range(0, n, ch):                                # in chunks: no host- or device-sized temporaries
        c = min(ch, n-i)
        v[i:i+c] = rng.standard_normal(c) if real else rng.standard_normal(c)+1j*rng.standard_normal(c)
    return v


def lanczos(eng, tol=1e-12, maxit=1000, seed=42, nev=1, stable=3, small=400, _coef=None, v0=None, keep=False, rtol=None, etol=None):
    """Lanczos with two vectors (Paige form, matvec accumulates into u):  u <- u + H v;  alpha = <v|u>;
    u <- u - alpha v;  beta = |u|;  (v, u) <- (u / beta, -beta v).  The lowest level (nev = 1) or the lowest nev
    distinct levels; stops when they change by less than tol for `stable` consecutive steps.  Sectors up to `small`
    states are diagonalised densely.  The run is recorded in eng._lanczos (coefficients and seed, or the dense lowest
    vector) for ground_state_vector; with _coef (Ritz coefficients) the same run is replayed and sum_k c_k v_k returned.
    v0: start vector (e.g. the ground state of a nearby parameter point) instead of a random one.
    keep: also keep the Lanczos vectors and form the lowest Ritz vector at the end (no replay; memory m vectors).  An
    integer keeps at most that many; a longer run drops them and ground_state_vector replays it.
    rtol (nev = 1): also require the residual of the lowest Ritz pair, beta_m |s_m| (last component of the lowest
    eigenvector of the tridiagonal matrix), below rtol max(1, |E|).  The energy criterion alone stops too early when
    the next level of the sector is close: the energy then moves by less than tol per step long before the vector
    is converged (LOG section 76: residual 5e-6 with a gap of 1e-3).
    etol (nev = 1): also require the energy-error estimate 2 r^2 / (theta_1 - theta_0) of that pair below
    etol max(1, |E|) (r the residual above, theta the two lowest Ritz values): the same condition as rtol, but only as
    strict as the gap of the sector demands."""
    from scipy.linalg import eigvalsh_tridiagonal
    xp, n = eng.xp, eng.dim
    fp32 = getattr(eng, 'fp32', False)
    dt = (np.float32 if eng.real else np.complex64) if fp32 else (np.float64 if eng.real else np.complex128)
    t0 = time.perf_counter()
    if n == 0:
        return None, 0, 0.0
    if n <= small:
        H = np.zeros((n, n), dt); x = xp.zeros(n, dt); y = xp.empty(n, dt)
        for c in range(n):
            x[:] = 0; x[c] = 1; eng.matvec(x, y); H[:, c] = y.get() if xp is not np else y
        w, V = np.linalg.eigh((H+H.conj().T)/2)
        eng._lanczos = dict(dense=V[:, 0].copy())
        return (float(w[0]) if nev == 1 else _distinct(w, nev)), n, time.perf_counter()-t0
    axpy, dot = _vec_ops(xp, eng.real, fp32)
    v = xp.array(v0, dtype=dt, copy=True) if v0 is not None else _random_vector(xp, n, eng.real, seed, fp32)
    v *= 1.0/math.sqrt(dot(v, v).real)
    u = xp.zeros(n, dt)
    scale2 = None
    if xp is np and hasattr(eng, 'lib') and not fp32:       # threaded loops from the generated library
        lib, m = eng.lib, n*(1 if eng.real else 2)
        P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        lib.vdot_re.restype = ctypes.c_double
        dot = lambda a, b: lib.vdot_re(P(a), P(b), ctypes.c_int64(m))
        axpy = lambda c, x, y: lib.axpy(ctypes.c_double(c), P(x), P(y), ctypes.c_int64(m))
        scale2 = lambda a, ca, b, cb: lib.scale2(P(a), ctypes.c_double(ca), P(b), ctypes.c_double(cb), ctypes.c_int64(m))
    A, B = [], []; e_old = None; hits = 0
    if _coef is not None:
        psi = xp.zeros(n, dt)
    kept = [] if keep else None
    kmax = None if keep is True else int(keep or 0)
    for it in range(1, maxit+1):
        if kept is not None and kmax is not None and len(kept) >= kmax:
            kept = None                                       # too long to keep: the vector comes from a replay
        if kept is not None:
            kept.append(v.copy())
        if _coef is not None:
            axpy(float(_coef[it-1]), v, psi)
            if it == len(_coef):
                return psi
        eng.matvec(v, u, accumulate=True)
        al = dot(v, u).real
        axpy(-al, v, u)
        be = math.sqrt(dot(u, u).real)
        A.append(al); B.append(be)
        if _coef is None:
            eng._lanczos = dict(A=A, B=B, seed=seed, v0=v0)
            if nev == 1:
                e = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))[0] if it > 1 else al
                ok = e_old is not None and abs(e-e_old) < tol*max(1.0, abs(e))
                if ok and (rtol is not None or etol is not None) and it > 2:
                    from scipy.linalg import eigh_tridiagonal
                    th, U = eigh_tridiagonal(np.array(A), np.array(B[:it-1]), select='i', select_range=(0, 1))
                    r = be*abs(U[-1, 0]); sc = max(1.0, abs(e))
                    ok = ((rtol is None or r < rtol*sc) and
                          (etol is None or 2*r*r/max(th[1]-th[0], 1e-6*sc) < etol*sc))
            else:
                ev = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, min(it-1, 3*nev+2))) if it > 1 else np.array([al])
                e = _distinct(ev, nev)
                ok = (e_old is not None and len(e) == nev and len(e_old) == nev and
                      all(abs(a-b) < tol*max(1.0, abs(a)) for a, b in zip(e, e_old)))
            hits = hits+1 if ok else 0
            if hits >= stable or be < 1e-12:
                if kept is not None:
                    from scipy.linalg import eigh_tridiagonal
                    _, U = eigh_tridiagonal(np.array(A), np.array(B[:it-1]), select='i', select_range=(0, 0))
                    psi = xp.zeros(n, dt)
                    for c, vk in zip(U[:, 0], kept):
                        axpy(float(c), vk, psi)
                    eng._lanczos['psi'] = psi
                    del kept
                return e, it, time.perf_counter()-t0
            e_old = e
        if scale2 is not None:
            scale2(u, 1.0/be, v, -be)
        else:
            u *= 1.0/be; v *= -be
        v, u = u, v
    return e_old, it, time.perf_counter()-t0


def ground_state_vector(eng):
    """Lowest Ritz vector of the last lanczos() run on eng (replayed with the same seed and operations), normalised.
    Needs three vectors of the sector, a fourth for the check.  Returns (psi, <psi|H|psi>, |H psi - E psi|)."""
    from scipy.linalg import eigh_tridiagonal
    xp, n = eng.xp, eng.dim
    L = eng._lanczos
    if 'psi' in L:
        psi = L['psi']
    elif 'dense' in L:
        psi = xp.asarray(L['dense'].astype(np.float64 if eng.real else np.complex128))
    else:
        A, B = np.array(L['A']), np.array(L['B'])
        m = len(A)
        _, U = eigh_tridiagonal(A, B[:m-1], select='i', select_range=(0, 0))
        psi = lanczos(eng, maxit=m, seed=L['seed'], small=0, _coef=U[:, 0], v0=L.get('v0'))
    axpy, dot = _vec_ops(xp, eng.real)
    psi *= 1.0/math.sqrt(dot(psi, psi).real)
    hp = xp.zeros_like(psi); eng.matvec(psi, hp)
    E = dot(psi, hp).real
    axpy(-E, psi, hp)
    res = math.sqrt(dot(hp, hp).real)
    del hp
    return psi, E, res


def _distinct(w, nev, merge=1e-9):
    out = []
    for e in w:
        if not out or abs(e-out[-1]) > merge*max(1.0, abs(e)):
            out.append(float(e))
        if len(out) == nev:
            break
    return out


# ------------------------------------------------------------------ complete sector lists
ROTATIONS = ('E', 'C4', 'C2', 'C4^3')


def sectors(cl, nup, ndn, spin_flip=True):
    """(k, irrep, flip, point_ops, multiplicity) covering the whole (N_up, N_dn) space: one momentum per point-group
    orbit (multiplicity = orbit size), all 1D irreps of the little group; where the little group is not abelian (D4
    at Gamma or M) only its rotation subgroup C4 is used, which is abelian, so every irrep is 1D and nothing is missed."""
    if hasattr(cl, 'irrep_table'):
        return _sectors_lattices(cl, nup, ndn, spin_flip)
    G = 2*math.pi*np.linalg.inv(cl.M.astype(float)).T
    ops = cl.point_ops()
    def key(k):
        n = np.linalg.solve(G, np.array(k)); n = np.round(n).astype(int)
        # reduce modulo the reciprocal lattice of Z^2 expressed in the cluster basis
        k2 = G @ n
        return (round((k2[0] % (2*math.pi))/(2*math.pi)*4*cl.N) % (4*cl.N), round((k2[1] % (2*math.pi))/(2*math.pi)*4*cl.N) % (4*cl.N))
    seen, out = set(), []
    for k in cl.momenta():
        if key(k) in seen:
            continue
        orbit = {key(np.array(el.D4[o], float) @ np.array(k)) for o in ops}
        seen |= orbit
        lg = cl.little_group(k)
        abelian = not (any(o in lg for o in ('C4', 'C4^3')) and any(o in lg for o in ('Mx', 'My', 'Md', "Md'")))
        pops = None if abelian else ROTATIONS
        lgu = [o for o in lg if pops is None or o in pops]
        chars = set()
        for ir in el.IRREPS:
            if [o for o in lgu if o in el.IRREPS[ir]] != lgu:
                continue
            c = tuple(complex(el.IRREPS[ir][o]) for o in lgu)
            if c in chars:
                continue
            chars.add(c)
            for fl in ((1, -1) if (spin_flip and nup == ndn) else (0,)):
                out.append((tuple(float(x) for x in k), ir, fl, pops, len(orbit)))
    return out


def sectors_named(cl, nup, ndn, spin_flip=True):
    """A complete block decomposition that resolves every irrep of the little group, in one pass and without solving
    anything twice (real Hamiltonians; clusters of edc_lattices).
    Per momentum star, with little group G_k:
      G_k abelian      : every irrep of G_k, as in sectors(); of a pair of complex-conjugate irreps only one is listed,
                         with degeneracy 2 (see conjugate_partners);
      G_k not abelian  : every one-dimensional irrep of the full G_k (A1, A2, B1, B2 of C6v; A1, A2 of C3v: the mirror
                         parity is resolved), and each two-dimensional irrep through ONE of the two conjugate irreps of
                         the rotation subgroup that it contains (a level of that block is a doublet), degeneracy 2.
    Returns a list of dicts: k, irrep (name in cl.irrep_table(k, allowed)), flip, allowed (None = all of G_k, else the
    rotations), star (size of the momentum star), degeneracy (1 or 2), dimension_of_irrep (1 or 2), chars ({operation:
    character}).  Each level of a block stands for star * degeneracy states; the dimensions of the blocks times these
    factors add up to the dimension of the (N_up, N_dn) space (checked in test_sectors_named.py)."""
    ops = cl.point_ops()
    def key(k):
        return tuple(round((float(v) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for v in k)
    flips = (1, -1) if (spin_flip and nup == ndn) else (0,)
    seen, out = set(), []
    for k in cl.momenta():
        if key(k) in seen:
            continue
        orbit = {key(cl.act_k(o, k)) for o in ops}; seen |= orbit
        lg = cl.little_group(k); kk = tuple(float(x) for x in k)
        abelian = all(cl._mul(a, b) == cl._mul(b, a) for a in lg for b in lg)
        cases = [(None, 1)] if abelian else [(None, 1), (cl.rotations(), 2)]
        for allowed, dim_ir in cases:
            tab = cl.irrep_table(k, allowed); H = [o for o in lg if allowed is None or o in allowed]
            secs = [(kk, ir, flips[0], allowed, len(orbit)) for ir in tab]
            partner = conjugate_partners(cl, secs)
            for i, ir in enumerate(tab):
                cplx = max(abs(complex(tab[ir][o]).imag) for o in H) > 1e-9
                if dim_ir == 2 and not cplx:
                    continue                                     # real irreps of the rotations: covered by the 1D irreps of G_k
                if partner[i] is not None:
                    continue                                     # the conjugate of an irrep already listed
                has_partner = any(p == i for p in partner)
                assert not (dim_ir == 2 and not has_partner), 'a complex rotation irrep without its conjugate'
                for fl in flips:
                    out.append(dict(k=kk, irrep=ir, flip=fl, allowed=allowed, star=len(orbit), degeneracy=2 if has_partner else 1,
                                    dimension_of_irrep=dim_ir, chars={o: complex(tab[ir][o]) for o in H}))
    return out


def conjugate_partners(cl, secs):
    """Blocks of a list from sectors() that need not be solved when the Hamiltonian is real in the occupation basis (real
    couplings; every model of this package).  Complex conjugation then commutes with H and maps the block (k, chi) onto
    (-k, chi*).  If -k is k (modulo the reciprocal lattice), or a point operation of the cluster that commutes with every
    operation of the block's group brings -k back to k, the block with the conjugate characters at the same k has the
    same spectrum (and conjugate eigenvectors).
    Returns a list `partner`: partner[i] = j < i if block i is the conjugate of block j (solve j, count it twice),
    else None.  Clusters of edc_lattices only; other clusters get no pairs."""
    partner = [None]*len(secs)
    if not hasattr(cl, 'irrep_table'):
        return partner
    def key(k):
        return tuple(round((float(v) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for v in k)
    tables = {}
    for i, (k, ir, fl, allowed, mult) in enumerate(secs):
        tk = (key(k), None if allowed is None else tuple(allowed))
        if tk not in tables:
            tab = cl.irrep_table(k, allowed); H = [o for o in cl.little_group(k) if allowed is None or o in allowed]
            mk = tuple(-float(v) for v in k)
            back = key(mk) == key(k) or any(key(cl.act_k(g, mk)) == key(k) and all(cl._mul(g, h) == cl._mul(h, g) for h in H)
                                            for g in cl.point_ops())
            tables[tk] = (tab, H, back)
        tab, H, back = tables[tk]
        if not back:
            continue
        chi = [complex(tab[ir][o]) for o in H]
        if max(abs(c.imag) for c in chi) < 1e-9:
            continue                                             # real characters: its own conjugate
        for j in range(i):
            kj, irj, flj, allowedj, _ = secs[j]
            if partner[j] is None and key(kj) == key(k) and flj == fl and allowedj == allowed and \
                    max(abs(complex(tab[irj][o])-c.conjugate()) for o, c in zip(H, chi)) < 1e-9:
                partner[i] = j; break
    return partner


def _sectors_lattices(cl, nup, ndn, spin_flip=True):
    """sectors() for edc_lattices clusters: momentum orbits under the point group, all 1D irreps of the little group,
    restricted to its rotation subgroup where the little group is not abelian."""
    ops = cl.point_ops()
    def key(k):
        return tuple(round((float(v) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for v in k)
    seen, out = set(), []
    for k in cl.momenta():
        if key(k) in seen:
            continue
        orbit = {key(cl.act_k(o, k)) for o in ops}
        seen |= orbit
        lg = cl.little_group(k)
        abelian = all(cl._mul(a, b) == cl._mul(b, a) for a in lg for b in lg)
        allowed = None if abelian else cl.rotations()
        for ir in cl.irrep_table(k, allowed):
            for fl in ((1, -1) if (spin_flip and nup == ndn) else (0,)):
                out.append((tuple(float(x) for x in k), ir, fl, allowed, len(orbit)))
    return out
