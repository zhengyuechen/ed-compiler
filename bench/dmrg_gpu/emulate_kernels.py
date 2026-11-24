#!/usr/bin/env python3
"""The index arithmetic of the CUDA kernels of edc_dmrg_gpu.py, transliterated to Python and compared with the compiled
host product (no GPU needed; small bond dimensions only, the loops are slow).  One function per kernel, same names,
same order of steps.  Usage: emulate_kernels.py [Lx c D]"""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed


def upper(a, n, x):
    lo, hi = 0, n
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if a[m] <= x:
            lo = m
        else:
            hi = m
    return lo


def csr(ops, S):
    ptr = np.zeros(len(ops)*S + 1, dtype=np.int32); ins, vals = [], []
    for t, (o, i, v) in enumerate(ops):
        o = np.asarray(o, dtype=np.int64); order = np.argsort(o, kind='stable'); cnt = np.bincount(o, minlength=S)
        ptr[t*S+1:(t+1)*S+1] = ptr[t*S] + np.cumsum(cnt); ins.append(np.asarray(i, dtype=np.int32)[order]); vals.append(np.asarray(v, dtype=np.float64)[order])
    return ptr, (np.concatenate(ins) if ins else np.zeros(1, np.int32)), (np.concatenate(vals) if vals else np.zeros(1))


def product(eng, r, v):
    p = eng.plan[r]; md = eng.model; S = p.c.S; nL, nR = p.nL, p.nR; NTL, NTR = max(1, p.c.NTL), max(1, p.c.NTR)
    DL, DR, C, R = p.DL, p.DR, p.C, p.R; cst, rst, ul, mr = (a.reshape(-1) for a in (p.cst, p.rst, p.ul, p.mr))
    cumc = (np.cumsum(p.cw, axis=1) - p.cw).reshape(-1); cumr = (np.cumsum(p.rh, axis=1) - p.rh).reshape(-1); Moff, Uoff = p.Moff, p.Uoff
    srcL, wstL, srcR, wstR = (a.reshape(-1) for a in (p.srcL, p.wstL, p.srcR, p.wstR)); WL, WR = p.WL, p.WR
    SLoff = ed._off(WL.astype(np.int64)*C); SRoff = ed._off(R.astype(np.int64)*WR); n = p.size
    hasL = md.site_ls[r] >= 0 and r > 0; hasR = md.site_rs[r] >= 0 and r < md.n-1
    LH, GL, RH, GR = eng.LH[r], eng.GL[r], eng.RH[r+1], eng.GR[r+1]; M = v; out = np.zeros(n); U = np.zeros(n); oU = np.zeros(n)
    if hasL:                                                           # ---- sl_build
        ptr, ein, ev = csr(md.lstacks[md.site_ls[r]].ops, S); SL = np.zeros(int(SLoff[-1]))
        for x in range(len(SL)):
            i = upper(SLoff, nL, x); loc = x - SLoff[i]; Ci = C[i]; rr, c = divmod(int(loc), int(Ci)); t = -1
            for tt in range(NTL):
                s_ = srcL[i*NTL+tt]
                if s_ < 0:
                    continue
                w0 = wstL[i*NTL+tt]
                if w0 <= rr < w0 + DL[s_]:
                    t, a, is_ = tt, rr - w0, s_; break
            if t < 0:
                continue
            sp = upper(cumc[i*S:(i+1)*S], S, c); b = c - cumc[i*S+sp]; acc = 0.0
            for e in range(ptr[t*S+sp], ptr[t*S+sp+1]):
                cs = cst[is_*S + ein[e]]
                if cs >= 0:
                    acc += ev[e]*M[Moff[is_] + a*C[is_] + cs + b]
            SL[x] = acc
    for i in range(nL):                                                # ---- products of the left half
        Di, Ci, W = int(DL[i]), int(C[i]), int(WL[i])
        if Di and Ci:
            O = out[Moff[i]:Moff[i+1]].reshape(Di, Ci); Mi = M[Moff[i]:Moff[i+1]].reshape(Di, Ci)
            if r > 0:
                O += LH[p.LHoff[i]:p.LHoff[i+1]].reshape(Di, Di) @ Mi
            if W and hasL:
                O += GL[p.GLoff[i]:p.GLoff[i+1]].reshape(Di, W) @ SL[SLoff[i]:SLoff[i+1]].reshape(W, Ci)
    for x in range(n):                                                 # ---- m2u
        j = upper(Uoff, nR, x); loc = x - Uoff[j]; Dj = DR[j]; rr, b = divmod(int(loc), int(Dj)); s = upper(cumr[j*S:(j+1)*S], S, rr); a = rr - cumr[j*S+s]
        i = ul[j*S+s]; U[x] = M[Moff[i] + a*C[i] + cst[i*S+s] + b]
    if hasR:                                                           # ---- sr_build
        ptr, ein, ev = csr(md.rstacks[md.site_rs[r]].ops, S); SR = np.zeros(int(SRoff[-1]))
        for x in range(len(SR)):
            j = upper(SRoff, nR, x); loc = x - SRoff[j]; W = WR[j]; rr, w = divmod(int(loc), int(W)); t = -1
            for tt in range(NTR):
                s_ = srcR[j*NTR+tt]
                if s_ < 0:
                    continue
                w0 = wstR[j*NTR+tt]
                if w0 <= w < w0 + DR[s_]:
                    t, b, js = tt, w - w0, s_; break
            if t < 0:
                continue
            sp = upper(cumr[j*S:(j+1)*S], S, rr); a = rr - cumr[j*S+sp]; acc = 0.0
            for e in range(ptr[t*S+sp], ptr[t*S+sp+1]):
                rs = rst[js*S + ein[e]]
                if rs >= 0:
                    acc += ev[e]*U[Uoff[js] + (rs + a)*DR[js] + b]
            SR[x] = acc
    for j in range(nR):                                                # ---- products of the right half
        Dj, Rj, W = int(DR[j]), int(R[j]), int(WR[j])
        if Dj and Rj:
            O = oU[Uoff[j]:Uoff[j+1]].reshape(Rj, Dj); Uj = U[Uoff[j]:Uoff[j+1]].reshape(Rj, Dj)
            if r < md.n-1:
                O += Uj @ RH[p.RHoff[j]:p.RHoff[j+1]].reshape(Dj, Dj)
            if W and hasR:
                O += SR[SRoff[j]:SRoff[j+1]].reshape(Rj, W) @ GR[p.GRoff[j]:p.GRoff[j+1]].reshape(W, Dj)
    ptr, ein, ev = csr([md.kinds[md.site_kind[r]].H], S); add = np.zeros(n)   # ---- hr_add
    for x in range(n):
        j = upper(Uoff, nR, x); loc = x - Uoff[j]; Dj = DR[j]; rr, b = divmod(int(loc), int(Dj)); sp = upper(cumr[j*S:(j+1)*S], S, rr); a = rr - cumr[j*S+sp]; acc = 0.0
        for e in range(ptr[sp], ptr[sp+1]):
            rs = rst[j*S + ein[e]]
            if rs >= 0:
                acc += ev[e]*U[Uoff[j] + (rs + a)*Dj + b]
        add[x] = acc
    oU += add
    for x in range(n):                                                 # ---- u2m_add
        i = upper(Moff, nL, x); loc = x - Moff[i]; Ci = C[i]; a, c = divmod(int(loc), int(Ci)); s = upper(cumc[i*S:(i+1)*S], S, c); b = c - cumc[i*S+s]; j = mr[i*S+s]
        out[x] += oU[Uoff[j] + (rst[j*S+s] + a)*DR[j] + b]
    return out


if __name__ == '__main__':
    Lx, c, D = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (3, 2, 12)
    lat = ed.kagome_cylinder(Lx, c, tail=True); model = ed.Model(lat); rng = np.random.default_rng(3); worst = 0.0
    for nup, K in ((lat.N//2, 0), (lat.N//2+1, 1 % c)):
        eng = ed.DMRG(model, nup, K); eng.sweep(D, 1e-2, 1e-6); eng.sweep(D, 1e-2, 1e-6); _p = lambda a: a.ctypes.data
        for r in range(eng.n):
            p = eng.plan[r]; n = p.size; v = rng.standard_normal(n); ref = np.zeros(n); U = np.empty(n); oU = np.empty(n)
            eng.lib.heff(p.ref, _p(eng.LH[r]), _p(eng.GL[r]), _p(eng.RH[r+1]), _p(eng.GR[r+1]), _p(eng.Q[r]), _p(eng.P[r+1]), _p(v), _p(ref), _p(U), _p(oU))
            got = product(eng, r, v); d = float(np.abs(got-ref).max()); worst = max(worst, d)
            print(f'  nup={nup} K={K} layer {r:2d} ({p.c.S:3d} states, tensor {n:6d}, labels {p.nL}/{p.nR}): largest difference {d:.1e} (largest entry {np.abs(ref).max():.2f})')
            if r < eng.n-1:
                eng.move_right(r, D, 0.0)
    print(f'{lat.name}, D = {D}: largest difference over all layers {worst:.1e}')
