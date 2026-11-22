#!/usr/bin/env python3
"""edc_rows -- the "ring of rows" engine of the ED compiler: total spin S = 0 together with both translations.

An Lx x Ly torus is a ring of Ly rows of Lx sites.  Basis label = one multiplet (spin, row momentum, third label) per
row and a planar web that couples the row spins to a singlet.  x-translation is a phase,
y-translation and the mirrors are phased permutations of labels (orbit representatives), the Hamiltonian is sparse and,
in sectors with a definite mirror parity, real.

Status of this file (2026-11-22): the set-up (web-move tables, representatives, sector matrix) is the verified Python
logic of the prototype wrapped in build_sector(); the product y = C x is a generated C/OpenMP or CUDA kernel;
the eigenvalue step is a restarted Arnoldi iteration written once for numpy and cupy (the basis is not orthogonal, so
the sector matrix is not symmetric).  Nearest-neighbour Heisenberg model, square (and triangular) tori, even row length.
"""
import sys, os, math, itertools, json, time, ctypes, hashlib, subprocess
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla


def build_sector(Lx, Ly, tri=False, Kx=0, ky=0, py_=1, px_=1, rowbasis='dimer', verbose=False, device='cpu', kagome=False):
    """Sector matrix C (scipy CSR) with  H |label~> = sum C[label', label] |label'~>  for the orbit representatives of the
    sector: total x-momentum Kx, y-momentum ky (integers, units 2 pi / L), mirror parities px_, py_ where they apply."""
    # kagome=True: Lx counts unit cells; a row holds Lx cells = 3 Lx sites (cell x: A = 3x, B = 3x+1 on the chain, C = 3x+2 the
    # apex).  Bonds inside a row: A_x-B_x, B_x-A_{x+1}, A_x-C_x, B_x-C_x; between rows: C_x(r)-A_x(r+1), C_x(r)-B_{x-1}(r+1).
    # The row translation moves by one cell, so momenta are in units of 2 pi / (number of cells).
    L1 = Lx; NK = Lx
    if kagome:
        assert L1 >= 2 and rowbasis in ('energy', 'trimer') and not tri
        Lx = 3*L1
        A_ = lambda x: 3*(x % L1); B_ = lambda x: 3*(x % L1)+1; C_ = lambda x: 3*(x % L1)+2
        ROWBONDS = [b for x in range(L1) for b in ((A_(x), B_(x)), (B_(x), A_(x+1)), (A_(x), C_(x)), (B_(x), C_(x)))]
        UPBONDS = [(C_(x), A_(x)) for x in range(L1)]+[(C_(x), B_(x-1)) for x in range(L1)]
        MIRROR = {}
        for x in range(L1):
            MIRROR[A_(x)] = B_(-x); MIRROR[B_(x)] = A_(-x); MIRROR[C_(x)] = C_(-x)
    else:
        ROWBONDS = [(i, (i+1) % Lx) for i in range(Lx if Lx > 2 else 1)]
        MIRROR = {i: (-i) % Lx for i in range(Lx)}
    STEP = 3 if kagome else 1
    assert Lx % 2 == 0, 'even row length only (integer row spins)'
    N = Lx*Ly; D = 1 << Lx; t00 = time.time(); rng = np.random.default_rng(7)
    bit = lambda s, i: (s >> i) & 1
    log = (lambda *a: print('#', *a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    # ================================================================ rows (2^Lx states, done once)
    Sp = np.zeros((D, D)); T = np.zeros((D, D)); Hrow = np.zeros((D, D)); Px = np.zeros((D, D))
    for s in range(D):
        t = 0; u = 0
        for i in range(Lx):
            if not bit(s, i):
                Sp[s | (1 << i), s] += 1
            else:
                t |= 1 << ((i+STEP) % Lx); u |= 1 << MIRROR[i]
        T[t, s] = 1; Px[u, s] = 1
        for i, j in ROWBONDS:
            if bit(s, i) == bit(s, j):
                Hrow[s, s] += 0.25
            else:
                Hrow[s, s] -= 0.25; Hrow[s ^ ((1 << i) | (1 << j)), s] += 0.5
    Sm = Sp.T; nup = np.array([bin(s).count('1') for s in range(D)])
    c1, c2 = rng.standard_normal(2)
    ROWBASIS = rowbasis
    if ROWBASIS == 'dimer':
        Dop = np.zeros((D, D))
        for s_ in range(D):
            for i in range(Lx//2):
                jj = i+Lx//2; b1, b2 = bit(s_, i), bit(s_, jj)
                Dop[s_ ^ (((b1 ^ b2) << i) | ((b1 ^ b2) << jj)), s_] += 1
        Agen = Dop+c1*(T+T.T)/2+c2*(T-T.T)/(2j)
    elif ROWBASIS == 'trimer':                                    # kagome: S^2 of every up triangle and of its A-B pair
        def _exch(pairs):
            M = np.zeros((D, D))
            for s_ in range(D):
                for i, jj in pairs:
                    if bit(s_, i) == bit(s_, jj):
                        M[s_, s_] += 0.25
                    else:
                        M[s_, s_] -= 0.25; M[s_ ^ ((1 << i) | (1 << jj)), s_] += 0.5
            return M
        Q1 = _exch([b for x in range(L1) for b in ((A_(x), B_(x)), (A_(x), C_(x)), (B_(x), C_(x)))]); Q2 = _exch([(A_(x), B_(x)) for x in range(L1)])
        Agen = Q1+rng.standard_normal()*Q2+0.37*c1*(T+T.T)/2+0.37*c2*(T-T.T)/(2j)
    else:
        Agen = Hrow+c1*(T+T.T)/2+c2*(T-T.T)/(2j)
    spins = []; mk = {}; mE = {}; U = []; pos = {}
    for a in range(0, Lx+1, 2):
        idx = np.nonzero(nup == (Lx+a)//2)[0]
        u, sv, vh = np.linalg.svd(Sp[:, idx])
        null = vh[np.sum(sv > 1e-9):].conj().T
        if null.shape[1] == 0:
            continue
        Q = np.zeros((D, null.shape[1]), complex); Q[idx] = null
        w, v = np.linalg.eigh(Q.conj().T @ Agen @ Q)
        spins.append(a); mk[a] = []; mE[a] = []
        for c in range(v.shape[1]):
            hw = Q @ v[:, c]
            mk[a].append(int(round(np.angle(np.vdot(hw, T @ hw))/(2*math.pi/NK))) % NK); mE[a].append(float(np.vdot(hw, Hrow @ hw).real))
            pos[(a, c)] = len(U); S = a/2; cur = hw
            for d in range(a+1):
                U.append(cur)
                if d < a:
                    M = S-d
                    cur = Sm @ cur/math.sqrt(S*(S+1)-M*(M-1))
    U = np.stack(U, axis=1); assert np.allclose(U.conj().T @ U, np.eye(D), atol=1e-9)
    mult = {a: len(mk[a]) for a in spins}
    HOFF = {}                                                        # row Hamiltonian between different multiplets of the same spin and momentum
    for a in spins:
        lst = []
        for i in range(mult[a]):
            for i2 in range(mult[a]):
                if i2 != i:
                    val = np.vdot(U[:, pos[(a, i2)]], Hrow @ U[:, pos[(a, i)]])
                    if abs(val) > 1e-10:
                        assert mk[a][i] == mk[a][i2]
                        lst.append((i2, i, val))
        HOFF[a] = lst
    ops = []
    for x in range(Lx):
        spx = np.zeros((D, D))
        for s in range(D):
            if not bit(s, x):
                spx[s | (1 << x), s] = 1
        ops.append([U.conj().T @ o @ U for o in ((spx+spx.T)/2, (spx-spx.T)/(2j), np.diag([0.5 if bit(s, x) else -0.5 for s in range(D)]))])
    tau = {}; rho = {}; werr = 0.0
    for a in spins:
        for a2 in spins:
            if abs(a2-a) > 2 or (a == 0 and a2 == 0):
                continue
            R = np.zeros((Lx, mult[a2], mult[a]), complex)
            for i2 in range(mult[a2]):
                for i in range(mult[a]):
                    for x in range(Lx):
                        blk = np.stack([ops[x][al][pos[(a2, i2)]:pos[(a2, i2)]+a2+1, pos[(a, i)]:pos[(a, i)]+a+1] for al in range(3)])
                        if (a, a2) not in tau and np.linalg.norm(blk) > 1e-8:
                            tau[(a, a2)] = blk/np.linalg.norm(blk)
                        if (a, a2) in tau:
                            r_ = np.vdot(tau[(a, a2)], blk); R[x, i2, i] = r_
                            werr = max(werr, float(np.linalg.norm(blk-r_*tau[(a, a2)])))
            if (a, a2) in tau:
                R[np.abs(R) < 1e-11] = 0; rho[(a, a2)] = R
    Pm = U.conj().T @ Px @ U; ibar = {}; iphase = {}
    for a in spins:
        ibar[a] = np.zeros(mult[a], np.int64); iphase[a] = np.zeros(mult[a], complex)
        for i in range(mult[a]):
            col = Pm[:, pos[(a, i)]]; j = int(np.argmax(np.abs(col)))
            assert abs(abs(col[j])-1) < 1e-8
            ibar[a][i] = [i2 for i2 in range(mult[a]) if pos[(a, i2)] == j][0]; iphase[a][i] = col[j]
    rowpsi = {}                                                      # psi(a, i) with psi^2 = gamma / (reflection phase): removes the phase of a mirror-symmetrised label
    for a in spins:
        rowpsi[a] = np.zeros(mult[a], complex)
        for i in range(mult[a]):
            gam = np.vdot(U[:, pos[(a, int(ibar[a][i]))]], np.conj(U[:, pos[(a, i)]]))
            assert abs(abs(gam)-1) < 1e-8, 'complex conjugation does not map the multiplet to its mirror partner'
            rowpsi[a][i] = np.exp(0.5j*np.angle(gam/iphase[a][i]))
    log(f'rows: multiplets {mult}, Wigner-Eckart residual {werr:.1e}')

    def pair_amplitudes(a, b, a2, b2):
        if (a, a2) not in rho or (b, b2) not in rho:
            return None
        if kagome:
            g = sum(np.einsum('pi,qj->pqij', rho[(a, a2)][i], rho[(b, b2)][j]) for i, j in UPBONDS)
        else:
            g = np.einsum('xpi,xqj->pqij', rho[(a, a2)], rho[(b, b2)])
        if tri:
            g = g+np.einsum('xpi,xqj->pqij', rho[(a, a2)], np.roll(rho[(b, b2)], -1, axis=0))
        g[np.abs(g) < 1e-10] = 0
        return g if np.any(g) else None
    GAMP = {(a, b, a2, b2): pair_amplitudes(a, b, a2, b2) for a in spins for b in spins for a2 in spins for b2 in spins}
    GAMP = {k: v for k, v in GAMP.items() if v is not None}

    # ================================================================ webs
    def stack_matching(avec, cvec):
        """Non-crossing matching with cvec[x] closers followed by avec[x]-cvec[x] openers at leg x; returns pairs of points."""
        stack = []; pairs = []; p = 0
        for a, c in zip(avec, cvec):
            for _ in range(c):
                pairs.append((stack.pop(), p)); p += 1
            for _ in range(a-c):
                stack.append(p); p += 1
        assert not stack
        return tuple(sorted(pairs))

    def chords_of(avec, cvec):
        grp = [x for x, a in enumerate(avec) for _ in range(a)]
        n = {}
        for p, q in stack_matching(avec, cvec):
            key = (grp[p], grp[q]); n[key] = n.get(key, 0)+1
        return n

    def closers_of(L, n):
        c = [0]*L
        for (x, y), m in n.items():
            c[max(x, y)] += m
        return tuple(c)

    def web_tensor(avec, match):
        """Components of a web on |S_r, M_r> (index = number of down spins of leg r); all 2^strands orientations at once."""
        grp = [r for r, a in enumerate(avec) for _ in range(a)]; shape = [a+1 for a in avec]; n_ = len(match)
        if n_ == 0:
            Wt = np.ones(shape)
        else:
            Bm = (np.arange(1 << n_)[:, None] >> np.arange(n_)[None, :]) & 1
            st_ = np.array([int(np.prod(shape[r+1:])) for r in range(len(avec))])
            gp = np.array([grp[p_] for p_, q_ in match]); gq = np.array([grp[q_] for p_, q_ in match])
            idx_ = ((1-Bm)*st_[gq][None, :]+Bm*st_[gp][None, :]).sum(axis=1)       # bit 0: first point up, second down; bit 1: reversed (-)
            Wt = np.bincount(idx_, weights=1.0-2.0*(Bm.sum(axis=1) & 1), minlength=int(np.prod(shape))).reshape(shape)
        for r, a in enumerate(avec):
            sh_ = [1]*len(avec); sh_[r] = a+1
            Wt = Wt*np.array([1/math.sqrt(math.comb(a, d)) for d in range(a+1)]).reshape(sh_)
        return Wt

    def all_paths(avec):
        """All closer vectors of webs on the legs avec."""
        L = len(avec); res = []
        suffix = [sum(avec[x:]) for x in range(L+1)]
        def rec(x, h, cur):
            if x == L:
                if h == 0:
                    res.append(tuple(cur))
                return
            for c in range(0, min(h, avec[x])+1):
                h2 = h+avec[x]-2*c
                if h2 <= suffix[x+1]:
                    cur.append(c); rec(x+1, h2, cur); cur.pop()
        rec(0, 0, [])
        return res

    MODEL = {}; TENS = {}
    # local patterns depend only on the row length, the row labels and the lattice type: keep them on disk between runs
    pfile = os.path.join(os.environ.get('EDC_CACHE', os.path.expanduser('~/.cache/edc')), f'rows_patterns_{"k" if kagome else ""}{Lx}_{ROWBASIS}.npz')
    tau_sig = np.concatenate([tau[k].ravel() for k in sorted(tau)])
    if os.path.exists(pfile) and not os.environ.get('EDC_ROWS_PY'):
        try:
            import pickle
            zf = np.load(pfile, allow_pickle=True)
            if zf['tau'].shape == tau_sig.shape and np.allclose(zf['tau'], tau_sig, atol=1e-12):
                MODEL.update(pickle.loads(zf['model'].tobytes()))
        except Exception:
            MODEL.clear()
    n_cached = len(MODEL)
    def model_tensors(tup):
        if tup not in TENS:
            paths = all_paths(tup)
            tens = np.stack([web_tensor(tup, stack_matching(tup, c)).reshape(-1) for c in paths], axis=1) if paths else np.zeros((math.prod(a+1 for a in tup), 0))
            TENS[tup] = (paths, tens, np.linalg.pinv(tens) if paths else None)
        return TENS[tup]

    def intertwine(tup, cvec, x, y, a2, b2):
        """Coefficients of  sum_alpha tau_alpha(leg x) tau_alpha(leg y) |web>  on the webs of the new tuple: list of (closers', coefficient)."""
        paths, tens, _ = model_tensors(tup)
        X = tens[:, paths.index(tuple(cvec))].reshape([a+1 for a in tup])
        out = 0
        for al in range(3):
            Y = np.tensordot(X, tau[(tup[x], a2)][al], axes=([x], [1])); Y = np.moveaxis(Y, -1, x)
            Y = np.tensordot(Y, tau[(tup[y], b2)][al], axes=([y], [1])); Y = np.moveaxis(Y, -1, y)
            out = out+Y
        new = list(tup); new[x] = a2; new[y] = b2; new = tuple(new)
        paths2, tens2, pinv2 = model_tensors(new)
        if not paths2:
            assert np.max(np.abs(out)) < 1e-9
            return new, []
        co = pinv2 @ out.reshape(-1)
        assert np.max(np.abs(tens2 @ co-out.reshape(-1))) < 1e-8
        return new, [(paths2[j], complex(co[j])) for j in np.nonzero(np.abs(co) > 1e-10)[0]]

    def local_move(a, b, a2, b2, nd, sig, th, shared):
        """Model: leg 0 (a strands), leg 1 (b strands), then one leg per bundle of leg 1 (sizes sig, nearest first) and per bundle of
        leg 0 (sizes th, in the same cyclic order); `shared`: the last bundle of leg 1 and the first of leg 0 end on the same leg.
        Returns list of (coefficient, chords among model legs)."""
        key = (a, b, a2, b2, nd, sig, th, shared)
        if key not in MODEL:
            far = list(sig)+list(th)
            if shared:
                far = list(sig[:-1])+[sig[-1]+th[0]]+list(th[1:])
            tup = (a, b)+tuple(far)
            n = {}
            if nd:
                n[(0, 1)] = nd
            for j, m in enumerate(sig):
                n[(1, 2+j)] = m
            off = 2+len(sig)-(1 if shared else 0)
            for i, m in enumerate(th):
                n[(0, off+i)] = m
            new, terms = intertwine(tup, closers_of(len(tup), n), 0, 1, a2, b2)
            MODEL[key] = [(co, chords_of(new, c)) for c, co in terms]
        return MODEL[key]

    def apply_move(avec, n, r, a2, b2):
        """Bonds between legs r and r+1 (cyclic) with a_r -> a2, a_{r+1} -> b2: list of (new avec, new chords, coefficient)."""
        L = len(avec); s = (r+1) % L
        get = lambda x, y: n.get((min(x, y), max(x, y)), 0)
        order = [(s+1+j) % L for j in range(L-2)]                     # the other legs, going on from leg r+1
        sig_l = [(x, get(s, x)) for x in order if get(s, x)]; th_l = [(x, get(r, x)) for x in order if get(r, x)]
        shared = bool(sig_l and th_l and sig_l[-1][0] == th_l[0][0])
        terms = local_move(avec[r], avec[s], a2, b2, get(r, s), tuple(m for _, m in sig_l), tuple(m for _, m in th_l), shared)
        if not terms:
            return []
        legs = [r, s]+[x for x, _ in sig_l]+[x for x, _ in th_l[(1 if shared else 0):]]
        base = {k: m for k, m in n.items() if r not in k and s not in k}
        new_avec = list(avec); new_avec[r] = a2; new_avec[s] = b2; new_avec = tuple(new_avec)
        res = []
        for co, ch in terms:
            n2 = dict(base)
            for (x, y), m in ch.items():
                X, Y = legs[x], legs[y]; k = (min(X, Y), max(X, Y)); n2[k] = n2.get(k, 0)+m
            res.append((new_avec, n2, co))
        return res

    # ================================================================ classes = (row spins, web); labels = class x row multiplets with momentum Kx
    PYPATH = bool(os.environ.get('EDC_ROWS_PY'))                 # Python routines (checks) instead of the generated C
    classes = []; cid = {}; by_tuple = {}
    if PYPATH:
        for avec in itertools.product(spins, repeat=Ly):
            ps = all_paths(avec)
            if ps:
                by_tuple[avec] = []
                for c in ps:
                    cid[(avec, c)] = len(classes); by_tuple[avec].append(len(classes)); classes.append((avec, c))
    else:                                                        # the same enumeration, in the same order, in C
        classes = enumerate_classes(Ly, spins)
        first = np.flatnonzero(np.concatenate(([True], np.any(classes.ca[1:] != classes.ca[:-1], axis=1))))
        tcount = np.diff(np.append(first, len(classes)))
        by_tuple = {tuple(int(v) for v in classes.ca[f]): range(int(f), int(f+c)) for f, c in zip(first, tcount)}
    ncl = len(classes)
    keep = {}; nkeep = {}
    for avec in by_tuple:
        if PYPATH:
            dims = [mult[a] for a in avec]; ks = np.zeros(dims, np.int64)
            for r, a in enumerate(avec):
                shape = [1]*Ly; shape[r] = dims[r]; ks = ks+np.array(mk[a]).reshape(shape)
            ok = (ks.reshape(-1) % NK) == Kx
            comp = np.full(ok.size, -1, np.int64); comp[ok] = np.arange(ok.sum()); keep[avec] = comp; nkeep[avec] = int(ok.sum())
    if not PYPATH:                                               # number of multiplet choices with total row momentum Kx, all tuples at once
        Tt = np.array(list(by_tuple), np.int64).reshape(len(by_tuple), Ly); vec = np.zeros((len(Tt), NK), np.int64); vec[:, 0] = 1
        for r in range(Ly):
            nv = np.zeros_like(vec)
            for a in spins:
                m_ = Tt[:, r] == a
                if m_.any():
                    nv[m_] = sum(np.roll(vec[m_], k_, axis=1) for k_ in mk[a])
            vec = nv
        nkeep = {avec: int(v) for avec, v in zip(by_tuple, vec[:, Kx])}
    KIDX = {}
    def kept(avec):
        if avec not in KIDX:
            KIDX[avec] = np.nonzero(keep[avec] >= 0)[0]
        return KIDX[avec]
    coff = np.zeros(ncl+1, np.int64)
    if PYPATH:
        for q, (avec, c) in enumerate(classes):
            coff[q+1] = coff[q]+nkeep[avec]
        nall = sum(math.prod(mult[a] for a in avec) for avec, c in classes)
    else:
        np.cumsum(np.repeat([nkeep[avec] for avec in by_tuple], tcount), out=coff[1:])
        nall = sum(math.prod(mult[a] for a in avec)*len(ids) for avec, ids in by_tuple.items())
    dim = int(coff[-1]); cat = math.comb(N, N//2)//(N//2+1)
    log(f'webs: {len(by_tuple)} spin tuples, {ncl} classes, labels {nall} (Catalan {cat}), Kx = {Kx}: {dim}, {time.time()-t00:.1f} s')
    assert nall == cat

    # web part of the Hamiltonian: for every class, pair of rows and spin change, the target classes
    t1 = time.time()
    WT = web_tables(classes, Ly, GAMP, local_move)             # generated C: one pass for the local patterns, one for the targets
    log(f'web moves: {WT.total} entries, {len(MODEL)} local patterns ({n_cached} from the cache), {time.time()-t1:.1f} s')
    if len(MODEL) > n_cached and not os.environ.get('EDC_ROWS_PY'):
        import pickle
        os.makedirs(os.path.dirname(pfile), exist_ok=True)
        np.savez(pfile, tau=tau_sig, model=np.frombuffer(pickle.dumps(MODEL), dtype=np.uint8))

    # ================================================================ symmetry: y-translation and mirrors on labels
    def class_map(axes):
        """Class q -> class and sign when new leg x carries old leg axes[x].  A strand changes sign when the order of its two
        end legs is reversed (the singlet is antisymmetric); with integer row spins a rotation gives +1, a mirror (-1)^(strands)."""
        inv = {old: newx for newx, old in enumerate(axes)}; out = np.zeros(ncl, np.int64); sgn = np.ones(ncl)
        for q, (avec, c) in enumerate(classes):
            n = chords_of(avec, c)
            n2 = {}; flips = 0
            for (x, y), m in n.items():
                X, Y = inv[x], inv[y]
                if X > Y:
                    flips += m
                n2[(min(X, Y), max(X, Y))] = m
            out[q] = cid[(tuple(avec[o] for o in axes), closers_of(Ly, n2))]; sgn[q] = -1.0 if flips % 2 else 1.0
        return out, sgn

    def unpack(lab):
        q = np.searchsorted(coff, lab, side='right')-1
        return q, lab-coff[q]

    def groups(lab):
        """Positions of the labels grouped by class: yields (class, positions) with one sort instead of one scan per class."""
        q, loc = unpack(lab)
        order = np.argsort(q, kind='stable'); qs = q[order]
        cut = np.concatenate(([0], np.flatnonzero(np.diff(qs))+1, [len(qs)]))
        for a_, b_ in zip(cut[:-1], cut[1:]):
            if b_ > a_:
                yield int(qs[a_]), order[a_:b_], loc

    def act(lab, axes, cmap, reflect):
        """Image labels and phases under a leg permutation (axes) optionally preceded by the row reflection x -> -x."""
        out = np.zeros_like(lab); ph = np.ones(len(lab), complex)
        for qq, sel, loc in groups(lab):
            avec = classes[qq][0]; dims = [mult[a] for a in avec]
            flat = kept(avec)[loc[sel]]
            I = list(np.unravel_index(flat, dims)) if Ly > 0 else []
            if reflect:
                f = np.ones(len(sel), complex)
                for r, a in enumerate(avec):
                    f = f*iphase[a][I[r]]; I[r] = ibar[a][I[r]]
                ph[sel] = f
            new = tuple(avec[o] for o in axes); ndims = [mult[a] for a in new]
            f2 = np.ravel_multi_index(tuple(I[o] for o in axes), ndims)
            out[sel] = coff[cmap[0][qq]]+keep[new][f2]; ph[sel] *= cmap[1][qq]
        return out, ph

    def columns(labs):
        """Images of H on the given labels: (row label, column position, value) arrays plus the diagonal."""
        rows_, cols_, vals_ = [], [], []; diag = np.zeros(len(labs))
        for qq, sel, loc in groups(labs):
            avec = classes[qq][0]; dims = [mult[a] for a in avec]
            flat = kept(avec)[loc[sel]]
            I = np.array(np.unravel_index(flat, dims)).reshape(Ly, -1)
            for r, a in enumerate(avec):
                diag[sel] += np.array(mE[a])[I[r]]
            strides = [math.prod(dims[x+1:]) for x in range(Ly)]
            for r, a in enumerate(avec):                             # row Hamiltonian, off-diagonal in the multiplet index (not in the 'energy' basis)
                for i2, i1, val in HOFF[a]:
                    m_ = np.nonzero(I[r] == i1)[0]
                    if len(m_):
                        ct = keep[avec][flat[m_]+(i2-i1)*strides[r]]; assert np.all(ct >= 0)
                        rows_.append(coff[qq]+ct); cols_.append(sel[m_]); vals_.append(np.full(len(m_), val))
            for r in range(Ly):
                s = (r+1) % Ly
                for a2 in spins:
                    for b2 in spins:
                        lst = WT.get((qq, r, a2, b2))
                        if not lst:
                            continue
                        g = GAMP[(avec[r], avec[s], a2, b2)]          # [i', j', i, j]
                        new = list(avec); new[r] = a2; new[s] = b2; new = tuple(new)
                        ndims = [mult[x] for x in new]; nstr = [math.prod(ndims[x+1:]) for x in range(Ly)]
                        gsub = g[:, :, I[r], I[s]]                    # [i', j', label]
                        ip, jp, li = np.nonzero(gsub)
                        if len(li) == 0:
                            continue
                        restflat = flat[li]-I[r][li]*strides[r]-I[s][li]*strides[s]
                        # re-express the other rows' indices with the new strides
                        tgt = ip*nstr[r]+jp*nstr[s]
                        for x in range(Ly):
                            if x != r and x != s:
                                tgt = tgt+I[x][li]*nstr[x]
                        ct = keep[new][tgt]; assert np.all(ct >= 0)
                        gv = gsub[ip, jp, li]
                        for q2, co in lst:
                            rows_.append(coff[q2]+ct); cols_.append(sel[li]); vals_.append(gv*co)
        return np.concatenate(rows_), np.concatenate(cols_), np.concatenate(vals_), diag

    def label_psi(labs):
        out_ = np.ones(len(labs), complex)
        for qq, sel, loc in groups(labs):
            avec = classes[qq][0]; dims = [mult[a] for a in avec]
            I = np.array(np.unravel_index(kept(avec)[loc[sel]], dims)).reshape(Ly, -1); f = np.ones(len(sel), complex)
            for r, a in enumerate(avec):
                f = f*rowpsi[a][I[r]]
            out_[sel] = f
        return out_

    if not os.environ.get('EDC_ROWS_PY'):                        # generated C for class maps, label images, matrix columns
        class_map, act, columns, label_psi, orbit_min, assemble_real, tabs = sector_routines(
            NK, Ly, Kx, spins, mult, mk, mE, ibar, iphase, rowpsi, classes, by_tuple, GAMP, HOFF, WT, coff)   # NK: momentum modulus
    else:
        orbit_min = assemble_real = tabs = None
    log(f'tables and routines ready, {time.time()-t00:.1f} s')
    ts = time.time()
    ident = list(range(Ly)); rot = [(x-1) % Ly for x in range(Ly)]; mir = [(-x) % Ly for x in range(Ly)]
    cm_rot = class_map(rot); cm_mir = class_map(mir); cm_id = (np.arange(ncl), np.ones(ncl))
    labs = np.arange(dim)
    # which mirrors are symmetries: test on the columns of a sample of labels
    def commutes(axes, cmap, reflect):
        smp = rng.choice(dim, min(dim, 200), replace=False)
        img, ph = act(smp, axes, cmap, reflect)
        r1, c1_, v1, d1 = columns(smp); r2, c2_, v2, d2 = columns(img)
        g1, p1 = act(r1, axes, cmap, reflect)
        A1 = sp.coo_matrix((v1*p1, (g1, c1_)), shape=(dim, len(smp))).tocsc(); A2 = sp.coo_matrix((v2*ph[c2_], (r2, c2_)), shape=(dim, len(smp))).tocsc()
        return float(abs(A1-A2).max()) < 1e-9 and float(np.max(np.abs(d1-d2))) < 1e-9
    assert commutes(rot, cm_rot, False)
    both = commutes(mir, cm_mir, False) and commutes(ident, cm_id, True)
    prodm = (not both) and commutes(mir, cm_mir, True)            # kagome strips: possibly no mirror at all
    if (device == 'gpu' and tabs is not None and both and (2*(ky % Ly)) % Ly == 0 and Kx in (0, NK//2)
            and dim >= int(os.environ.get('EDC_ROWS_GPU_MIN', 100000))):
        g = gpu_sector(tabs, cm_rot, cm_mir, cm_id, dim, Ly, ky % Ly, py_, px_, int(os.environ.get('CHUNK', 400000)), log)
        if g['imag'] < 1e-9:                                     # representatives, columns and assembly on the GPU
            return dict(C=None, gpu=g, real=True, dim=g['dim'], group=g['group'], mirrors='both', singlets=cat, kx_dim=dim, classes=ncl,
                        local_patterns=len(MODEL), seconds=time.time()-t00)
    # group elements = rotations x mirror combinations, generated on the fly (nothing of size |G| x dim is stored).
    # One-dimensional characters: chi(T_y^n M_y^p M_x^q) = exp(2 pi i ky n / Ly) py^p px^q  (env KY, PY, PX; default trivial).
    ky = ky % Ly
    real_ky = (2*ky) % Ly == 0                                    # the leg mirror maps ky -> -ky
    comp2 = lambda g2, g1: (g2[0][g1[0]], g2[1][g1[0]]*g1[1])     # apply g1 first
    base = [((labs, np.ones(dim, complex)), 1.0)]
    if both:
        gx = act(labs, ident, cm_id, True); base.append((gx, px_))
        if real_ky:
            gy = act(labs, mir, cm_mir, False); base += [(gy, py_), (comp2(gy, gx), py_*px_)]
    elif real_ky and prodm:
        base.append((act(labs, mir, cm_mir, True), py_*px_))
    rotg = act(labs, rot, cm_rot, False)
    ngroup = len(base)*Ly
    if orbit_min is not None:
        rep = orbit_min(np.stack([g[0] for g, ch in base]), rotg[0])
    else:
        rep = labs.copy()
        for g, ch in base:
            cur = g[0]
            for nrot in range(Ly):
                np.minimum(rep, cur, out=rep); cur = rotg[0][cur]
    reps = np.nonzero(rep == labs)[0]
    fac = np.zeros(dim, complex); srep = np.zeros(len(reps), complex)   # |l~> = fac[l] |rep(l)~>; s = coefficient of the representative in |rep~>
    for g, ch in base:
        cp_, cph = g[0][reps], g[1][reps]
        for nrot in range(Ly):
            chi = ch*np.exp(2j*math.pi*ky*nrot/Ly)
            fac[cp_] = chi/cph                                    # h |rep> = cph |l'>  ->  |l'~> = chi(h) / cph |rep~>
            stab = cp_ == reps; srep[stab] += np.conj(chi)*cph[stab]
            cph = rotg[1][cp_]*cph; cp_ = rotg[0][cp_]
    del base
    good = np.abs(srep) > 1e-9; reps = reps[good]
    newidx = np.full(dim, -1, np.int64); newidx[reps] = np.arange(len(reps))
    t_sym = time.time()-ts; ts = time.time()
    fast = None
    if assemble_real is not None and both and Kx in (0, NK//2):
        fast = assemble_real(reps, rep, newidx, fac, label_psi(reps), int(os.environ.get('CHUNK', 400000)))
        if fast[3] >= 1e-9:
            fast = None
    if fast is not None:
        nd = len(reps); Cg = sp.csr_matrix((fast[2], fast[1], fast[0]), shape=(nd, nd)); made_real = True; imag_left = fast[3]
        log(f'sector matrix (C assembly): dimension {nd}, {Cg.nnz} non-zeros, {time.time()-ts:.1f} s')
    else:
        log(f'representatives: {len(reps)} of {dim} labels, group {ngroup}, {time.time()-ts:.1f} s')
        CH = int(os.environ.get('CHUNK', 400000)); blocks = []; dgs = []; t_col = 0.0; t_asm = 0.0
        for c0 in range(0, len(reps), CH):
            tq = time.time(); rr, cc, vv, dg = columns(reps[c0:c0+CH]); t_col += time.time()-tq; tq = time.time()
            ri = newidx[rep[rr]]; okm = ri >= 0
            blk = sp.coo_matrix(((vv*fac[rr])[okm], (ri[okm], cc[okm])), shape=(len(reps), len(dg))).tocsc()
            blk.data[np.abs(blk.data) < 1e-10] = 0; blk.eliminate_zeros(); blocks.append(blk); dgs.append(dg)
            if len(reps) > CH:
                log(f'columns {c0+len(dg)}/{len(reps)}, {time.time()-ts:.0f} s')
            del rr, cc, vv, ri, okm; t_asm += time.time()-tq
        tq = time.time(); Cg = (sp.hstack(blocks, format='csc')+sp.diags(np.concatenate(dgs))).tocsr(); del blocks
        log(f'columns {t_col:.1f} s, assembly of the chunks {t_asm:.1f} s, final conversion {time.time()-tq:.1f} s')
        t_build = time.time()-ts; ts = time.time()
        nd = len(reps)
        nd = len(reps)
        made_real = False; imag_left = None
        if both and Kx in (0, NK//2):                                  # every symmetrised label is a real vector times a phase: absorb it
            Dg = label_psi(reps)
            Cr = (sp.diags(1/Dg) @ Cg @ sp.diags(Dg)).tocsr()
            imag_left = float(np.max(np.abs(Cr.data.imag))) if Cr.nnz else 0.0
            if imag_left < 1e-9:
                Cg = Cr.real.tocsr(); Cg.sort_indices(); made_real = True
    return dict(C=Cg, real=made_real, dim=nd, group=ngroup, mirrors='both' if both else 'product only', singlets=cat, kx_dim=dim,
                classes=ncl, local_patterns=len(MODEL), seconds=time.time()-t00)


# ---------------------------------------------------------------------------------------------------------------------
# web-move tables in C: for every web (class), every pair of adjacent legs and every change of the two row spins, the
# list of target webs.  Pass 1 reads the strand bundles around the two legs and returns a 64-bit key of the local
# pattern; the coefficients of the distinct patterns are computed once (small tensors, in Python); pass 2 rewires.
TABLES_C = r"""
#include <stdint.h>
#include <string.h>
#define ML 12
static void chords(int L, const int8_t *a, const int8_t *c, int n[ML][ML]) {
    int stack[128], sp = 0;
    memset(n, 0, sizeof(int) * ML * ML);
    for (int x = 0; x < L; x++) {
        for (int k = 0; k < c[x]; k++) { int y = stack[--sp]; n[y][x]++; n[x][y]++; }
        for (int k = 0; k < a[x] - c[x]; k++) stack[sp++] = x;
    }
}
static uint64_t pattern(int L, const int8_t *a, int n[ML][ML], int r, int a2, int b2, int *legs) {
    int s = (r + 1) % L, nsig = 0, nth = 0, sig[ML], th[ML], sl[ML], tl[ML], sh = 0, m = 0;
    for (int j = 0; j < L - 2; j++) { int x = (s + 1 + j) % L; if (n[s][x]) { sig[nsig] = n[s][x]; sl[nsig++] = x; } }
    for (int j = 0; j < L - 2; j++) { int x = (s + 1 + j) % L; if (n[r][x]) { th[nth] = n[r][x]; tl[nth++] = x; } }
    int shared = nsig && nth && sl[nsig - 1] == tl[0];
    uint64_t k = (uint64_t)1 << 63;
#define PUT(v, w) { k |= (uint64_t)(v) << sh; sh += (w); }
    PUT(a[r] / 2, 2) PUT(a[s] / 2, 2) PUT(a2 / 2, 2) PUT(b2 / 2, 2) PUT(n[r][s], 3) PUT(shared, 1) PUT(nsig, 3) PUT(nth, 3)
    for (int j = 0; j < nsig; j++) PUT(sig[j], 3)
    for (int j = 0; j < nth; j++) PUT(th[j], 3)
    legs[m++] = r; legs[m++] = s;
    for (int j = 0; j < nsig; j++) legs[m++] = sl[j];
    for (int j = shared ? 1 : 0; j < nth; j++) legs[m++] = tl[j];
    return k;
}
void rows_pass1(int64_t ncl, int L, const int8_t *ca, const int8_t *cc, const uint8_t *allow, uint64_t *keys) {
    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t q = 0; q < ncl; q++) {
        int n[ML][ML], legs[2 * ML]; const int8_t *a = ca + q * L;
        chords(L, a, cc + q * L, n);
        for (int r = 0; r < L; r++) {
            int s = (r + 1) % L;
            for (int t = 0; t < 16; t++) {
                int64_t slot = (q * L + r) * 16 + t; int a2 = 2 * (t >> 2), b2 = 2 * (t & 3);
                keys[slot] = 0;
                if (L == 2 && r == 1) continue;
                if (!allow[((a[r] / 2 * 4 + a[s] / 2) * 4 + a2 / 2) * 4 + b2 / 2]) continue;
                keys[slot] = pattern(L, a, n, r, a2, b2, legs);
            }
        }
    }
}
static int64_t paths(int L, const int8_t *a, const int *suffix, int x, int h, int8_t *cur, int8_t *ca, int8_t *cc, int64_t n) {
    if (x == L) { if (h == 0) { if (ca) { memcpy(ca + n * L, a, L); memcpy(cc + n * L, cur, L); } return n + 1; } return n; }
    int cm = h < a[x] ? h : a[x];
    for (int c = 0; c <= cm; c++) { int h2 = h + a[x] - 2 * c; if (h2 <= suffix[x + 1]) { cur[x] = (int8_t)c; n = paths(L, a, suffix, x + 1, h2, cur, ca, cc, n); } }
    return n;
}
/* distinct values of a real matrix (rounded to 1e-12) by hashing: codes[e] = index into table; returns the number of
   distinct values, or -1 when there are more than 65535 */
#include <math.h>
int64_t rows_value_codes(int64_t n, const double *va, uint16_t *codes, double *table) {
    enum { CAP = 1 << 18 };
    static int64_t key[CAP]; static int32_t idx[CAP];
    memset(idx, 0xff, sizeof(idx));
    int64_t nt = 0;
    for (int64_t e = 0; e < n; e++) {
        int64_t k = llround(va[e] * 1e12); uint64_t h = ((uint64_t)k * 0x9E3779B97F4A7C15ull) >> 46;
        while (idx[h] >= 0 && key[h] != k) h = (h + 1) & (CAP - 1);
        if (idx[h] < 0) { if (nt == 65535) return -1; key[h] = k; idx[h] = (int32_t)nt; table[nt++] = (double)k * 1e-12; }
        codes[e] = (uint16_t)idx[h];
    }
    return nt;
}
/* all webs: for every vector of strand numbers (first leg slowest) every vector of closers; count only when ca == 0 */
int64_t rows_enumerate(int L, int ns, const int8_t *spins, int8_t *ca, int8_t *cc) {
    int64_t n = 0, nt = 1;
    for (int i = 0; i < L; i++) nt *= ns;
    for (int64_t t = 0; t < nt; t++) {
        int8_t a[ML], cur[ML]; int suffix[ML + 1]; int64_t r = t;
        for (int x = L - 1; x >= 0; x--) { a[x] = spins[r % ns]; r /= ns; }
        suffix[L] = 0;
        for (int x = L - 1; x >= 0; x--) suffix[x] = suffix[x + 1] + a[x];
        n = paths(L, a, suffix, 0, 0, cur, ca, cc, n);
    }
    return n;
}
static int64_t find(const uint64_t *keys, int64_t n, uint64_t k) {
    int64_t lo = 0, hi = n - 1;
    while (lo <= hi) { int64_t mid = (lo + hi) / 2; if (keys[mid] == k) return mid; if (keys[mid] < k) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
int64_t rows_pass2(int64_t ncl, int L, const int8_t *ca, const int8_t *cc, const uint64_t *keys, int64_t np_, const uint64_t *pkeys,
                   const int32_t *pptr, const int32_t *tptr, const int8_t *trip, const uint64_t *ckeys, const int32_t *cids,
                   const int64_t *ptr, int32_t *out_cls, int32_t *out_term) {
    int64_t errors = 0;
    #pragma omp parallel for schedule(dynamic, 256) reduction(+:errors)
    for (int64_t q = 0; q < ncl; q++) {
        int n[ML][ML], n2[ML][ML], legs[2 * ML]; const int8_t *a = ca + q * L;
        chords(L, a, cc + q * L, n);
        for (int r = 0; r < L; r++) {
            int s = (r + 1) % L;
            for (int t = 0; t < 16; t++) {
                int64_t slot = (q * L + r) * 16 + t;
                if (!keys[slot]) continue;
                int a2 = 2 * (t >> 2), b2 = 2 * (t & 3);
                int64_t p = find(pkeys, np_, keys[slot]);
                if (p < 0) { errors++; continue; }
                pattern(L, a, n, r, a2, b2, legs);
                for (int32_t e = pptr[p]; e < pptr[p + 1]; e++) {
                    memcpy(n2, n, sizeof(n2));
                    for (int x = 0; x < L; x++) { n2[r][x] = n2[x][r] = 0; n2[s][x] = n2[x][s] = 0; }
                    for (int32_t j = tptr[e]; j < tptr[e + 1]; j++) {
                        int X = legs[trip[3 * j]], Y = legs[trip[3 * j + 1]], m = trip[3 * j + 2];
                        n2[X][Y] += m; n2[Y][X] += m;
                    }
                    uint64_t k = 0;
                    for (int y = 0; y < L; y++) {
                        int c = 0, ay = y == r ? a2 : (y == s ? b2 : a[y]);
                        for (int x = 0; x < y; x++) c += n2[x][y];
                        k |= (uint64_t)((ay / 2) | (c << 2)) << (5 * y);
                    }
                    int64_t z = find(ckeys, ncl, k), o = ptr[slot] + (e - pptr[p]);
                    if (z < 0) { errors++; z = 0; }
                    out_cls[o] = cids[z]; out_term[o] = e;
                }
            }
        }
    }
    return errors;
}
"""


class WebTables:
    """Targets of the web moves: get((class, leg, a2, b2)) -> list of (class', coefficient), as the dictionary it replaces."""
    def __init__(self, L, ptr, cls, coef):
        self.L, self.ptr, self.cls, self.coef, self.total = L, ptr, cls, coef, int(len(cls))

    def get(self, key, default=None):
        q, r, a2, b2 = key
        slot = (q*self.L+r)*16+(a2//2)*4+b2//2; s, e = int(self.ptr[slot]), int(self.ptr[slot+1])
        return list(zip(self.cls[s:e].tolist(), self.coef[s:e].tolist())) if e > s else default


class ClassArrays:
    """Webs as two int8 arrays [number of webs, legs]: strands per leg and closers per leg."""
    def __init__(self, ca, cc):
        self.ca, self.cc = ca, cc

    def __len__(self):
        return len(self.ca)


def enumerate_classes(L, spins):
    lib = _compile_c(TABLES_C); P = ctypes.c_void_p
    lib.rows_enumerate.argtypes = [ctypes.c_int, ctypes.c_int, P, P, P]; lib.rows_enumerate.restype = ctypes.c_int64
    sp_ = np.ascontiguousarray(spins, np.int8); n = lib.rows_enumerate(L, len(sp_), sp_.ctypes.data, None, None)
    ca = np.zeros((n, L), np.int8); cc = np.zeros((n, L), np.int8)
    lib.rows_enumerate(L, len(sp_), sp_.ctypes.data, ca.ctypes.data, cc.ctypes.data)
    return ClassArrays(ca, cc)


def web_tables(classes, L, GAMP, local_move):
    lib = _compile_c(TABLES_C); P = ctypes.c_void_p; I64 = ctypes.c_int64
    lib.rows_pass1.argtypes = [I64, ctypes.c_int, P, P, P, P]
    lib.rows_pass2.argtypes = [I64, ctypes.c_int, P, P, P, I64, P, P, P, P, P, P, P, P, P]; lib.rows_pass2.restype = I64
    ncl = len(classes); ptr_ = lambda a: a.ctypes.data
    if hasattr(classes, 'ca'):
        ca, cc = classes.ca, classes.cc
    else:
        ca = np.ascontiguousarray([c[0] for c in classes], np.int8).reshape(ncl, L); cc = np.ascontiguousarray([c[1] for c in classes], np.int8).reshape(ncl, L)
    assert L <= 12 and int(ca.max()) <= 6
    allow = np.zeros(256, np.uint8)
    for (a, b, a2, b2) in GAMP:
        allow[((a//2*4+b//2)*4+a2//2)*4+b2//2] = 1
    keys = np.zeros(ncl*L*16, np.uint64)
    lib.rows_pass1(ncl, L, ptr_(ca), ptr_(cc), ptr_(allow), ptr_(keys))
    nzk = keys != 0; pkeys = np.ascontiguousarray(np.unique(keys[nzk]))
    pptr, tptr, trip, tcoef = [0], [0], [], []
    for k in pkeys.tolist():                                     # decode the local pattern and compute its coefficients (cached)
        f = lambda w: None
        bits = [2, 2, 2, 2, 3, 1, 3, 3]; vals = []; sh = 0
        for w in bits:
            vals.append((k >> sh) & ((1 << w)-1)); sh += w
        a, b, a2, b2, nd, shared, nsig, nth = vals
        sig = tuple((k >> (sh+3*j)) & 7 for j in range(nsig)); sh += 3*nsig
        th = tuple((k >> (sh+3*j)) & 7 for j in range(nth))
        for co, ch in local_move(2*a, 2*b, 2*a2, 2*b2, nd, sig, th, bool(shared)):
            tcoef.append(co)
            for (x, y), m in ch.items():
                trip += [x, y, m]
            tptr.append(len(trip)//3)
        pptr.append(len(tcoef))
    pptr = np.ascontiguousarray(pptr, np.int32); tptr = np.ascontiguousarray(tptr, np.int32)
    trip = np.ascontiguousarray(trip if trip else [0, 0, 0], np.int8); tcoef = np.asarray(tcoef if tcoef else [0], complex)
    ck = np.zeros(ncl, np.uint64)
    for y in range(L):
        ck |= ((ca[:, y].astype(np.uint64)//2) | (cc[:, y].astype(np.uint64) << 2)) << np.uint64(5*y)
    order = np.argsort(ck); ckeys = np.ascontiguousarray(ck[order]); cids = np.ascontiguousarray(order, np.int32)
    nz = nzk; counts = np.zeros(len(keys), np.int64)
    pi = np.searchsorted(pkeys, keys[nz]); counts[nz] = pptr[pi+1]-pptr[pi]
    ptr = np.zeros(len(keys)+1, np.int64); np.cumsum(counts, out=ptr[1:])
    out_cls = np.zeros(max(1, int(ptr[-1])), np.int32); out_term = np.zeros(max(1, int(ptr[-1])), np.int32)
    err = lib.rows_pass2(ncl, L, ptr_(ca), ptr_(cc), ptr_(keys), len(pkeys), ptr_(pkeys), ptr_(pptr), ptr_(tptr), ptr_(trip), ptr_(ckeys),
                         ptr_(cids), ptr_(ptr), ptr_(out_cls), ptr_(out_term))
    assert err == 0, f'web tables: {err} targets not found'
    n_ = int(ptr[-1])
    wt = WebTables(L, ptr, out_cls[:n_], tcoef[out_term[:n_]]); wt.ca, wt.cc, wt.ckeys, wt.cids = ca, cc, ckeys, cids
    return wt


# ---------------------------------------------------------------------------------------------------------------------
# sector routines in C.  A label is (web class q, one digit per row); a digit is a row multiplet (spin, index).  Labels of
# a spin tuple t are numbered  toff[t] + (q - qfirst[t]) * nk[t] + rank(digits),  rank = position among the digit strings
# with total row momentum Kx (counted with the table cnt[t][row][momentum], as with Lin tables).
SECTOR_C = r"""
#include <stdint.h>
#include <string.h>
#define ML 12
typedef struct {
    int64_t L, Lx, Kx, M, nt;
    const int8_t *dA; const int32_t *dK, *dBar, *abase, *amult; const double *dE, *dPh, *dPsi;
    const int8_t *tup_a; const int64_t *cnt, *nk, *toff; const int32_t *qfirst, *ct;
} Z;
static inline int md(int64_t x, int64_t m) { return (int)(((x % m) + m) % m); }
void rows_cnt(const Z *z, int64_t *cnt) {
    int64_t L = z->L, Lx = z->Lx;
    for (int64_t t = 0; t < z->nt; t++) {
        int64_t *c = cnt + t * (L + 1) * Lx;
        for (int64_t k = 0; k < Lx; k++) c[L * Lx + k] = (k == 0);
        for (int64_t r = L - 1; r >= 0; r--) {
            int a2 = z->tup_a[t * L + r] / 2, base = z->abase[a2], m = z->amult[a2];
            for (int64_t k = 0; k < Lx; k++) {
                int64_t v = 0;
                for (int i = 0; i < m; i++) v += c[(r + 1) * Lx + md(k - z->dK[base + i], Lx)];
                c[r * Lx + k] = v;
            }
        }
    }
}
static inline int64_t rank_(const Z *z, int64_t t, const int *d) {
    int64_t L = z->L, Lx = z->Lx, c = 0, pre = 0; const int64_t *cn = z->cnt + t * (L + 1) * Lx;
    for (int64_t r = 0; r < L; r++) {
        int base = z->abase[z->tup_a[t * L + r] / 2];
        for (int g = base; g < d[r]; g++) c += cn[(r + 1) * Lx + md(z->Kx - pre - z->dK[g], Lx)];
        pre += z->dK[d[r]];
    }
    return c;
}
static inline int64_t label_of(const Z *z, int64_t q, const int *d) {
    int64_t t = z->ct[q];
    return z->toff[t] + (q - z->qfirst[t]) * z->nk[t] + rank_(z, t, d);
}
static inline void decode(const Z *z, int64_t lab, int64_t *q, int *d) {
    int64_t lo = 0, hi = z->nt, L = z->L, Lx = z->Lx;
    while (hi - lo > 1) { int64_t mid = (lo + hi) / 2; if (z->toff[mid] <= lab) lo = mid; else hi = mid; }
    int64_t t = lo, rem = lab - z->toff[t], comp = rem % z->nk[t], pre = 0; const int64_t *cn = z->cnt + t * (L + 1) * Lx;
    *q = z->qfirst[t] + rem / z->nk[t];
    for (int64_t r = 0; r < L; r++) {
        int a2 = z->tup_a[t * L + r] / 2, base = z->abase[a2], m = z->amult[a2], g = base;
        for (; g < base + m - 1; g++) { int64_t w = cn[(r + 1) * Lx + md(z->Kx - pre - z->dK[g], Lx)]; if (comp < w) break; comp -= w; }
        d[r] = g; pre += z->dK[g];
    }
}
static void chords(int L, const int8_t *a, const int8_t *c, int n[ML][ML]) {
    int stack[128], sp = 0;
    memset(n, 0, sizeof(int) * ML * ML);
    for (int x = 0; x < L; x++) {
        for (int k = 0; k < c[x]; k++) { int y = stack[--sp]; n[y][x]++; n[x][y]++; }
        for (int k = 0; k < a[x] - c[x]; k++) stack[sp++] = x;
    }
}
static int64_t find(const uint64_t *keys, int64_t n, uint64_t k) {
    int64_t lo = 0, hi = n - 1;
    while (lo <= hi) { int64_t mid = (lo + hi) / 2; if (keys[mid] == k) return mid; if (keys[mid] < k) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
int64_t rows_class_map(int64_t ncl, int L, const int8_t *ca, const int8_t *cc, const int32_t *inv, const uint64_t *ckeys,
                       const int32_t *cids, int32_t *out, double *sgn) {
    int64_t errors = 0;
    #pragma omp parallel for schedule(dynamic, 512) reduction(+:errors)
    for (int64_t q = 0; q < ncl; q++) {
        int n[ML][ML], n2[ML][ML], a2[ML], flips = 0; const int8_t *a = ca + q * L; uint64_t k = 0;
        chords(L, a, cc + q * L, n); memset(n2, 0, sizeof(n2));
        for (int x = 0; x < L; x++) {
            a2[inv[x]] = a[x];
            for (int y = x + 1; y < L; y++) if (n[x][y]) {
                n2[inv[x]][inv[y]] = n2[inv[y]][inv[x]] = n[x][y];
                if (inv[x] > inv[y]) flips += n[x][y];
            }
        }
        for (int y = 0; y < L; y++) { int c = 0; for (int x = 0; x < y; x++) c += n2[x][y]; k |= (uint64_t)((a2[y] / 2) | (c << 2)) << (5 * y); }
        int64_t zf = find(ckeys, ncl, k);
        if (zf < 0) { errors++; zf = 0; }
        out[q] = cids[zf]; sgn[q] = (flips & 1) ? -1.0 : 1.0;
    }
    return errors;
}
void rows_act(const Z *z, int64_t nlab, const int64_t *lab, const int32_t *axes, const int32_t *cmap, const double *csign, int reflect,
              int64_t *out, double *ph) {
    int64_t L = z->L;
    #pragma omp parallel for schedule(dynamic, 4096)
    for (int64_t i = 0; i < nlab; i++) {
        int d[ML], d2[ML]; int64_t q; double re = 1.0, im = 0.0;
        decode(z, lab[i], &q, d);
        if (reflect) for (int x = 0; x < L; x++) {
            double a = z->dPh[2 * d[x]], b = z->dPh[2 * d[x] + 1], t = re * a - im * b; im = re * b + im * a; re = t; d[x] = z->dBar[d[x]];
        }
        for (int x = 0; x < L; x++) d2[x] = d[axes[x]];
        out[i] = label_of(z, cmap[q], d2); ph[2 * i] = re * csign[q]; ph[2 * i + 1] = im * csign[q];
    }
}
void rows_psi(const Z *z, int64_t nlab, const int64_t *lab, double *out) {
    #pragma omp parallel for schedule(dynamic, 4096)
    for (int64_t i = 0; i < nlab; i++) {
        int d[ML]; int64_t q; double re = 1.0, im = 0.0;
        decode(z, lab[i], &q, d);
        for (int x = 0; x < z->L; x++) { double a = z->dPsi[2 * d[x]], b = z->dPsi[2 * d[x] + 1], t = re * a - im * b; im = re * b + im * a; re = t; }
        out[2 * i] = re; out[2 * i + 1] = im;
    }
}
void rows_orbit_min(int64_t dim, int nb, const int64_t *base, const int64_t *rot, int L, int64_t *rep) {
    #pragma omp parallel for schedule(static)
    for (int64_t l = 0; l < dim; l++) {
        int64_t m = l;
        for (int b = 0; b < nb; b++) {
            int64_t cur = base[(int64_t)b * dim + l];
            for (int n = 0; n < L; n++) { if (cur < m) m = cur; cur = rot[cur]; }
        }
        rep[l] = m;
    }
}
/* raw columns -> entries of the real sector matrix: map targets to representatives, multiply by the orbit factor and the
   label phases, add the diagonal, sort every column by row, merge equal rows, drop |v| < 1e-10.  Returns the entry count;
   the columns are packed in place into out_row / out_val, count[c] entries each; *imag = largest imaginary part seen. */
int64_t rows_fold(int64_t ncol, int64_t c0, const int64_t *colptr, const int64_t *rows, const double *vals, const double *diag,
                  const int64_t *rep, const int64_t *newidx, const double *fac, const double *Dg, int32_t *out_row, double *out_val,
                  int64_t *count, double *imag) {
    double worst = 0.0;
    #pragma omp parallel for schedule(dynamic, 256) reduction(max:worst)
    for (int64_t c = 0; c < ncol; c++) {
        int64_t o = colptr[c] + c, n = 0; double dr = Dg[2 * (c0 + c)], di = Dg[2 * (c0 + c) + 1];
        int32_t *R = out_row + o; double *V = out_val + o;
        for (int64_t e = colptr[c]; e < colptr[c + 1]; e++) {
            int64_t lab = rows[e], ri = newidx[rep[lab]];
            if (ri < 0) continue;
            double ar = vals[2 * e] * fac[2 * lab] - vals[2 * e + 1] * fac[2 * lab + 1], ai = vals[2 * e] * fac[2 * lab + 1] + vals[2 * e + 1] * fac[2 * lab];
            double br = ar * dr - ai * di, bi = ar * di + ai * dr, qr = Dg[2 * ri], qi = Dg[2 * ri + 1], qq = qr * qr + qi * qi;
            double vr = (br * qr + bi * qi) / qq, vi = (bi * qr - br * qi) / qq;
            if (vi > worst) worst = vi; if (-vi > worst) worst = -vi;
            R[n] = (int32_t)ri; V[n] = vr; n++;
        }
        R[n] = (int32_t)(c0 + c); V[n] = diag[c]; n++;
        for (int64_t gap = n / 2; gap > 0; gap = gap == 2 ? 1 : (int64_t)(gap / 2.2)) {
            for (int64_t i = gap; i < n; i++) {
                int32_t r = R[i]; double v = V[i]; int64_t j = i;
                for (; j >= gap && R[j - gap] > r; j -= gap) { R[j] = R[j - gap]; V[j] = V[j - gap]; }
                R[j] = r; V[j] = v;
            }
        }
        int64_t m = 0;
        for (int64_t i = 0; i < n; ) {
            int32_t r = R[i]; double v = 0.0;
            for (; i < n && R[i] == r; i++) v += V[i];
            if (v > 1e-10 || v < -1e-10) { R[m] = r; V[m] = v; m++; }
        }
        count[c] = m;
    }
    int64_t tot = 0;
    for (int64_t c = 0; c < ncol; c++) {
        int64_t o = colptr[c] + c;
        if (o != tot) { memmove(out_row + tot, out_row + o, count[c] * sizeof(int32_t)); memmove(out_val + tot, out_val + o, count[c] * sizeof(double)); }
        tot += count[c];
    }
    *imag = worst;
    return tot;
}
#include <omp.h>
/* scatter the packed columns (rows sorted inside each column) into CSR; every thread owns a range of rows */
void rows_scatter(int64_t ncol, int64_t c0, const int64_t *cptr, const int32_t *row, const double *val, int64_t nrows, int64_t *cursor,
                  int32_t *ci, double *va) {
    #pragma omp parallel
    {
        int64_t T = omp_get_num_threads(), t = omp_get_thread_num(), r0 = nrows * t / T, r1 = nrows * (t + 1) / T;
        for (int64_t c = 0; c < ncol; c++) {
            int64_t lo = cptr[c], hi = cptr[c + 1], a = lo, b = hi;
            while (a < b) { int64_t m = (a + b) / 2; if (row[m] < r0) a = m + 1; else b = m; }
            for (int64_t e = a; e < hi && row[e] < r1; e++) { int64_t o = cursor[row[e]]++; ci[o] = (int32_t)(c0 + c); va[o] = val[e]; }
        }
    }
}
/* raw columns of H: for every input label the list of (target label, value); count pass (colptr == 0) or fill pass */
void rows_columns(const Z *z, int64_t nlab, const int64_t *lab, const int64_t *wptr, const int32_t *wcls, const double *wcoef,
                  const int64_t *gptr, const int32_t *gd1, const int32_t *gd2, const double *gval, const int64_t *hptr, const int32_t *hd,
                  const double *hval, const int64_t *colptr, int64_t *count, int64_t *rows, double *vals, double *diag) {
    int64_t L = z->L, M = z->M;
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t i = 0; i < nlab; i++) {
        int d[ML]; int64_t q, n = 0, o = colptr ? colptr[i] : 0; double dg = 0.0;
        decode(z, lab[i], &q, d);
        for (int x = 0; x < L; x++) {
            dg += z->dE[d[x]];
            for (int64_t e = hptr[d[x]]; e < hptr[d[x] + 1]; e++) {
                if (colptr) { int keep = d[x]; d[x] = hd[e]; rows[o + n] = label_of(z, q, d); d[x] = keep; vals[2 * (o + n)] = hval[2 * e]; vals[2 * (o + n) + 1] = hval[2 * e + 1]; }
                n++;
            }
        }
        for (int r = 0; r < L; r++) {
            if (L == 2 && r == 1) continue;
            int s = (r + 1) % L, k1 = d[r], k2 = d[s];
            for (int64_t e = gptr[(int64_t)k1 * M + k2]; e < gptr[(int64_t)k1 * M + k2 + 1]; e++) {
                int64_t slot = (q * L + r) * 16 + (z->dA[gd1[e]] / 2) * 4 + z->dA[gd2[e]] / 2;
                if (!colptr) { n += wptr[slot + 1] - wptr[slot]; continue; }
                d[r] = gd1[e]; d[s] = gd2[e];
                for (int64_t w = wptr[slot]; w < wptr[slot + 1]; w++) {
                    rows[o + n] = label_of(z, wcls[w], d);
                    vals[2 * (o + n)] = gval[2 * e] * wcoef[2 * w] - gval[2 * e + 1] * wcoef[2 * w + 1];
                    vals[2 * (o + n) + 1] = gval[2 * e] * wcoef[2 * w + 1] + gval[2 * e + 1] * wcoef[2 * w];
                    n++;
                }
            }
            d[r] = k1; d[s] = k2;
        }
        if (colptr) diag[i] = dg; else count[i] = n;
    }
}
"""


class _Z(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in ('L', 'Lx', 'Kx', 'M', 'nt')]+[(n, ctypes.c_void_p) for n in (
        'dA', 'dK', 'dBar', 'abase', 'amult', 'dE', 'dPh', 'dPsi', 'tup_a', 'cnt', 'nk', 'toff', 'qfirst', 'ct')]


def sector_routines(Lx, Ly, Kx, spins, mult, mk, mE, ibar, iphase, rowpsi, classes, by_tuple, GAMP, HOFF, WT, coff):
    """C versions of class_map, act, columns and label_psi of build_sector (same arguments and results)."""
    lib = _compile_c(SECTOR_C); P = ctypes.c_void_p; I64 = ctypes.c_int64; L = Ly; ncl = len(classes); ad = lambda a: a.ctypes.data
    c8 = lambda v, t: np.ascontiguousarray(v, t)
    abase = np.zeros(4, np.int32); amult = np.zeros(4, np.int32); dA, dK, dE, dBar, dPh, dPsi = [], [], [], [], [], []
    for a in spins:
        abase[a//2] = len(dA); amult[a//2] = mult[a]
        for i in range(mult[a]):
            dA.append(a); dK.append(mk[a][i]); dE.append(mE[a][i]); dBar.append(abase[a//2]+int(ibar[a][i])); dPh.append(iphase[a][i]); dPsi.append(rowpsi[a][i])
    M = len(dA); dA = c8(dA, np.int8); dK = c8(dK, np.int32); dE = c8(dE, np.float64); dBar = c8(dBar, np.int32)
    dPh = c8(dPh, np.complex128); dPsi = c8(dPsi, np.complex128)
    tuples = list(by_tuple); nt = len(tuples); tup_a = c8(tuples, np.int8).reshape(nt, L)
    qfirst = np.zeros(nt, np.int32); ncls = np.zeros(nt, np.int64); ct = np.zeros(ncl, np.int32)
    for t, avec in enumerate(tuples):
        ids = by_tuple[avec]; qfirst[t] = ids[0]; ncls[t] = len(ids); ct[ids[0]:ids[0]+len(ids)] = t
        assert ids[-1]-ids[0]+1 == len(ids)
    cnt = np.zeros((nt, L+1, Lx), np.int64); nk = np.zeros(nt, np.int64); toff = np.zeros(nt+1, np.int64)
    z = _Z(L, Lx, Kx, M, nt, ad(dA), ad(dK), ad(dBar), ad(abase), ad(amult), ad(dE), ad(dPh), ad(dPsi), ad(tup_a), ad(cnt), ad(nk), ad(toff),
           ad(qfirst), ad(ct))
    lib.rows_cnt.argtypes = [P, P]; lib.rows_cnt(ctypes.byref(z), ad(cnt))
    nk[:] = cnt[:, 0, Kx]; np.cumsum(ncls*nk, out=toff[1:])
    assert np.array_equal(toff[:-1][ncls*nk > 0], np.asarray(coff)[qfirst][ncls*nk > 0]) and toff[-1] == coff[-1]
    # transitions of two neighbouring rows by digit pair, and the off-diagonal row Hamiltonian by digit
    lists = [[] for _ in range(M*M)]
    for (a, b, a2, b2), g in GAMP.items():
        for ip, jp, i, j in zip(*np.nonzero(g)):
            lists[(abase[a//2]+i)*M+abase[b//2]+j].append((abase[a2//2]+ip, abase[b2//2]+jp, g[ip, jp, i, j]))
    gptr = np.zeros(M*M+1, np.int64); gptr[1:] = np.cumsum([len(v) for v in lists]); flat = [e for v in lists for e in v]
    gd1 = c8([e[0] for e in flat], np.int32); gd2 = c8([e[1] for e in flat], np.int32); gval = c8([e[2] for e in flat], np.complex128)
    hl = [[] for _ in range(M)]
    for a in spins:
        for i2, i1, val in HOFF[a]:
            hl[abase[a//2]+i1].append((abase[a//2]+i2, val))
    hptr = np.zeros(M+1, np.int64); hptr[1:] = np.cumsum([len(v) for v in hl]); hflat = [e for v in hl for e in v]
    hd = c8([e[0] for e in hflat] or [0], np.int32); hval = c8([e[1] for e in hflat] or [0], np.complex128)
    wptr = c8(WT.ptr, np.int64); wcls = c8(WT.cls if WT.total else [0], np.int32); wcoef = c8(WT.coef if WT.total else [0], np.complex128)
    lib.rows_class_map.argtypes = [I64, ctypes.c_int, P, P, P, P, P, P, P]; lib.rows_class_map.restype = I64
    lib.rows_act.argtypes = [P, I64, P, P, P, P, ctypes.c_int, P, P]; lib.rows_psi.argtypes = [P, I64, P, P]
    lib.rows_columns.argtypes = [P, I64]+[P]*16
    keepalive = (dA, dK, dE, dBar, dPh, dPsi, abase, amult, tup_a, qfirst, ct, cnt, nk, toff, gptr, gd1, gd2, gval, hptr, hd, hval, wptr, wcls, wcoef)

    def class_map(axes):
        inv = np.zeros(L, np.int32); inv[list(axes)] = np.arange(L, dtype=np.int32)
        out = np.zeros(ncl, np.int32); sgn = np.ones(ncl)
        err = lib.rows_class_map(ncl, L, ad(WT.ca), ad(WT.cc), ad(inv), ad(WT.ckeys), ad(WT.cids), ad(out), ad(sgn))
        assert err == 0
        return out.astype(np.int64), sgn

    def act(lab, axes, cmap, reflect):
        lab = c8(lab, np.int64); ax = c8(axes, np.int32); cm = c8(cmap[0], np.int32); cs = c8(cmap[1], np.float64)
        out = np.zeros(len(lab), np.int64); ph = np.zeros(len(lab), np.complex128)
        lib.rows_act(ctypes.byref(z), len(lab), ad(lab), ad(ax), ad(cm), ad(cs), int(bool(reflect)), ad(out), ad(ph))
        return out, ph

    def label_psi(labs):
        labs = c8(labs, np.int64); out = np.zeros(len(labs), np.complex128)
        lib.rows_psi(ctypes.byref(z), len(labs), ad(labs), ad(out))
        return out

    def columns(labs):
        labs = c8(labs, np.int64); n = len(labs); count = np.zeros(n, np.int64); diag = np.zeros(n)
        args = (ctypes.byref(z), n, ad(labs), ad(wptr), ad(wcls), ad(wcoef), ad(gptr), ad(gd1), ad(gd2), ad(gval), ad(hptr), ad(hd), ad(hval))
        lib.rows_columns(*args, None, ad(count), None, None, None)
        ptr = np.zeros(n+1, np.int64); np.cumsum(count, out=ptr[1:]); tot = max(1, int(ptr[-1]))
        rows = np.zeros(tot, np.int64); vals = np.zeros(tot, np.complex128)
        lib.rows_columns(*args, ad(ptr), None, ad(rows), ad(vals), ad(diag))
        tot = int(ptr[-1])
        return rows[:tot], np.repeat(np.arange(n), count), vals[:tot], diag

    lib.rows_orbit_min.argtypes = [I64, ctypes.c_int, P, P, ctypes.c_int, P]
    lib.rows_fold.argtypes = [I64, I64]+[P]*12; lib.rows_fold.restype = I64
    lib.rows_scatter.argtypes = [I64, I64, P, P, P, I64, P, P, P]

    def orbit_min(base_perms, rot_perm):
        base_perms = c8(base_perms, np.int64); rot_perm = c8(rot_perm, np.int64); rep = np.zeros(base_perms.shape[1], np.int64)
        lib.rows_orbit_min(base_perms.shape[1], base_perms.shape[0], ad(base_perms), ad(rot_perm), L, ad(rep))
        return rep

    def assemble_real(reps, rep, newidx, fac, Dg, chunk):
        """Real sector matrix in CSR straight from the raw columns: (row pointers, columns, values, largest imaginary part)."""
        reps = c8(reps, np.int64); rep = c8(rep, np.int64); newidx = c8(newidx, np.int64); fac = c8(fac, np.complex128); Dg = c8(Dg, np.complex128)
        n = len(reps); pieces = []; worst = 0.0; rowcount = np.zeros(n, np.int64)
        for c0 in range(0, n, chunk):
            labs = c8(reps[c0:c0+chunk], np.int64); m = len(labs); count = np.zeros(m, np.int64); diag = np.zeros(m)
            args = (ctypes.byref(z), m, ad(labs), ad(wptr), ad(wcls), ad(wcoef), ad(gptr), ad(gd1), ad(gd2), ad(gval), ad(hptr), ad(hd), ad(hval))
            lib.rows_columns(*args, None, ad(count), None, None, None)
            ptr = np.zeros(m+1, np.int64); np.cumsum(count, out=ptr[1:]); tot = max(1, int(ptr[-1]))
            rows = np.zeros(tot, np.int64); vals = np.zeros(tot, np.complex128)
            lib.rows_columns(*args, ad(ptr), None, ad(rows), ad(vals), ad(diag))
            orow = np.zeros(tot+m, np.int32); oval = np.zeros(tot+m); im = ctypes.c_double(0.0)
            k = lib.rows_fold(m, c0, ad(ptr), ad(rows), ad(vals), ad(diag), ad(rep), ad(newidx), ad(fac), ad(Dg), ad(orow), ad(oval), ad(count),
                              ctypes.byref(im))
            worst = max(worst, im.value); orow = orow[:k].copy(); oval = oval[:k].copy()
            rowcount += np.bincount(orow, minlength=n); pieces.append((c0, count, orow, oval)); del rows, vals
        rp = np.zeros(n+1, np.int64); np.cumsum(rowcount, out=rp[1:]); cursor = rp[:-1].copy()
        ci = np.zeros(int(rp[-1]), np.int32); va = np.zeros(int(rp[-1]))
        for c0, count, orow, oval in pieces:
            cptr = np.zeros(len(count)+1, np.int64); np.cumsum(count, out=cptr[1:])
            lib.rows_scatter(len(count), c0, ad(cptr), ad(orow), ad(oval), n, ad(cursor), ad(ci), ad(va))
        return rp, ci, va, worst

    columns.keepalive = keepalive
    tabs = dict(scal=(L, Lx, Kx, M, nt), dA=dA, dK=dK, dBar=dBar, abase=abase, amult=amult, dE=dE, dPh=dPh, dPsi=dPsi, tup_a=tup_a, cnt=cnt, nk=nk,
                toff=toff, qfirst=qfirst, ct=ct, wptr=wptr, wcls=wcls, wcoef=wcoef, gptr=gptr, gd1=gd1, gd2=gd2, gval=gval, hptr=hptr, hd=hd, hval=hval)
    return class_map, act, columns, label_psi, orbit_min, assemble_real, tabs


# ---------------------------------------------------------------------------------------------------------------------
# the same sector routines as CUDA kernels: images of all labels, orbit minima, raw columns, folding onto representatives
SECTOR_CUDA = r"""
typedef long long i64; typedef int i32; typedef signed char i8; typedef unsigned long long u64;
struct Z { i64 L, Lx, Kx, M, nt; const i8 *dA; const i32 *dK, *dBar, *abase, *amult; const double *dE, *dPh, *dPsi; const i8 *tup_a;
           const i64 *cnt, *nk, *toff; const i32 *qfirst, *ct; };
#define ZARGS const i64 zL, const i64 zLx, const i64 zKx, const i64 zM, const i64 znt, const i8 *dA, const i32 *dK, const i32 *dBar, \
    const i32 *abase, const i32 *amult, const double *dE, const double *dPh, const double *dPsi, const i8 *tup_a, const i64 *cnt, \
    const i64 *nk, const i64 *toff, const i32 *qfirst, const i32 *ct
#define ZINIT Z z_; z_.L = zL; z_.Lx = zLx; z_.Kx = zKx; z_.M = zM; z_.nt = znt; z_.dA = dA; z_.dK = dK; z_.dBar = dBar; z_.abase = abase; \
    z_.amult = amult; z_.dE = dE; z_.dPh = dPh; z_.dPsi = dPsi; z_.tup_a = tup_a; z_.cnt = cnt; z_.nk = nk; z_.toff = toff; \
    z_.qfirst = qfirst; z_.ct = ct; const Z *z = &z_;
__device__ inline int md(i64 x, i64 m) { return (int)(((x % m) + m) % m); }
__device__ inline i64 rank_(const Z *z, i64 t, const int *d) {
    i64 L = z->L, Lx = z->Lx, c = 0, pre = 0; const i64 *cn = z->cnt + t * (L + 1) * Lx;
    for (i64 r = 0; r < L; r++) {
        int base = z->abase[z->tup_a[t * L + r] / 2];
        for (int g = base; g < d[r]; g++) c += cn[(r + 1) * Lx + md(z->Kx - pre - z->dK[g], Lx)];
        pre += z->dK[d[r]];
    }
    return c;
}
__device__ inline i64 label_of(const Z *z, i64 q, const int *d) {
    i64 t = z->ct[q];
    return z->toff[t] + (q - z->qfirst[t]) * z->nk[t] + rank_(z, t, d);
}
__device__ inline void decode(const Z *z, i64 lab, i64 *q, int *d) {
    i64 lo = 0, hi = z->nt, L = z->L, Lx = z->Lx;
    while (hi - lo > 1) { i64 mid = (lo + hi) / 2; if (z->toff[mid] <= lab) lo = mid; else hi = mid; }
    i64 t = lo, rem = lab - z->toff[t], comp = rem % z->nk[t], pre = 0; const i64 *cn = z->cnt + t * (L + 1) * Lx;
    *q = z->qfirst[t] + rem / z->nk[t];
    for (i64 r = 0; r < L; r++) {
        int a2 = z->tup_a[t * L + r] / 2, base = z->abase[a2], m = z->amult[a2], g = base;
        for (; g < base + m - 1; g++) { i64 w = cn[(r + 1) * Lx + md(z->Kx - pre - z->dK[g], Lx)]; if (comp < w) break; comp -= w; }
        d[r] = g; pre += z->dK[g];
    }
}
extern "C" __global__ void k_act(ZARGS, const i64 nlab, const i64 *lab, const i32 *axes, const i32 *cmap, const double *csign,
                                 const int reflect, i64 *out, double *ph) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= nlab) return;
    ZINIT
    int d[12], d2[12]; i64 q; double re = 1.0, im = 0.0;
    decode(z, lab[i], &q, d);
    if (reflect) for (int x = 0; x < zL; x++) {
        double a = dPh[2 * d[x]], b = dPh[2 * d[x] + 1], t = re * a - im * b; im = re * b + im * a; re = t; d[x] = dBar[d[x]];
    }
    for (int x = 0; x < zL; x++) d2[x] = d[axes[x]];
    out[i] = label_of(z, cmap[q], d2); ph[2 * i] = re * csign[q]; ph[2 * i + 1] = im * csign[q];
}
extern "C" __global__ void k_psi(ZARGS, const i64 nlab, const i64 *lab, double *out) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= nlab) return;
    ZINIT
    int d[12]; i64 q; double re = 1.0, im = 0.0;
    decode(z, lab[i], &q, d);
    for (int x = 0; x < zL; x++) { double a = dPsi[2 * d[x]], b = dPsi[2 * d[x] + 1], t = re * a - im * b; im = re * b + im * a; re = t; }
    out[2 * i] = re; out[2 * i + 1] = im;
}
extern "C" __global__ void k_orbit_min(const i64 dim, const int nb, const i64 *base, const i64 *rot, const int L, i64 *rep) {
    i64 l = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (l >= dim) return;
    i64 m = l;
    for (int b = 0; b < nb; b++) { i64 cur = base[(i64)b * dim + l]; for (int n = 0; n < L; n++) { if (cur < m) m = cur; cur = rot[cur]; } }
    rep[l] = m;
}
extern "C" __global__ void k_columns(ZARGS, const i64 nlab, const i64 *lab, const i64 *wptr, const i32 *wcls, const double *wcoef,
                                     const i64 *gptr, const i32 *gd1, const i32 *gd2, const double *gval, const i64 *hptr, const i32 *hd,
                                     const double *hval, const i64 *colptr, const int fill, i64 *count, i64 *rows, double *vals, double *diag) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (i >= nlab) return;
    ZINIT
    int d[12]; i64 q, n = 0, o = fill ? colptr[i] : 0, L = zL, M = zM; double dg = 0.0;
    decode(z, lab[i], &q, d);
    for (int x = 0; x < L; x++) {
        dg += dE[d[x]];
        for (i64 e = hptr[d[x]]; e < hptr[d[x] + 1]; e++) {
            if (fill) { int keep = d[x]; d[x] = hd[e]; rows[o + n] = label_of(z, q, d); d[x] = keep; vals[2 * (o + n)] = hval[2 * e]; vals[2 * (o + n) + 1] = hval[2 * e + 1]; }
            n++;
        }
    }
    for (int r = 0; r < L; r++) {
        if (L == 2 && r == 1) continue;
        int s = (r + 1) % L, k1 = d[r], k2 = d[s];
        for (i64 e = gptr[(i64)k1 * M + k2]; e < gptr[(i64)k1 * M + k2 + 1]; e++) {
            i64 slot = (q * L + r) * 16 + (dA[gd1[e]] / 2) * 4 + dA[gd2[e]] / 2;
            if (!fill) { n += wptr[slot + 1] - wptr[slot]; continue; }
            d[r] = gd1[e]; d[s] = gd2[e];
            for (i64 w = wptr[slot]; w < wptr[slot + 1]; w++) {
                rows[o + n] = label_of(z, wcls[w], d);
                vals[2 * (o + n)] = gval[2 * e] * wcoef[2 * w] - gval[2 * e + 1] * wcoef[2 * w + 1];
                vals[2 * (o + n) + 1] = gval[2 * e] * wcoef[2 * w + 1] + gval[2 * e + 1] * wcoef[2 * w];
                n++;
            }
        }
        d[r] = k1; d[s] = k2;
    }
    if (fill) diag[i] = dg; else count[i] = n;
}
extern "C" __global__ void k_fold(const i64 ntot, const i64 *rows, const double *vals, const i32 *col, const i64 *rep, const i64 *newidx,
                                  const double *fac, const double *Dg, u64 *key, double *v, double *vim) {
    i64 e = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (e >= ntot) return;
    i64 lab = rows[e], ri = newidx[rep[lab]];
    if (ri < 0) { key[e] = 0xFFFFFFFFFFFFFFFFull; v[e] = 0.0; vim[e] = 0.0; return; }
    double dr = Dg[2 * (i64)col[e]], di = Dg[2 * (i64)col[e] + 1];
    double ar = vals[2 * e] * fac[2 * lab] - vals[2 * e + 1] * fac[2 * lab + 1], ai = vals[2 * e] * fac[2 * lab + 1] + vals[2 * e + 1] * fac[2 * lab];
    double br = ar * dr - ai * di, bi = ar * di + ai * dr, qr = Dg[2 * ri], qi = Dg[2 * ri + 1], qq = qr * qr + qi * qi;
    key[e] = ((u64)ri << 32) | (u64)(unsigned int)col[e]; v[e] = (br * qr + bi * qi) / qq; vim[e] = (bi * qr - br * qi) / qq;
}
"""


def gpu_sector(tabs, cm_rot, cm_mir, cm_id, dim, Ly, ky, py_, px_, chunk, log):
    """Sector matrix on the GPU for both mirrors and a real character: representatives, raw columns, folding, CSR, value table."""
    import cupy as cp, cupyx, cmath
    t0 = time.time(); mod = cp.RawModule(code=SECTOR_CUDA)
    kact, kpsi, kmin, kcol, kfold = [mod.get_function(n) for n in ('k_act', 'k_psi', 'k_orbit_min', 'k_columns', 'k_fold')]
    dv = {k: cp.asarray(v) for k, v in tabs.items() if k != 'scal'}; L, Lx, Kx, M, nt = tabs['scal']
    zargs = (np.int64(L), np.int64(Lx), np.int64(Kx), np.int64(M), np.int64(nt), dv['dA'], dv['dK'], dv['dBar'], dv['abase'], dv['amult'], dv['dE'],
             dv['dPh'], dv['dPsi'], dv['tup_a'], dv['cnt'], dv['nk'], dv['toff'], dv['qfirst'], dv['ct'])
    grid = lambda n: (((int(n)+255)//256,), (256,))

    def act(lab, axes, cmap, reflect):
        out = cp.zeros(len(lab), cp.int64); ph = cp.zeros(len(lab), cp.complex128)
        kact(*grid(len(lab)), zargs+(np.int64(len(lab)), lab, cp.asarray(np.asarray(axes, np.int32)), cp.asarray(np.asarray(cmap[0], np.int32)),
                                    cp.asarray(np.asarray(cmap[1], np.float64)), np.int32(1 if reflect else 0), out, ph))
        return out, ph
    labs = cp.arange(dim, dtype=cp.int64)
    ident = list(range(Ly)); rot = [(x-1) % Ly for x in range(Ly)]; mir = [(-x) % Ly for x in range(Ly)]
    gx = act(labs, ident, cm_id, True); gy = act(labs, mir, cm_mir, False); rotg = act(labs, rot, cm_rot, False)
    gyx = (gy[0][gx[0]], gy[1][gx[0]]*gx[1])
    base = [(labs, None, 1.0), (gx[0], gx[1], px_), (gy[0], gy[1], py_), (gyx[0], gyx[1], py_*px_)]
    bp = cp.stack([b[0] for b in base]); rep = cp.zeros(dim, cp.int64)
    kmin(*grid(dim), (np.int64(dim), np.int32(len(base)), bp, rotg[0], np.int32(Ly), rep)); del bp
    reps = cp.flatnonzero(rep == labs); fac = cp.zeros(dim, cp.complex128); srep = cp.zeros(len(reps), cp.complex128)
    for perm, ph, ch in base:
        cp_ = perm[reps]; cph = ph[reps] if ph is not None else cp.ones(len(reps), cp.complex128)
        for nrot in range(Ly):
            chi = ch*cmath.exp(2j*math.pi*ky*nrot/Ly)
            fac[cp_] = chi/cph
            stab = cp_ == reps; srep += cp.where(stab, np.conj(chi)*cph, 0)
            cph = rotg[1][cp_]*cph; cp_ = rotg[0][cp_]
    del base, gx, gy, gyx, rotg, labs
    reps = reps[cp.abs(srep) > 1e-9]; nd = int(len(reps))
    newidx = cp.full(dim, -1, cp.int64); newidx[reps] = cp.arange(nd, dtype=cp.int64)
    Dg = cp.zeros(nd, cp.complex128); kpsi(*grid(nd), zargs+(np.int64(nd), reps, Dg))
    log(f'GPU: {nd} representatives of {dim} labels, {time.time()-t0:.1f} s')
    tb = [dv[k] for k in ('wptr', 'wcls', 'wcoef', 'gptr', 'gd1', 'gd2', 'gval', 'hptr', 'hd', 'hval')]
    pk, pv = [], []; worst = 0.0; dummy = cp.zeros(2, cp.int64); dummyd = cp.zeros(4); t1 = time.time()
    for c0 in range(0, nd, chunk):
        lab = reps[c0:c0+chunk]; m = int(len(lab)); count = cp.zeros(m, cp.int64); diag = cp.zeros(m)
        kcol(*grid(m), zargs+(np.int64(m), lab, *tb, dummy, np.int32(0), count, dummy, dummyd, diag))
        ptr = cp.zeros(m+1, cp.int64); ptr[1:] = cp.cumsum(count); tot = int(ptr[-1])
        rows = cp.zeros(max(tot, 1), cp.int64); vals = cp.zeros(max(tot, 1), cp.complex128)
        kcol(*grid(m), zargs+(np.int64(m), lab, *tb, ptr, np.int32(1), count, rows, vals, diag))
        col = (cp.searchsorted(ptr[1:], cp.arange(tot, dtype=cp.int64), side='right')+c0).astype(cp.int32)
        key = cp.zeros(tot+m, cp.uint64); v = cp.zeros(tot+m); vim = cp.zeros(max(tot, 1))
        kfold(*grid(tot), (np.int64(tot), rows, vals, col, rep, newidx, fac, Dg, key, v, vim))
        worst = max(worst, float(cp.abs(vim).max())); del rows, vals, col, vim
        cidx = cp.arange(c0, c0+m, dtype=cp.uint64); key[tot:] = (cidx << np.uint64(32)) | cidx; v[tot:] = diag
        order = cp.argsort(key); key = key[order]; v = v[order]; del order
        flag = cp.ones(len(key), cp.bool_); flag[1:] = key[1:] != key[:-1]
        seg = cp.cumsum(flag)-1; sums = cp.zeros(int(seg[-1])+1); cupyx.scatter_add(sums, seg, v)
        uk = key[flag]; km = (uk != np.uint64(0xFFFFFFFFFFFFFFFF)) & (cp.abs(sums) > 1e-10)
        pk.append(uk[km]); pv.append(sums[km]); del key, v, flag, seg, sums, uk, km
    K = cp.concatenate(pk); V = cp.concatenate(pv); del pk, pv, rep, newidx, fac
    order = cp.argsort(K); K = K[order]; V = V[order]; del order
    rowi = (K >> np.uint64(32)).astype(cp.int64); ci = (K & np.uint64(0xFFFFFFFF)).astype(cp.int32); del K
    rp = cp.zeros(nd+1, cp.int64); rp[1:] = cp.cumsum(cp.bincount(rowi, minlength=nd)); del rowi
    log(f'GPU: columns, folding and CSR: {len(ci)} non-zeros, {time.time()-t1:.1f} s')
    vt, code = cp.unique(cp.rint(V*1e12).astype(cp.int64), return_inverse=True)
    if len(vt) <= 65535:
        vals = (code.astype(cp.uint16), vt.astype(cp.float64)*1e-12); compact = True; del V
    else:
        vals = (V,); compact = False
    return dict(rp=rp, ci=ci, vals=vals, compact=compact, dim=nd, nnz=int(len(ci)), imag=worst, group=4*Ly, seconds=time.time()-t0)


# ---------------------------------------------------------------------------------------------------------------------
# generated product kernels (one source, two targets) and the engine
def kernel_source(real, target, compact=False, lanes=1):
    """C/OpenMP or CUDA source of  y = C x  for a CSR matrix with 8-byte row pointers and 4-byte column indices.
    compact (real matrices): 2-byte code into a table of the distinct values instead of an 8-byte value per entry.
    lanes > 1 (CUDA, real): that many threads share one row and add their partial sums by shuffles."""
    T = ('double' if real else ('complex<double>' if target == 'cuda' else 'double _Complex'))
    vals = 'const unsigned short *cd, const double *vt' if compact else f'const {T} *va'
    term = 'vt[cd[e]]' if compact else 'va[e]'
    if target == 'cuda' and lanes > 1:
        assert real
        return f'''
#define W {lanes}
extern "C" __global__ void rows_spmv(const long long n, const long long *rp, const int *ci, {vals}, const double *x, double *y) {{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, i = tid / W;
    int lane = (int)(tid % W);
    double s = 0.0;
    if (i < n) for (long long e = rp[i] + lane; e < rp[i + 1]; e += W) s += {term} * x[ci[e]];
    for (int off = W / 2; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off, W);
    if (lane == 0 && i < n) y[i] = s;
}}
'''
    if target == 'cuda':
        return ('#include <cupy/complex.cuh>\n' if not real else '')+f'''
typedef {T} T;
extern "C" __global__ void rows_spmv(const long long n, const long long *rp, const int *ci, {vals}, const T *x, T *y) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    T s = 0;
    for (long long e = rp[i]; e < rp[i + 1]; ++e) s += {term} * x[ci[e]];
    y[i] = s;
}}
'''
    return f'''
#include <stdint.h>
#include <complex.h>
typedef {T} T;
void rows_spmv(int64_t n, const int64_t *rp, const int32_t *ci, {vals.replace('unsigned short', 'uint16_t')}, const T *x, T *y) {{
    #pragma omp parallel for schedule(dynamic, 2048)
    for (int64_t i = 0; i < n; i++) {{
        T s = 0;
        for (int64_t e = rp[i]; e < rp[i + 1]; e++) s += {term} * x[ci[e]];
        y[i] = s;
    }}
}}
'''


def _compile_c(src):
    import edc
    cache = os.environ.get('EDC_CACHE', os.path.expanduser('~/.cache/edc')); os.makedirs(cache, exist_ok=True)
    name = 'rows_'+hashlib.sha1((src+' '.join(edc.CC)).encode()).hexdigest()[:16]
    so = os.path.join(cache, name+('.dylib' if sys.platform == 'darwin' else '.so'))
    if not os.path.exists(so):
        c = os.path.join(cache, name+'.c'); open(c, 'w').write(src)
        subprocess.check_call(edc.CC+[c, '-o', so])
    return ctypes.CDLL(so)


def arnoldi_lowest(matvec, n, xp, dtype, nev=1, tol=1e-12, m=60, maxrestart=200, seed=1, stable=3):
    """Lowest eigenvalues (smallest real part) of a matrix given by matvec: Arnoldi with full re-orthogonalisation and
    explicit restart.  Stopping rule = the one of the S_z engine's Lanczos: the lowest nev Ritz values change by less
    than tol * max(1, |E|) for `stable` consecutive steps.  Returns (values, number of products, last change)."""
    rng = np.random.default_rng(seed)
    v = xp.asarray(rng.standard_normal(n)).astype(dtype); v /= xp.linalg.norm(v)
    m = min(m, n-1); products = 0; V = xp.zeros((m+1, n), dtype); e_old = None; hits = 0; change = np.inf
    host = lambda a: np.asarray(a.get() if hasattr(a, 'get') else a)
    for restart in range(maxrestart):
        Hd = xp.zeros((m+1, m), dtype); V[0] = v; k = m
        for j in range(m):
            w = matvec(V[j]); products += 1
            for _ in range(2):
                h = V[:j+1].conj() @ w; w = w-h @ V[:j+1]; Hd[:j+1, j] += h
            beta = xp.linalg.norm(w); Hd[j+1, j] = beta; V[j+1] = w/beta
            if j+1 >= max(nev, 2):
                Hs = host(Hd[:j+2, :j+1])
                if not np.all(np.isfinite(Hs)):                    # exact breakdown: the Krylov space is invariant
                    k = j+1; Hs = np.nan_to_num(Hs); e = np.sort(np.linalg.eigvals(Hs[:k, :k]).real)[:nev]; return e, products, 0.0
                e = np.sort(np.linalg.eigvals(Hs[:j+1, :j+1]).real)[:nev]
                ok = e_old is not None and len(e_old) == len(e) and all(abs(a-b) < tol*max(1.0, abs(a)) for a, b in zip(e, e_old))
                change = float(np.max(np.abs(e-e_old))) if e_old is not None and len(e_old) == len(e) else np.inf
                hits = hits+1 if ok else 0; e_old = e
                if hits >= stable:
                    return e, products, change
        Hm = host(Hd); ev, ey = np.linalg.eig(Hm[:k, :k]); order = np.argsort(ev.real)[:nev]
        comb = ey[:, order].sum(axis=1)
        if not np.iscomplexobj(np.zeros(1, dtype)):
            comb = comb.real if np.linalg.norm(comb.real) > np.linalg.norm(comb.imag) else comb.imag
        v = xp.asarray(comb.astype(dtype)) @ V[:k]; v /= xp.linalg.norm(v)
    return e_old, products, change


class RowsEngine:
    """Ring-of-rows sector of the nearest-neighbour Heisenberg model on an Lx x Ly torus (rows along x)."""
    def __init__(self, Lx, Ly, lattice='square', Kx=0, ky=0, px=1, py=1, device='cpu', rowbasis='dimer', J=1.0, verbose=False):
        t0 = time.perf_counter(); verbose = verbose or bool(os.environ.get('EDC_ROWS_VERBOSE'))
        if lattice == 'kagome' and rowbasis == 'dimer':
            rowbasis = 'trimer'
        b = build_sector(Lx, Ly, lattice == 'triangular', Kx, ky, py, px, rowbasis, verbose, device, kagome=(lattice == 'kagome'))
        if b.get('gpu') is not None:
            import cupy as cp
            g = b.pop('gpu'); b.pop('C'); self.info = b; self.dim = int(g['dim']); self.nnz = int(g['nnz']); self.real = True; self.J = float(J)
            self.dtype = np.float64; self.device = device; self.xp = cp; self.compact = bool(g['compact']); self._host = None
            self._rp, self._ci, self._vals = g['rp'], g['ci'], g['vals']
            self.stored_bytes = int(self._rp.nbytes+self._ci.nbytes+sum(v.nbytes for v in self._vals)); self.gpu_setup = True
            self._k = cp.RawKernel(kernel_source(True, 'cuda', self.compact), 'rows_spmv'); self.lanes = 1
            best = self.time_product(5); self.lane_times = {1: best}
            for w in (8, 16, 32):
                k_old, l_old = self._k, self.lanes
                self._k = cp.RawKernel(kernel_source(True, 'cuda', self.compact, w), 'rows_spmv'); self.lanes = w
                t = self.time_product(5); self.lane_times[w] = t
                if t < best:
                    best = t
                else:
                    self._k, self.lanes = k_old, l_old
            self.setup_seconds = time.perf_counter()-t0
            return
        C = b.pop('C'); C.sort_indices(); self.gpu_setup = False
        self.info = b; self.dim = int(C.shape[0]); self.nnz = int(C.nnz); self.real = bool(b['real']); self.J = float(J)
        self.dtype = np.float64 if self.real else np.complex128; self.device = device
        rp = np.ascontiguousarray(C.indptr, np.int64); ci = np.ascontiguousarray(C.indices, np.int32)
        va = np.ascontiguousarray(C.data, self.dtype)
        self.compact = False; vals = (va,)
        if self.real and os.environ.get('EDC_ROWS_COMPACT', '1') != '0':     # few distinct values: table + 2-byte code per entry
            lib = _compile_c(TABLES_C); lib.rows_value_codes.restype = ctypes.c_int64
            lib.rows_value_codes.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            code = np.zeros(max(1, len(va)), np.uint16); vt = np.zeros(65536)
            nv = lib.rows_value_codes(len(va), va.ctypes.data, code.ctypes.data, vt.ctypes.data)
            if nv >= 0:
                self.compact = True; self.values = int(nv); vals = (code, np.ascontiguousarray(vt[:max(1, nv)]))
        self.stored_bytes = int(rp.nbytes+ci.nbytes+sum(v.nbytes for v in vals))
        if device == 'gpu':
            import cupy as cp
            self.xp = cp; self.lanes = 1
            self._k = cp.RawKernel(kernel_source(self.real, 'cuda', self.compact), 'rows_spmv')
            self._rp, self._ci, self._vals = cp.asarray(rp), cp.asarray(ci), tuple(cp.asarray(v) for v in vals)
            if self.real and self.dim > 1000:                       # several lanes per row: keep the fastest of 1, 8, 16, 32
                best = self.time_product(5); self.lane_times = {1: best}
                for w in (8, 16, 32):
                    k_old, l_old = self._k, self.lanes
                    self._k = cp.RawKernel(kernel_source(True, 'cuda', self.compact, w), 'rows_spmv'); self.lanes = w
                    t = self.time_product(5); self.lane_times[w] = t
                    if t < best:
                        best = t
                    else:
                        self._k, self.lanes = k_old, l_old
        else:
            self.xp = np; self._lib = _compile_c(kernel_source(self.real, 'c', self.compact)); P = ctypes.c_void_p
            self._lib.rows_spmv.argtypes = [ctypes.c_int64]+[P]*(4+len(vals)); self._rp, self._ci, self._vals = rp, ci, vals
        self._host = (rp, ci, va)
        self.setup_seconds = time.perf_counter()-t0

    def matvec(self, x, y=None):
        if y is None:
            y = self.xp.empty(self.dim, self.dtype)
        if self.device == 'gpu':
            self._k(((self.dim*self.lanes+255)//256,), (256,), (np.int64(self.dim), self._rp, self._ci, *self._vals, x, y))
        else:
            x = np.ascontiguousarray(x, self.dtype)
            self._lib.rows_spmv(self.dim, self._rp.ctypes.data, self._ci.ctypes.data, *[v.ctypes.data for v in self._vals], x.ctypes.data, y.ctypes.data)
        return y

    def time_product(self, repeats=30):
        xp = self.xp; x = xp.asarray(np.random.default_rng(1).standard_normal(self.dim)).astype(self.dtype); y = self.matvec(x); best = 1e9
        for _ in range(repeats):
            t0 = time.perf_counter(); self.matvec(x, y)
            if self.device == 'gpu':
                xp.cuda.Device().synchronize()
            best = min(best, time.perf_counter()-t0)
        return best

    def ground_energy(self, nev=1, tol=1e-12, m=60):
        t0 = time.perf_counter()
        if self.dim <= 400 and self._host is not None:              # tiny sectors: dense
            rp, ci, va = self._host; ev = np.sort(np.linalg.eigvals(sp.csr_matrix((va, ci, rp), shape=(self.dim,)*2).toarray()).real)
            return [float(self.J*v) for v in ev[:nev]], 0, time.perf_counter()-t0, 0.0
        e, products, res = arnoldi_lowest(lambda v: self.matvec(v), self.dim, self.xp, self.dtype, nev=nev, tol=tol, m=m)
        return [float(self.J*v) for v in e], products, time.perf_counter()-t0, res
