#!/usr/bin/env python3
"""edc_scan — a parameter scan solved as one family (reduced basis on one stored connectivity).

The Hamiltonian is linear in the run-time coupling table, so the P Hamiltonians of a scan are combinations of a few of
them (T = rank of the P coupling tables; a J1-J2 scan at fixed J1 has T = 2).  The scan keeps one engine (one
connectivity, SubEngine.retune between points) and a small set of exact ground states, the reduced basis V:
  1. solve one scan point exactly (Lanczos started from the current reduced solution), orthogonalise, append to V;
  2. apply the T probe Hamiltonians once to the new vector and keep the parts outside the span of V:  W_s = (1 - V V^+) H_s v;
  3. for every scan point p: diagonalise the m x m matrix V^+ H(p) V; the residual H(p) V c - E V c is a combination of the
     stored W, so its norm costs no product;
  4. repeat at the point with the largest estimated error until every point is below the tolerance.
Products are spent only in steps 1 and 2; their number does not depend on the number of scan points.
A small residual proves that a reduced solution is an eigenstate, not that it is the lowest: a level with a hidden quantum
number (total spin) can cross.  Protection: sentinel points and a final audit solved from random vectors, and exact solves
started from guess + random (see start_vector).
Lowest level of the sector only.  Memory: (1 + T) m vectors of the sector; with light=True only the m basis vectors are
kept and the T m vectors H_s v_i are recomputed when a vector is added (T m more products per addition).
"""
import time, math
import numpy as np
import scipy.linalg as sl
import edc_sub
from edc_hubbard_sym import ground_state_vector


def _ops(xp, real):
    """y += a x with a real or complex scalar, and the inner product <a|b>."""
    if xp is np:
        from scipy.linalg import blas
        ax = blas.daxpy if real else blas.zaxpy
        dot = (lambda a, b: float(np.dot(a, b))) if real else (lambda a, b: complex(np.vdot(a, b)))
        return (lambda a, x, y: ax(x, y, a=a)), dot
    import cupy as cp
    from edc_gpu import _dotc, _ddot
    k = cp.ElementwiseKernel('T x, T a', 'T y', 'y = y + a * x', 'edc_scan_axpy')
    dt = np.float64 if real else np.complex128
    return (lambda a, x, y: k(x, dt(a), y)), (_ddot if real else (lambda a, b: complex(_dotc(a, b))))


def affine_terms(tables):
    """tables: P coupling tables (rows).  Returns (probes, A): T rows that span all of them and the P x T coefficients
    with tables[p] = sum_s A[p, s] tables[probes[s]]."""
    C = np.asarray(tables, float)
    _, R, piv = sl.qr(C.T, mode='economic', pivoting=True)
    d = np.abs(np.diag(R)); T = int(np.sum(d > 1e-10*d[0]))
    probes = sorted(int(p) for p in piv[:T])
    A = np.linalg.lstsq(C[probes].T, C.T, rcond=None)[0].T
    assert np.abs(A @ C[probes]-C).max() < 1e-10*max(1.0, np.abs(C).max()), 'coupling tables are not spanned by the probes'
    return probes, A


def start_vector(eng, guess, mode, seed, eta=None):
    """Start vector of an exact solve.  'random': a random vector (finds the lowest level of the sector whatever the
    earlier points gave).  'guess': the vector handed in (fastest, but Lanczos started from an eigenvector of a hidden
    symmetry, e.g. another total spin, never leaves it: a level crossing is missed).  'mixed': guess + eta * random
    (unit vectors), which keeps a component on every level; with eta = 0.5 it passed the crossing tests of LOG section 63,
    with eta = 0.1 and 0.02 it stopped on the wrong level just behind a crossing (errors 5e-3 and 4e-2)."""
    from edc_hubbard_sym import _random_vector
    xp = eng.xp
    if guess is None or mode == 'random':
        return None
    if mode == 'guess':
        return guess
    import os
    eta = float(os.environ.get('EDC_SCAN_ETA', 0.5)) if eta is None else eta
    _, dot = _ops(xp, eng.real)
    r = _random_vector(xp, int(eng.dim), eng.real, seed)
    r *= eta/math.sqrt(abs(dot(r, r)))
    r += guess*(1.0/math.sqrt(abs(dot(guess, guess))))
    return r


def reduced_basis_scan(eng, models, tol=1e-10, solve_tol=1e-12, max_basis=64, start='random', sentinels=3, audit=2, solve_rtol='auto',
                       certify=False, log=None, light=False):
    """Lowest level of the engine's sector at every model of the list (same bonds, different couplings).
    tol: target for the estimated energy error, relative to max(1, |E|).
    solve_rtol: what every exact solve has to reach beyond the energy criterion of the Lanczos routine.  'auto': the
    energy-error estimate 2 r^2 / gap of its Ritz pair below tol / 10 (r the residual); a number: the residual, relative
    to max(1, |E|); None: the energy criterion alone.  A basis built from vectors that are converged in energy only
    cannot reach tol in a sector whose two lowest levels are close: every point is then solved exactly (LOG section 76).
    start: start vector of the exact solves after the sentinels, see start_vector.
    sentinels: this many points, evenly spaced over the list, are solved first from random vectors.
    audit: after convergence this many untrained points are solved from random vectors and compared; a disagreement adds
    the point to the basis and the loop continues.  The residual shows that a reduced solution is an eigenstate, not that
    it is the lowest one; sentinels and audit are the protection against a level that no exact solve has seen.
    certify: at the end form the residual vector of every point explicitly (no products, m T vector updates per point).
    light: keep only the basis vectors.  The residual then comes from the raw inner products <H_s v_i|H_t v_j> minus the
    part inside the basis (a difference of numbers of the size E^2: residuals below about 1e-7 |E| are not resolved).
    Returns dict(E, error_estimate, residual, basis, trained, audit, products..., seconds)."""
    xp, n, P = eng.xp, int(eng.dim), len(models)
    axpy, dot = _ops(xp, eng.real)
    solve_etol = None
    if solve_rtol == 'auto':
        solve_rtol, solve_etol = None, max(0.1*tol, 1e-14)
    dts = np.float64 if eng.real else np.complex128
    tables = [edc_sub.cpl_array(eng._relabel(m)) for m in models]
    probes, A = affine_terms(tables); T = len(probes)
    t0 = time.perf_counter()
    st = dict(solves=0, lanczos_products=0, replay_products=0, term_products=0, retunes=0)
    V, W = [], [[] for _ in range(T)]
    box = dict(h=np.zeros((T, 0, 0), dts), G=np.zeros((T, T, 0, 0), dts))
    cur = [None]
    vsize = 8 if eng.real else 16
    if eng.device == 'gpu':
        import cupy as cp
        free = lambda: cp.cuda.runtime.memGetInfo()[0]
    else:
        import os
        free = lambda: 0.5*os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')

    def tune(p):
        if cur[0] != p:
            eng.retune(models[p]); cur[0] = p; st['retunes'] += 1

    def reduced(p):
        m = len(V); h, G = box['h'], box['G']
        hp = np.tensordot(A[p], h, axes=1); hp = (hp+hp.conj().T)/2
        w, c = sl.eigh(hp); c0 = c[:, 0]
        Gp = np.einsum('s,t,stij->ij', A[p], A[p], G)
        if light:                                                # raw Gram minus the part inside the span of V
            hr = np.tensordot(A[p], h, axes=1); Gp = Gp-hr.conj().T @ hr
        r2 = max(float(np.real(c0.conj() @ Gp @ c0)), 0.0)
        far = w[w-w[0] > 1e-6*max(1.0, abs(w[0]))]              # a degenerate lowest level: the gap to the next distinct one
        gap = float(far[0]-w[0]) if len(far) else None
        return float(w[0]), c0, math.sqrt(r2), gap

    def combine(coefs, vecs):
        out = xp.zeros(n, dts)
        for c, v in zip(coefs, vecs):
            axpy(c, v, out)
        return out

    def solve(p, v0):
        """Exact lowest state at point p."""
        tune(p)
        keep = max(0, int(0.5*free()/(n*vsize))-4)               # Krylov vectors kept instead of a replay, if they fit
        E, it, _ = eng.ground_energy(tol=solve_tol, v0=v0, keep=keep if keep >= 8 else False, rtol=solve_rtol, etol=solve_etol)
        st['solves'] += 1; st['lanczos_products'] += int(it)
        psi, _, res = ground_state_vector(eng)
        st['replay_products'] += 1 + (0 if 'psi' in eng._lanczos or 'dense' in eng._lanczos else int(it))
        return float(np.atleast_1d(E)[0]), xp.asarray(psi, dtype=dts)

    def append(psi):
        """Orthogonalise psi against V and add it; False if it was already inside the span."""
        m = len(V)
        for _ in range(2):
            for u in V:
                axpy(-dot(u, psi), u, psi)
        nrm = math.sqrt(abs(dot(psi, psi)))
        if nrm < 1e-9:
            return False
        psi *= 1.0/nrm
        V.append(psi)
        hn = np.zeros((T, m+1, m+1), dts); hn[:, :m, :m] = box['h']
        Gn = np.zeros((T, T, m+1, m+1), dts); Gn[:, :, :m, :m] = box['G']
        g = np.zeros((T, m), dts); new = []
        if light:
            for s in range(T):
                tune(probes[s])
                w = xp.zeros(n, dts); eng.matvec(psi, w); st['term_products'] += 1
                col = np.array([dot(u, w) for u in V], dts)
                hn[s, :, m] = col; hn[s, m, :] = col.conj(); new.append(w)
            wv = xp.zeros(n, dts)
            for s in range(T):
                tune(probes[s])
                for i in range(m):                               # H_s v_i again, only for its inner products with the new H_t v_m
                    eng.matvec(V[i], wv); st['term_products'] += 1
                    for t in range(T):
                        v = dot(wv, new[t]); Gn[s, t, i, m] = v; Gn[t, s, m, i] = np.conj(v)
            for s in range(T):
                for t in range(T):
                    Gn[s, t, m, m] = dot(new[s], new[t])
            del new, wv
            box['h'], box['G'] = hn, Gn
            return True
        for s in range(T):
            tune(probes[s])
            w = xp.zeros(n, dts); eng.matvec(psi, w); st['term_products'] += 1
            col = np.zeros(m+1, dts)
            for _ in range(2):
                for i, u in enumerate(V):
                    d = dot(u, w); axpy(-d, u, w); col[i] += d
            hn[s, :, m] = col; hn[s, m, :] = col.conj()
            for i in range(m):                                   # older W lose their component along the new vector
                g[s, i] = dot(psi, W[s][i]); axpy(-g[s, i], psi, W[s][i])
            new.append(w)
        for s in range(T):
            for t in range(T):
                Gn[s, t, :m, :m] -= np.outer(g[s].conj(), g[t])
        for s in range(T):
            W[s].append(new[s])
        for s in range(T):
            for t in range(T):
                for i in range(m+1):
                    v = dot(W[s][i], W[t][m]); Gn[s, t, i, m] = v; Gn[t, s, m, i] = np.conj(v)
        box['h'], box['G'] = hn, Gn
        return True

    def all_points():
        E = np.zeros(P); R = np.zeros(P); est = np.zeros(P); coef = [None]*P
        for q in range(P):
            E[q], coef[q], R[q], gap = reduced(q)
            est[q] = (2*R[q]**2/gap if gap is not None else R[q])/max(1.0, abs(E[q]))
        return E, R, est, coef

    trained, hist, audits = [], [], []
    for p in sorted({int(round(x)) for x in np.linspace(0, P-1, max(1, min(sentinels, P)))}):
        _, psi = solve(p, None); append(psi); trained.append(p)
    rng = np.random.default_rng(7); left = audit
    while True:
        E, R, est, coef = all_points()
        cand = est.copy(); cand[trained] = 0.0
        p = int(np.argmax(cand)); m = len(V)
        hist.append((m, float(cand[p]), float(R.max())))
        if log:
            log(f'basis {m}: largest estimated error {cand[p]:.2e} at point {p}, largest residual {R.max():.2e}')
        if cand[p] < tol or m >= max_basis or len(trained) >= P:
            rest = [q for q in range(P) if q not in trained]
            if left <= 0 or not rest or m >= max_basis:
                break
            q = int(rng.choice(rest)); left -= 1                # audit: an exact solve from a random vector
            Eq, psi = solve(q, None); d = float(Eq-E[q]); audits.append((q, d))
            if log:
                log(f'audit at point {q}: exact - reduced = {d:.2e}')
            trained.append(q)
            if abs(d) > 100*tol*max(1.0, abs(Eq)):
                append(psi); left = audit                       # a level the basis did not hold: start the audit again
            continue
        _, psi = solve(p, start_vector(eng, combine(coef[p], V), start, 1000+len(trained)))
        ok = append(psi); trained.append(p)
        if not ok and cand[p] < 100*tol:
            break
    E, R, est, coef = all_points()
    out = dict(E=E.tolist(), error_estimate=est.tolist(), residual=R.tolist(), basis=len(V), trained=sorted(set(trained)), terms=T,
               history=hist, audit=audits, start=start, **st)
    if certify and not light:                                    # explicit residual vectors: sum_i c_i sum_s A[p, s] W_s,i
        Rx = np.zeros(P)
        for q in range(P):
            rv = xp.zeros(n, dts)
            for i in range(len(V)):
                for s in range(T):
                    axpy(coef[q][i]*A[q, s], W[s][i], rv)
            Rx[q] = math.sqrt(abs(dot(rv, rv)))
        out['residual_explicit'] = Rx.tolist()
    out['products'] = st['lanczos_products']+st['replay_products']+st['term_products']
    out['seconds'] = time.perf_counter()-t0
    out['_basis'] = (V, coef)
    return out
