#!/usr/bin/env python3
"""edc_levels — the lowest distinct levels of one block by plain two-vector Lanczos, stopped by an error bound.

Every Ritz value theta_i of the tridiagonal matrix comes with the residual of its Ritz vector for free,
r_i = beta_m |s_mi| (s_mi: last component of its eigenvector of the tridiagonal matrix).  Some exact level lies within r_i
of theta_i, and within r_i^2 / gap_i when theta_i is isolated (gap_i: distance to the nearest other level; here the nearest
Ritz value that is not a copy, which can overestimate it while a level in between has not appeared, hence a factor 10).
A Ritz value counts as a level when min(r_i, 10 r_i^2 / gap_i) < tol max(1, |E_0|); the run stops at the first
check at which the lowest nev distinct levels pass and no Ritz value below the highest of them fails, plus `confirm`
further checks that return the same levels.  Lanczos without reorthogonalisation returns converged levels again; the
copies are merged.  A level exactly degenerate inside the block is returned once.
Copies: two or more Ritz values within `merge` of each other count as one converged level, without looking at their
residuals, once they agree to tol; while a copy is still settling the group takes the value the level was accepted
with at an earlier check, or fails if there is none (the lowest member of a settling group was 1.7e-9 off in one of
363 levels).  Lanczos produces a copy only of a level that has converged, and the eigenvectors of the tridiagonal matrix
for numerically equal eigenvalues are arbitrary mixtures, so the last components of a converged value and of its
emerging copy cannot be told apart (with the residual test applied to each, runs for eight levels ended late or not
at all).

What no rule on one Lanczos run can exclude: a level whose weight in the start vector is so small that it has not
appeared yet.  `confirm` buys margin against that (an emerging copy of a level interrupts the count, so large values
may never be reached when many levels are asked for); a second run from another seed is the real check.

Against "a Ritz value that has not moved for 200 steps" (kagome 27, nev = 4, 66 blocks, reference ARPACK):
1.9 times fewer steps, largest error 7e-12 against 1e-9."""
import math
import numpy as np


def lowest_levels(e, nev, tol=1e-10, maxit=4000, seed=42, every=20, confirm=0, merge=1e-8, log=None):
    """Returns (levels, steps, converged, info).  e: an engine with matvec(x, y, accumulate), dim, real, xp (edc_sub.SubEngine).
    info: dict(residuals of the returned levels, first_pass: step at which the rule first held)."""
    from scipy.linalg import eigh_tridiagonal
    import edc_hubbard_sym as hs
    xp = e.xp
    axpy, dot = hs._vec_ops(xp, e.real, False)
    dt = np.float64 if e.real else np.complex128
    v = hs._random_vector(xp, e.dim, e.real, seed, False)
    v *= 1.0/math.sqrt(dot(v, v).real)
    u = xp.zeros(e.dim, dt)
    A, B = [], []
    top = min(maxit, e.dim); first = None; held = 0; last_lev = None; known = []   # known: levels accepted at any check so far
    lev, res = [], []
    for it in range(1, top+1):
        e.matvec(v, u, accumulate=True)
        al = float(np.real(dot(v, u))); axpy(-al, v, u); be = math.sqrt(float(np.real(dot(u, u))))
        A.append(al); B.append(be)
        end = it == top or be < 1e-12
        if (it % every == 0 or end) and it > 1:
            w, S = eigh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, min(it-1, 10*nev+60)))
            scale = max(1.0, abs(w[0])); r = be*np.abs(S[-1, :])
            lev, res, clean = [], [], True
            i = 0
            while i < len(w):
                j = i
                while j+1 < len(w) and w[j+1]-w[i] < merge*scale:
                    j += 1                                       # w[i..j]: one Ritz value and its copies
                if len(lev) == nev:
                    break
                val = float(w[i])
                if j > i:                                        # copies: converged once they agree to tol; a copy that is still
                    est = 0.0 if w[j]-w[i] < tol*scale else math.inf   # settling can lie on either side of the level,
                    if est:                                      # unless the level was accepted at an earlier check
                        near = [q for q in known if w[i]-merge*scale < q < w[j]+merge*scale]
                        if near:
                            est, val = 0.0, near[0]
                else:
                    d = np.abs(w-w[i]); far = d[d > merge*scale]
                    gap = float(far.min()) if len(far) else 1.0
                    est = min(r[i], 10*r[i]**2/gap)
                if est >= tol*scale:
                    if not end:
                        clean = False; break                     # a Ritz value below the nev-th level that is not a level yet
                else:
                    lev.append(val); res.append(float(est))
                    if not any(abs(val-q) < merge*scale for q in known):
                        known.append(val)
                i = j+1
            ok = clean and (len(lev) == nev or it == e.dim)
            if log:
                log(f'step {it}: {len(lev)} levels pass, clean {clean}')
            if ok:
                first = first or it
                same = last_lev is not None and len(last_lev) == len(lev) and max(abs(p-q) for p, q in zip(last_lev, lev)) < 10*tol*scale
                held = held+1 if same else 0
                last_lev = lev
                if held >= confirm or it == e.dim:
                    return lev, it, True, dict(residuals=res, first_pass=first)
            else:
                held = 0; last_lev = None
        if end:
            break
        u *= 1.0/be; v *= -be
        v, u = u, v
    return lev, it, False, dict(residuals=res, first_pass=first)
