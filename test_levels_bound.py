#!/usr/bin/env python3
"""edc_levels.lowest_levels against ARPACK (restarted, fully reorthogonalised Lanczos) on every block of a few sectors.
Usage: test_levels_bound.py [cpu|gpu]"""
import sys, math
import numpy as np, scipy.sparse.linalg as sla
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub, edc_levels

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
worst = 0.0
for kind, a, nup, nev, J2 in (('kagome', (2, 2, -2, 2), 13, 4, 0.0), ('kagome', (2, 2, -2, 2), 15, 4, 0.1), ('kagome', (3, 0, 0, 3), 18, 4, 0.0)):
    cl = L.LatticeCluster(kind, a[:2], a[2:]); model = cl.heisenberg(1.0, J2) if J2 else cl.heisenberg(1.0)
    steps = blocks = 0; err = 0.0
    for k, ir, fl, pops, mult in hs.sectors(cl, nup, cl.N-nup):
        e = edc_sub.SubEngine(model, nup, cl, k, ir, fl, device=device, point_ops=pops)
        if e.dim < 3000:
            continue
        lev, it, conv, info = edc_levels.lowest_levels(e, nev)
        assert conv and len(lev) == nev, (kind, nup, ir, it)
        xp = e.xp; dt = np.float64 if e.real else np.complex128; x = xp.zeros(e.dim, dt); y = xp.zeros(e.dim, dt)
        def mv(z):
            x[:] = xp.asarray(z.reshape(-1)); e.matvec(x, y); return (y.get() if device == 'gpu' else y.copy())
        vals = sorted(sla.eigsh(sla.LinearOperator((e.dim, e.dim), matvec=mv, dtype=dt), k=nev+6, which='SA', tol=1e-11, ncv=44, return_eigenvectors=False))
        ref = []
        for v in vals:
            if not ref or abs(v-ref[-1]) > 1e-8*abs(vals[0]):
                ref.append(v)
        err = max(err, max(abs(p-q) for p, q in zip(lev, ref[:nev]))); steps += it; blocks += 1
    worst = max(worst, err)
    print(f'{kind} {a} N={cl.N} nup={nup} J2={J2} nev={nev}: {blocks} blocks, {steps} steps, largest difference to ARPACK {err:.1e}')
assert worst < 1e-9, worst
print('levels (bound rule): all checks passed', device)
