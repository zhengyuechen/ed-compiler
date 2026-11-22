#!/usr/bin/env python3
"""Partly stored connectivity (SubEngine(stored='partial')): the first rows from the stored entries, the others on the
fly.  Checks the product against the fully stored and the on-the-fly product, after a retune as well, the ground
energy and the multi-vector product.  Usage: test_partial.py [cpu|gpu]"""
import sys, os, time
import numpy as np
import edc_lattices as L, edc_sub

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
worst = 0.0
for kind, a, nup, k, ir, flip, J in (('kagome', (2, 2, -2, 2), 12, (0.0, 0.0), 'X0', 1, (1.0, 0.05)),
                                     ('kagome', (3, 0, 0, 3), 15, (0.0, 2*np.pi/3), 'X0', 0, (1.0, 0.2)),
                                     ('triangular', (4, 2, -2, 6), 14, (0.0, 0.0), 'X0', 1, (1.0, 0.1))):
    cl = L.LatticeCluster(kind, a[:2], a[2:])
    def model(J1, J2):
        m = cl.heisenberg(J1, J2); m.keep_zero = True; return m
    full = edc_sub.SubEngine(model(*J), nup, cl, k, ir, flip, device=device, stored=True)
    fly = edc_sub.SubEngine(model(*J), nup, cl, k, ir, flip, device=device, stored=False)
    xp = full.xp; n = full.dim; rng = np.random.default_rng(1)
    x = rng.standard_normal(n) if full.real else rng.standard_normal(n)+1j*rng.standard_normal(n); x = xp.asarray(x)
    line = f'{kind} {a} N={cl.N} dim={n} {"real" if full.real else "complex"}:'
    for frac in (0.07, 0.4, 0.9):
        os.environ['EDC_STORED_FRACTION'] = str(frac)
        part = edc_sub.SubEngine(model(*J), nup, cl, k, ir, flip, device=device, stored='partial')
        del os.environ['EDC_STORED_FRACTION']
        assert 0 < part.stored_rows < n, (part.stored_rows, n)
        d = []
        for JJ in (J, (0.7, -0.3)):
            for e in (full, fly, part):
                e.retune(model(*JJ))
            y0 = full.matvec(x, xp.zeros_like(x)); y1 = fly.matvec(x, xp.zeros_like(x)); y2 = part.matvec(x, xp.zeros_like(x))
            y3 = part.matvec(x, y2.copy(), accumulate=True)
            d += [float(abs(y2-y0).max()), float(abs(y2-y1).max()), float(abs(y3-2*y0).max())]
        E0 = full.ground_energy()[0]; E2 = part.ground_energy()[0]; d.append(abs(float(E0)-float(E2)))
        R = 3; X = xp.asarray(rng.standard_normal((n, R)) if full.real else rng.standard_normal((n, R))+1j*rng.standard_normal((n, R)))
        Y0 = full.batched(R)(X, xp.zeros_like(X)); Y2 = part.batched(R)(X, xp.zeros_like(X))
        Y3 = part.batched(R)(X, Y2.copy(), accumulate=True)
        d += [float(abs(Y2-Y0).max()), float(abs(Y3-2*Y0).max())]
        assert float(abs(y0).max()) > 0.1 and float(E0) < -1.0
        worst = max(worst, max(d)); line += f'  {part.stored_rows/n:.2f} of the rows stored: {max(d):.1e}'
        for e in (full, fly):
            e.retune(model(*J))
    print(line + f'  (E0 = {float(E0):.10f})')
# entry formats: 5 bytes (one-byte value codes, when there are fewer than 256) against 6 and 8 bytes
for kind, a, nup, k, ir, flip, J in (('kagome', (3, 0, 0, 3), 15, (0.0, 2*np.pi/3), 'X0', 0, (1.0, 0.2)), ('kagome', (2, 2, -2, 2), 12, (0.0, 0.0), 'X0', 1, (1.0, 0.05))):
    cl = L.LatticeCluster(kind, a[:2], a[2:]); res = {}
    for fmt in (5, 6, 8):
        e = edc_sub.SubEngine(cl.heisenberg(*J), nup, cl, k, ir, flip, device=device, stored=False)
        if not e.build_stored(fmt=fmt):
            continue
        xp = e.xp; rng = np.random.default_rng(3); n = e.dim
        x = xp.asarray(rng.standard_normal(n) if e.real else rng.standard_normal(n)+1j*rng.standard_normal(n))
        X = xp.asarray(rng.standard_normal((n, 2)) if e.real else rng.standard_normal((n, 2))+1j*rng.standard_normal((n, 2)))
        res[fmt] = (e.matvec(x, xp.zeros_like(x)), e.batched(2)(X, xp.zeros_like(X)) if fmt != 8 else None, e.stored_bytes, e.stored_format)
        assert e.stored_format == fmt
    fmts = sorted(res); ref = res[fmts[-1]]
    d = max([float(abs(res[f][0]-ref[0]).max()) for f in fmts]+[float(abs(res[f][1]-res[6][1]).max()) for f in fmts if f != 8 and 6 in res])
    worst = max(worst, d)
    print(f'{kind} {a} dim={n}: formats {fmts}, stored MiB {[round(res[f][2]/2**20, 1) for f in fmts]}, largest difference {d:.1e}')
assert worst < 1e-10, worst
print('partial: all checks passed', device)
