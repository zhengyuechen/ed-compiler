#!/usr/bin/env python3
"""GPU any-momentum engine vs the CPU one, every momentum sector of small systems."""
import numpy as np, edc, edc_k, edc_gpu
cases = [('ring 12', edc.ring(12), 6, edc.ring_translations(12), [edc_k.ring_characters(12, q) for q in range(12)]),
         ('ring 12, nup 5, field', None, 5, edc.ring_translations(12), [edc_k.ring_characters(12, q) for q in range(12)]),
         ('square 4x3', edc.square(4, 3), 6, edc.torus_translations(4, 3), [edc_k.torus_characters(4, 3, qx, qy) for qy in range(3) for qx in range(4)]),
         ('ring 20', edc.ring(20), 10, edc.ring_translations(20), [edc_k.ring_characters(20, q) for q in range(20)])]
m = edc.ring(12); [m.field(i, 0.3) for i in range(12)]
cases[1] = ('ring 12, nup 5, field', m, 5, edc.ring_translations(12), cases[1][4])
for name, model, nup, perms, chars_of in cases:
    worst, dims = 0.0, 0
    for chars in chars_of:
        c = edc_k.KEngine(model, nup, perms, chars, threads=4); g = edc_gpu.GPUKEngine(model, nup, perms, chars)
        assert c.dim == g.dim, (name, c.dim, g.dim)
        dims += g.dim
        if g.dim:
            worst = max(worst, abs(c.ground_energy(tol=1e-13)[0]-g.ground_energy(tol=1e-13)[0]))
    print(f'{name:24s} sum of dims {dims:8d}  max |E_gpu - E_cpu| {worst:.1e}', flush=True)
