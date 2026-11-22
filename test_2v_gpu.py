#!/usr/bin/env python3
"""Two-vector Lanczos (accumulating matvec) vs the three-vector form, real and complex sectors."""
import math, edc_lattice as el, edc_gpu_lean as gl
worst = 0.0; n = 0
for cl, J2 in ((el.tilted(4, 2), 0.55), (el.Cluster((4, 0), (0, 4)), 0.5), (el.tilted(3, 1), 0.3)):
    m = cl.j1j2(1.0, J2)
    for k in cl.momenta()[:6]:
        g = gl.GPULean(m, cl.N//2, cl, k, 'A1' if 'Mx' not in cl.little_group(k) else 'A1')
        if not g.dim:
            continue
        a = g.ground_energy(tol=1e-13, two_vectors=False)[0]; b = g.ground_energy(tol=1e-13, two_vectors=True)[0]
        worst = max(worst, abs(a-b)); n += 1
        a2 = g.ground_energy(tol=1e-13, nev=2, two_vectors=False)[0]; b2 = g.ground_energy(tol=1e-13, nev=2, two_vectors=True)[0]
        worst = max(worst, max(abs(x-y) for x, y in zip(a2, b2)))
print(f'two-vector vs three-vector Lanczos: {n} sectors (real and complex), max |dE| {worst:.1e}')
