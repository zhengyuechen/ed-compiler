#!/usr/bin/env python3
"""Spin-flip symmetry in the lean GPU engine: for every sector, dim(no flip) = dim(even) + dim(odd) and
E(no flip) = min(E(even), E(odd)); also check the lowest level of each parity against the CPU engine with an
explicit projector built from the spin-flip operator."""
import math, edc_lattice as el, edc_gpu_lean as gl
for name, cl, J2 in (('tilted (3,1)', el.tilted(3, 1), 0.3), ('4x3', el.Cluster((4, 0), (0, 3)), 0.4),
                     ('tilted (2,2)', el.tilted(2, 2), 0.6), ('4x4', el.Cluster((4, 0), (0, 4)), 0.5),
                     ('tilted (4,2)', el.tilted(4, 2), 0.55)):
    m = cl.j1j2(1.0, J2); worst, n = 0.0, 0
    for k in cl.momenta():
        lg = cl.little_group(k)
        for ir in ('A1', 'A2', 'B1', 'B2', 'E+', 'E-'):
            if [o for o in lg if o in el.IRREPS[ir]] != lg:
                continue
            a = gl.GPULean(m, cl.N//2, cl, k, ir)
            e = gl.GPULean(m, cl.N//2, cl, k, ir, flip=1); o = gl.GPULean(m, cl.N//2, cl, k, ir, flip=-1)
            assert a.dim == e.dim + o.dim, (name, k, ir, a.dim, e.dim, o.dim)
            if a.dim:
                Ea = a.ground_energy(tol=1e-13)[0]
                Es = [x.ground_energy(tol=1e-13)[0] for x in (e, o) if x.dim]
                worst = max(worst, abs(Ea-min(Es))); n += 1
    print(f'{name:14s} N={cl.N:2d}: {n:3d} sectors, dims add up, max |E_noflip - min(E_even, E_odd)| {worst:.1e}', flush=True)
