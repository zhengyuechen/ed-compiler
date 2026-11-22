#!/usr/bin/env python3
"""General-lattice memory-lean GPU engine vs the CPU any-group engine, every momentum and 1D irrep of small clusters."""
import edc, edc_k, edc_lattice as el, edc_gpu_lean as gl

def sectors(cl):
    for k in cl.momenta():
        lg = cl.little_group(k)
        for ir in ('A1', 'A2', 'B1', 'B2', 'E+', 'E-'):
            if [o for o in lg if o in el.IRREPS[ir]] != lg:
                continue
            yield k, ir

for name, cl, J2 in (('tilted (3,1)', el.tilted(3, 1), 0.3), ('4x3', el.Cluster((4, 0), (0, 3)), 0.4),
                     ('tilted (2,2)', el.tilted(2, 2), 0.6), ('4x4', el.Cluster((4, 0), (0, 4)), 0.5),
                     ('tilted (4,2)', el.tilted(4, 2), 0.55)):
    m = cl.j1j2(1.0, J2)
    worst, n, nreal = 0.0, 0, 0
    for k, ir in sectors(cl):
        perms, chars, ops = cl.group(k, ir)
        c = edc_k.KEngine(m, cl.N//2, perms, chars, threads=8)
        gp = gl.GPULean(m, cl.N//2, cl, k, ir)
        assert c.dim == gp.dim, (name, k, ir, c.dim, gp.dim)
        if c.dim:
            worst = max(worst, abs(c.ground_energy(tol=1e-13)[0]-gp.ground_energy(tol=1e-13)[0])); n += 1; nreal += gp.real
    print(f'{name:14s} N={cl.N:2d} J2={J2}: {n:3d} sectors ({nreal} real), max |E_gpu - E_cpu| {worst:.1e}', flush=True)
