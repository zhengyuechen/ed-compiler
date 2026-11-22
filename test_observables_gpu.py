#!/usr/bin/env python3
"""Ground-state vector replay and M^2(Q) in a symmetric sector (GPU) vs the same quantity from the ground state of the
full S_z = 0 space without symmetries (CPU, scipy eigsh on the generated matrix-free kernel)."""
import math, numpy as np, edc, edc_lattice as el, edc_gpu as eg, edc_gpu_lean as gl
from scipy.sparse.linalg import LinearOperator, eigsh

for name, cl, J2 in (('4x4', el.Cluster((4, 0), (0, 4)), 0.0), ('4x4', el.Cluster((4, 0), (0, 4)), 0.4),
                     ('tilted (4,2)', el.tilted(4, 2), 0.0)):
    m = cl.j1j2(1.0, J2)
    g = gl.GPULean(m, cl.N//2, cl, (0.0, 0.0), 'A1', flip=1)
    E0, it, _ = g.ground_energy(tol=1e-14)
    psi, E, res = eg.ground_state_vector(g)
    Qsets = {'(pi,pi)': [(math.pi, math.pi)], '(pi,0)': [(math.pi, 0.0), (0.0, math.pi)]}
    gpu = {q: gl.m2(g, psi, cl, Qs) for q, Qs in Qsets.items()}
    # reference: full S_z = 0 space, no symmetry
    c = edc.Engine(m, cl.N//2, threads=8)
    Hop = LinearOperator((c.dim, c.dim), matvec=lambda x: c.matvec(np.ascontiguousarray(x, dtype=np.float64).ravel()), dtype=np.float64)
    ev, vec = eigsh(Hop, k=1, which='SA', tol=1e-13); phi = vec[:, 0]
    ref = {}
    for q, Qs in Qsets.items():
        o = edc.Engine(gl.structure_factor_model(cl, Qs), cl.N//2, threads=8)
        ref[q] = (2*float(phi @ o.matvec(np.ascontiguousarray(phi)))+0.75*cl.N)/(cl.N*(cl.N+2))
    print(f'{name:12s} J2={J2}: E0 sector {E0:.12f} full {ev[0]:.12f}; replay <H> {E:.12f} residual {res:.1e}; '
          + '; '.join(f'M2{q} gpu {gpu[q]:.10f} cpu {ref[q]:.10f}' for q in Qsets), flush=True)
