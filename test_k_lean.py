#!/usr/bin/env python3
"""Memory-lean any-momentum GPU engine vs the table-based GPU engine, every momentum of small systems."""
import edc, edc_k, edc_gpu
for name, model, nup, perms, chars_of in (
        ('ring 12', edc.ring(12), 6, edc.ring_translations(12), [edc_k.ring_characters(12, q) for q in range(12)]),
        ('ring 20', edc.ring(20), 10, edc.ring_translations(20), [edc_k.ring_characters(20, q) for q in range(20)]),
        ('ring 22, nup 12', edc.ring(22), 12, edc.ring_translations(22), [edc_k.ring_characters(22, q) for q in range(22)]),
        ('square 4x3', edc.square(4, 3), 6, edc.torus_translations(4, 3), [edc_k.torus_characters(4, 3, qx, qy) for qy in range(3) for qx in range(4)])):
    worst = 0.0
    for chars in chars_of:
        a = edc_gpu.GPUKEngine(model, nup, perms, chars); b = edc_gpu.GPUKEngineLean(model, nup, perms, chars)
        assert a.dim == b.dim, (name, a.dim, b.dim)
        if a.dim:
            worst = max(worst, abs(a.ground_energy(tol=1e-13)[0]-b.ground_energy(tol=1e-13)[0]))
    print(f'{name:18s} max |E_lean - E_table| over all momenta {worst:.1e}', flush=True)
