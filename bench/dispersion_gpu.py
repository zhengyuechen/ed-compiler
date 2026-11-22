#!/usr/bin/env python3
"""Lowest energy in every momentum sector of the Heisenberg ring, Sz = 0 and Sz = 1, on the GPU.
Usage: dispersion_gpu.py <L>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cupy as cp, edc, edc_k, edc_gpu
L = int(sys.argv[1])
m = edc.ring(L); perms = edc.ring_translations(L)
for nup in (L//2, L//2+1):
    for q in range(L//2+1):
        t0 = time.perf_counter()
        e = edc_gpu.GPUKEngine(m, nup, perms, edc_k.ring_characters(L, q))
        E, it, t = e.ground_energy()
        print(json.dumps(dict(L=L, nup=nup, Sz=nup-L//2, q=q, k=2*np.pi*q/L, dim=e.dim, E=E, iterations=it, setup_seconds=e.setup_seconds,
                              lanczos_seconds=t, total_seconds=time.perf_counter()-t0)), flush=True)
        del e; cp.get_default_memory_pool().free_all_blocks()
