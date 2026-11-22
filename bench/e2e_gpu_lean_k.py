#!/usr/bin/env python3
"""Memory-lean any-momentum ring.  Usage: e2e_gpu_lean_k.py <L> <q> [nup]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cupy as cp, edc, edc_k, edc_gpu
L, q = int(sys.argv[1]), int(sys.argv[2]); nup = int(sys.argv[3]) if len(sys.argv) > 3 else L//2
t0 = time.perf_counter(); e = edc_gpu.GPUKEngineLean(edc.ring(L), nup, edc.ring_translations(L), edc_k.ring_characters(L, q))
setup = time.perf_counter()-t0
E, it, t = e.ground_energy()
print(json.dumps(dict(code='edc-gpu-lean-k', model=f'ring:{L}', q=q, nup=nup, dim=e.dim, full_dim=e.full_dim, setup_seconds=setup,
                      device_gib=e.device_gib, iterations=it, lanczos_seconds=t, seconds_per_iteration=t/it, E0=E, E0_per_site=E/L)), flush=True)
