#!/usr/bin/env python3
"""Memory-lean GPU engine vs the table engine on the same model.  Usage: e2e_gpu_lean.py <model>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp, edc_gpu, models
name = sys.argv[1]; m, g = models.build(name)
t0 = time.perf_counter(); e = edc_gpu.GPUEngineLean(m, m.n//2, g); setup = time.perf_counter()-t0
mv = e.time_matvec(); E, it, t = e.ground_energy()
print(json.dumps(dict(code='edc-gpu-lean', model=name, dim=e.dim, full_dim=e.full_dim, setup_seconds=setup, device_gib=e.device_gib,
                      matvec_seconds=mv, iterations=it, lanczos_seconds=t, seconds_per_iteration=t/it, E0=E)), flush=True)
