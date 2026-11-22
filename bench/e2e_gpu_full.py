#!/usr/bin/env python3
"""GPU end-to-end with the whole setup on the GPU (symmetric models).  Usage: e2e_gpu_full.py <model>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp, edc_gpu, models
name = sys.argv[1]
m, g = models.build(name)
t0 = time.perf_counter(); e = edc_gpu.GPUEngineFull(m, m.n//2, g); setup = time.perf_counter()-t0
mv = e.time_matvec()
E, it, t = e.ground_energy()
free, total = cp.cuda.runtime.memGetInfo()
print(json.dumps(dict(code='edc-gpu-full', model=name, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), dim=e.dim,
                      full_dim=e.full_dim, setup_seconds=setup, nvrtc_seconds=e.compile_seconds, gpu_setup_seconds=e.gpu_setup_seconds,
                      device_gib=e.device_gib, pool_gib=cp.get_default_memory_pool().total_bytes()/2**30, matvec_seconds=mv,
                      iterations=it, lanczos_seconds=t, seconds_per_iteration=t/it, E0=E, E0_per_site=E/m.n)), flush=True)
