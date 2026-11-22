#!/usr/bin/env python3
"""GPU end-to-end ground-state energy with edc_gpu.  Usage: e2e_gpu.py <model> <cpu threads for setup>"""
import sys, os, json, time, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp, edc_gpu, models
name, nt = sys.argv[1], int(sys.argv[2])
m, g = models.build(name)
t0 = time.perf_counter(); e = edc_gpu.GPUEngine(m, m.n//2, group=g, threads=nt); setup = time.perf_counter()-t0
mv = e.time_matvec()
E, it, t = e.ground_energy()
print(json.dumps(dict(code='edc-gpu', model=name, threads=nt, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                      dim=e.dim, setup_seconds=setup, cpu_setup_seconds=e.cpu_setup_seconds, nvrtc_seconds=e.compile_seconds,
                      upload_seconds=e.upload_seconds, device_gib=e.device_gib, matvec_seconds=mv, iterations=it,
                      lanczos_seconds=t, seconds_per_iteration=t/it, E0=E,
                      host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)), flush=True)
