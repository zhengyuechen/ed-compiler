#!/usr/bin/env python3
"""Memory-lean GPU engine with the rectangle's space group (A1).  Usage: e2e_gpu_lean_rect.py <Lx> <Ly>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cupy as cp, edc, edc_gpu
Lx, Ly = int(sys.argv[1]), int(sys.argv[2])
m = edc.square(Lx, Ly)
t0 = time.perf_counter(); e = edc_gpu.GPUEngineLeanRect(m, m.n//2, Lx, Ly); setup = time.perf_counter()-t0
mv = e.time_matvec(reps=2); E, it, t = e.ground_energy()
print(json.dumps(dict(code='edc-gpu-lean-rect', model=f'rect:{Lx}x{Ly}', group_order=e.group_order, dim=e.dim, full_dim=e.full_dim,
                      setup_seconds=setup, device_gib=e.device_gib, matvec_seconds=mv, iterations=it, lanczos_seconds=t,
                      seconds_per_iteration=t/it, E0=E, E0_per_site=E/m.n)), flush=True)
