#!/usr/bin/env python3
"""Tables of the three-level comparison from the JSON files of levels.py.    compare.py <dir with the JSON files> [threads of the CPU reference, default 32]"""
import sys, os, json, glob
import numpy as np

d = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 32
load = lambda f: json.load(open(f)) if os.path.exists(f) else None
cases = sorted({os.path.basename(f)[5:-5] for f in glob.glob(d + '/gpu3_*.json')}, key=lambda s: (int(s.split('_')[0][1:]), s))
x = lambda a, b: f'{a/b:.1f}x' if a and b else '–'


def ms(v):
    return f'{v*1e3:.2f} ms' if v < 1 else f'{v:.2f} s'


print('# CPU engine against GPU v3 at three levels, same saved input (YC6, 7 rings, 60 spins, K = 0, FP64)\n')
print(f'CPU: {T} library threads on a whole 64-core node, BLAS on one thread.  GPU: one H100, 4 CPU cores.  Ratio = CPU time / GPU time.\n')
for c in cases:
    g = load(f'{d}/gpu3_{c}.json'); h = load(f'{d}/host_c{T}_{c}.json'); k = load(f'{d}/check_{c}.json'); P = g['problem']
    print(f'## {c.replace("_", ", ").replace("Sz", "S_z = ").replace("D", "D = ")}\n')
    print(f'State: two warm-up sweeps and {P["sweeps_at_D"]} at D with {P["made_with"]}, E = {P["E_state"]:.10f}, largest bond {max(P["bond_sizes_state"])}; {P["layers"]} layers; '
          f'the last sweep before the state was saved: {P["last_sweep_products"]} products in {P["last_sweep_seconds"]:.2f} s.')
    if k:
        print(f'Device product against host product on the GPU node, every layer: largest |w_device − w_host| / |w_host| = {k["worst"]:.1e}.')
    if h is None:
        print('\nCPU half not there yet.\n'); continue
    Lg, Lh = g['layers'], h['layers']
    dv = max(abs(a['vHv']-b['vHv']) for a, b in zip(Lg, Lh)); dz = max(abs(a['zHv']-b['zHv'])/max(1e-300, a['norm_Hv']) for a, b in zip(Lg, Lh))
    print(f'CPU node against GPU node, every layer: largest difference of v·Hv {dv:.1e}, of z·Hv / |Hv| {dz:.1e} (z a fixed random vector).\n')
    big = int(np.argmax([a['size'] for a in Lg])); fl = g['sum_flops']
    print('| level | CPU | H100 | ratio |\n|---|---|---|---|')
    print(f'| 1. one product, largest tensor (layer {big}, {Lg[big]["size"]:,} numbers) | {ms(Lh[big]["product_seconds_min"])} | {ms(Lg[big]["product_seconds_min"])} | {x(Lh[big]["product_seconds_min"], Lg[big]["product_seconds_min"])} |')
    print(f'| 1. one product of every layer, summed | {ms(h["sum_product_seconds"])} | {ms(g["sum_product_seconds"])} | {x(h["sum_product_seconds"], g["sum_product_seconds"])} |')
    print(f'| 1. matrix-product throughput over those products ({fl/1e9:.1f} GFLOP of work) | {h["gflops_all_layers"]:.0f} GFLOP/s | {g["gflops_all_layers"]:.0f} GFLOP/s | |')
    print(f'| 2. the solve of every layer, summed ({h["sum_solve_products"]} / {g["sum_solve_products"]} products) | {ms(h["sum_solve_seconds"])} | {ms(g["sum_solve_seconds"])} | {x(h["sum_solve_seconds"], g["sum_solve_seconds"])} |')
    for a, b in zip(h['level3']['sweeps'], g['level3']['sweeps']):
        print(f'| 3. {a["sweep"]} sweep, whole ({a["products"]} / {b["products"]} products) | {a["seconds"]:.2f} s | {b["seconds"]:.2f} s | {x(a["seconds"], b["seconds"])} |')
        for key, name in (('solve', 'solves'), ('moves', 'moves'), ('environments', 'environments')):
            print(f'| 3. {a["sweep"]} sweep, {name} | {a[key]:.2f} s ({100*a[key]/a["seconds"]:.0f} %) | {b[key]:.2f} s ({100*b[key]/b["seconds"]:.0f} %) | {x(a[key], b[key])} |')
        print(f'| 3. {a["sweep"]} sweep, energy | {a["E"]:.10f} | {b["E"]:.10f} | {a["E"]-b["E"]:+.1e} |')
    print('\nPer layer (one product; work = matrix-product flops; sector = largest bond sector, in brackets the work-weighted mean):\n')
    print('| layer | spins | tensor | sector | work | CPU | H100 | ratio | CPU GFLOP/s | H100 GFLOP/s | solve CPU | solve H100 | ratio | products |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for a, b in zip(Lh, Lg):
        print(f'| {a["layer"]} | {int(round(np.log2(a["layer_states"])))} | {a["size"]:,} | {a["largest_sector"]} ({a["work_weighted_sector"]:.0f}) | {a["flops"]/1e9:.2f} GFLOP | {ms(a["product_seconds_min"])} | {ms(b["product_seconds_min"])} | '
              f'{x(a["product_seconds_min"], b["product_seconds_min"])} | {a["gflops"]:.0f} | {b["gflops"]:.0f} | {ms(a["solve_seconds_min"])} | {ms(b["solve_seconds_min"])} | {x(a["solve_seconds_min"], b["solve_seconds_min"])} | {a["solve_products"]} / {b["solve_products"]} |')
    parts = {}
    for b in Lg:
        for key, v in b.get('device_parts_per_product', {}).items():
            parts[key] = parts.get(key, 0.0) + v
    tot = sum(parts.values())
    if tot:
        print('\nH100, one product of every layer, by part: ' + ', '.join(f'{key} {1e3*v:.2f} ms ({100*v/tot:.0f} %)' for key, v in sorted(parts.items(), key=lambda kv: -kv[1])) + '.')
    th = {n: load(f'{d}/host_c{n}_{c}.json') for n in (1, 8, 32, 64)}
    if sum(v is not None for v in th.values()) > 1:
        print('\nCPU threads, one product of every layer, summed: ' + '; '.join(f'{n} threads {ms(v["sum_product_seconds"])} ({v["gflops_all_layers"]:.0f} GFLOP/s)' for n, v in th.items() if v) + '.')
    print()

# ---- corrected CPU baseline: host2.py (the Lanczos loop in matrix form), numpy BLAS on 1 thread, levels 2 and 3
rows = []
for c in cases:
    g = load(f'{d}/gpu3_{c}.json'); h = load(f'{d}/host_c{T}_{c}.json'); h2 = load(f'{d}/host2_b1_{c}.json')
    if not (g and h and h2):
        continue
    rows.append((c, g, h, h2))
if rows:
    print('## Corrected CPU baseline (host2.py: Lanczos orthogonalisation in matrix form, numpy BLAS on 1 thread; same files, same node)\n')
    print('| case | solve of every layer: CPU before / after / H100 | ratio after | expansion sweep: before / after / H100 | ratio after | closing sweep: before / after / H100 | ratio after |\n|---|---|---|---|---|---|---|')
    for c, g, h, h2 in rows:
        a, b = h['level3']['sweeps'], h2['level3']['sweeps']; gg = g['level3']['sweeps']
        print(f'| {c} | {h["sum_solve_seconds"]:.1f} / {h2["sum_solve_seconds"]:.1f} / {g["sum_solve_seconds"]:.2f} s | {x(h2["sum_solve_seconds"], g["sum_solve_seconds"])} | '
              f'{a[0]["seconds"]:.1f} / {b[0]["seconds"]:.1f} / {gg[0]["seconds"]:.2f} s | {x(b[0]["seconds"], gg[0]["seconds"])} | {a[1]["seconds"]:.1f} / {b[1]["seconds"]:.1f} / {gg[1]["seconds"]:.2f} s | {x(b[1]["seconds"], gg[1]["seconds"])} |')
    print()
    for c, g, h, h2 in rows:
        dd = [(abs(p['E']-q['E']), p['products'], q['products']) for p, q in zip(h['level3']['sweeps'], h2['level3']['sweeps'])]
        print(f'{c}: energy after the expansion / closing sweep, original minus corrected CPU engine: {dd[0][0]:.0e} / {dd[1][0]:.0e} (products {dd[0][1]}/{dd[0][2]}, {dd[1][1]}/{dd[1][2]}).')
    print('(Same arithmetic up to rounding; where a solve stops one step earlier or later the products differ by a few and the energies by up to 1e-7.)\n')
