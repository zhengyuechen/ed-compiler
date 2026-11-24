#!/usr/bin/env python3
"""The performance problem of the DMRG GPU benchmark: kagome YC6 cylinder with 7 rings, 60 spins, nearest-neighbour
Heisenberg model (J = 1), lowest state of S_z = 0 (momentum 0 around the cylinder; S_z = 1 with `nup=31`).

    bench60.py D [gpu|gpu1a|gpu1|gpu1t|gpu2|gpu3|gpu3d|gpu3s|host]      MEMHOOK=1: also the in-use peak of the cupy pool (not for timing) [nup] [K] [out.json] [tag]

Prints and stores every sweep (energy, seconds, where the time went) and the summary the benchmark asks for:
time of a full sweep at the final bond dimension, total time, time to reach each energy target, peak device memory
(memory pool of the process, and the device total from nvidia-smi if a sampler wrote one), energy against the reference."""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed

REF = -26.64456                                                        # block2, SU(2): -26.64454847 at 1200 multiplets, last digits extrapolated
TARGETS = {'1e-3': REF+1e-3, '3e-4': REF+3e-4, '1e-4': REF+1e-4}       # energy targets (absolute error of the total energy)
D = int(sys.argv[1]); where = sys.argv[2] if len(sys.argv) > 2 else 'gpu'; nup = int(sys.argv[3]) if len(sys.argv) > 3 else 30
K = int(sys.argv[4]) if len(sys.argv) > 4 else 0; outf = sys.argv[5] if len(sys.argv) > 5 else None; tag = sys.argv[6] if len(sys.argv) > 6 else ''
lat = ed.kagome_cylinder(7, 3); model = ed.Model(lat); assert lat.N == 60
if where in ('gpu', 'gpu1a', 'gpu1', 'gpu1t', 'gpu2', 'gpu3', 'gpu3d', 'gpu3s'):
    import cupy as cp, edc_dmrg_gpu as eg
    if where in ('gpu1t', 'gpu2'):
        import edc_dmrg_gpu2 as e2
    if where in ('gpu3', 'gpu3d'):
        import edc_dmrg_gpu3 as e3
    if where == 'gpu3s':
        import edc_dmrg_gpu3s as e3s
    cls = {'gpu': eg.DMRGGPU, 'gpu1a': eg.DMRGGPU1, 'gpu1': eg.DMRGGPU1b, 'gpu1t': e2.DMRGGPU1t if where == 'gpu1t' else None, 'gpu2': e2.DMRGGPU2 if where == 'gpu2' else None,
           'gpu3': e3.DMRGGPU3 if where == 'gpu3' else None, 'gpu3d': e3.DMRGGPU3d if where == 'gpu3d' else None, 'gpu3s': e3s.DMRGGPU3s if where == 'gpu3s' else None}[where]; version = cls.version; dev = cp.cuda.runtime.getDeviceProperties(0)['name'].decode(); where = 'gpu'
else:
    import edc_dmrg_gpu_stub as eg                                     # host run of the same schedule (no cupy needed)
    cls = ed.DMRG; version = 'host'; dev = 'cpu'
print(f'{lat.name}, N = {lat.N}, nup = {nup}, K = {K}, D = {D}, back end {version} {tag} on {dev}, labels (S_z, k), double precision', flush=True)
hook = None
if os.environ.get('MEMHOOK') == '1' and where == 'gpu':
    class Peak(cp.cuda.MemoryHook):
        name = 'peak_in_use'
        def __init__(self):
            self.cur = 0; self.peak = 0
        def malloc_postprocess(self, **kw):
            self.cur += kw['mem_size']; self.peak = max(self.peak, self.cur)
        def free_postprocess(self, **kw):
            self.cur -= kw['mem_size']
    hook = Peak(); hook.__enter__()
SEED = int(os.environ.get('SEED', '0'))                                  # SEED=n: another random stream (default 0); SWEEPS=n: only n sweeps at the full D
_prog = outf + '.sweeps' if outf else None                                 # every finished sweep is appended here at once, so a run that is
if _prog and os.path.exists(_prog):                                        # still going (or was cut off) already gives its times to each energy
    os.remove(_prog)


def _log(line):
    print(line, flush=True)
    if _prog:
        open(_prog, 'a').write(time.strftime('%H:%M:%S') + ' ' + line + '\n')


t0 = time.perf_counter(); E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=cls, log=_log, **({'seed': SEED} if SEED else {}), **({'sweeps': int(os.environ['SWEEPS'])} if os.environ.get('SWEEPS') else {})); total = time.perf_counter()-t0
if hook is not None:
    hook.__exit__(None, None, None)
full = [r for r in rows if r['D'] == D]; exp = [r['seconds'] for r in full if r['expansion'] > 0]; clo = [r['seconds'] for r in full if r['expansion'] == 0]
reach = {k: next((r['elapsed'] for r in rows if r['E'] <= v), None) for k, v in TARGETS.items()} if nup == 30 else {}
summ = dict(problem='kagome YC6, 7 rings, 60 spins, Heisenberg', nup=nup, K=K, D=D, version=version, tag=tag, device=dev, host_threads=dict(cores=os.environ.get('SLURM_CPUS_PER_TASK'), OMP=os.environ.get('OMP_NUM_THREADS'), OPENBLAS=os.environ.get('OPENBLAS_NUM_THREADS')), symmetries='U(1) S_z and momentum around the cylinder (Z_3)',
            precision='float64', seed=SEED, E=E, error=(E-REF if nup == 30 else None), total_seconds=total, sweep_seconds_expansion=float(np.mean(exp)) if exp else None,
            sweep_seconds_closing=float(np.mean(clo)) if clo else None, seconds_to_target=reach, products=eng.stats['products'], largest_bond=max(eng.bond_sizes()))
if where == 'gpu':
    summ['peak_pool_GiB'] = eng.peak/2**30; tot = {}
    for r in rows:
        for k, v in r['gpu'].items():
            tot[k] = tot.get(k, 0.0) + v
    detail = {k: tot.pop(k) for k in list(tot) if k.startswith('mv_')}  # parts of the moves (version 1 on): inside 'moves'
    edetail = {k: tot.pop(k) for k in list(tot) if k.startswith('env_')}  # parts of the environments (version 2): inside 'environments'
    tot['moves'] = sum(r['svd'] for r in rows); tot['environments'] = sum(r['env'] for r in rows)
    tot['other'] = total - sum(tot.values()); summ['seconds_by_part'] = tot
    if edetail:
        summ['environments_detail'] = edetail
    if hook is not None:
        summ['peak_in_use_GiB'] = hook.peak/2**30; summ['timing_not_comparable'] = 'memory hook active'
    if getattr(eng, 'threads_seen', None):
        summ['blas_threads_seen'] = eng.threads_seen
    if hasattr(eng, 'selector'):
        summ['selector'] = eng.selector.report(); eng.selector.save()
    if detail:
        summ['moves_detail'] = detail; summ['blocks_factored_on_device'] = eng.blocks_dev; summ['blocks_by_host_svd'] = eng.blocks_host
print('summary: ' + json.dumps(summ), flush=True)
if outf:
    json.dump(dict(summary=summ, sweeps=rows), open(outf, 'w'), indent=1)
