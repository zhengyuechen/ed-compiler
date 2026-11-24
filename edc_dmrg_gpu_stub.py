"""The sweep schedule of edc_dmrg_gpu.run for a host-only run (no cupy on the machine)."""
import time
import edc_dmrg as ed


def run(model, D, nup=None, K=0, cls=ed.DMRG, sweeps=None, tol=3e-5, seed=0, log=print):
    nup = model.lat.N//2 if nup is None else nup; eng = cls(model, nup, K, seed=seed); alphas = list(ed.ALPHAS); rows = []
    alphas = alphas[:sweeps] if sweeps and sweeps <= len(alphas) else alphas + [0.0]*((sweeps or 0)-len(alphas)); t0 = time.perf_counter()
    sched = [(max(min(D, 64), D//4), 1e-2, tol, 1e-2), (max(min(D, 64), D//2), 1e-2, tol, 1e-2)] + [(D, al, tol if al > 0 else tol/30, 0.0) for al in alphas]
    for sw, (Dw, al, tl, noise) in enumerate(sched):
        ts = time.perf_counter(); p0 = eng.stats['products']; s0 = {k: eng.stats[k] for k in ('t_products', 't_svd', 't_env')}
        E = eng.sweep(Dw, al, tl, noise=noise); te = time.perf_counter()
        rows.append(dict(sweep=sw+1, D=Dw, expansion=al, E=E, seconds=te-ts, elapsed=te-t0, products=eng.stats['products']-p0, largest_bond=max(eng.bond_sizes()),
                         solve=eng.stats['t_products']-s0['t_products'], svd=eng.stats['t_svd']-s0['t_svd'], env=eng.stats['t_env']-s0['t_env'], gpu={}))
        if log:
            log(f'  sweep {sw+1:2d} D={Dw:5d} exp={al:.0e}: E = {E:.10f} | {te-ts:7.2f} s (elapsed {te-t0:8.2f}) | products {rows[-1]["products"]:5d}')
    return E, eng, rows
