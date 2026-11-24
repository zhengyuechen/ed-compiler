#!/usr/bin/env python3
"""The CPU engine (edc_dmrg.DMRG) against the GPU back end v3 (edc_dmrg_gpu3.DMRGGPU3) at three levels, both fed the SAME
saved input.  The timing method follows Menczer et al. (arXiv:2407.07411), who report the cumulative time of the local
eigensolver and the matrix-product throughput in FLOP/s; here every level is timed on its own.

    levels.py make  D nup K state.npz pass.npz [gpu3|host] [sweeps at D before the state is saved, default 3]
    levels.py run   host|host2|gpu3 state.npz pass.npz out.json [levels, default 123]      host2: see host2.py; NUMPY_BLAS_THREADS=n
    levels.py check state.npz pass.npz out.json            (needs a GPU; device product against host product, one process)

make   runs the production schedule (two warm-up sweeps, then `sweeps` sweeps at D), saves the state (centre at layer 0)
       and writes the "pass file": the centre carried from layer 0 to the last layer by the host engine WITHOUT a solve and
       without truncation.  For every layer r it holds C_r (the centre tensor when the centre is at r), A_r (the
       left-orthonormal tensor it leaves behind) and the sector sizes of the bond to its right.
run    replays the pass in one engine: at every layer the tensors come from the file, the left environment is grown by the
       engine itself, the right one comes from reading the state.  So both engines hold the same tensors at every layer.
       level 1  one product H_eff v with v = C_r / |C_r|: repeated, minimum and median; and three numbers of the result
                (v.w, |w|, z.w with a fixed random z) so that two runs can be compared without storing vectors
       level 2  the layer solve of the production code from the start C_r (tolerance 3e-5, no random part), timed whole:
                set-up, products, orthogonalisation, the final combination; the tensor is put back afterwards
       level 3  from the state read again: one production sweep with expansion (1e-3, tolerance 3e-5) and one closing
                sweep (no expansion, tolerance 1e-6), with the engine's own split into solves, moves and environments
check  both engines in one process on the GPU node: |w_device - w_host| / |w_host| for every layer.

Matrix-product work of one product, counted from the sector sizes (2 m n k per product of an m x k with a k x n matrix):
    left   LH_i (D_i x D_i) M_i (D_i x C_i)   and   GL_i (D_i x W_i) S^L_i (W_i x C_i)
    right  U_j (R_j x D_j) RH_j (D_j x D_j)   and   S^R_j (R_j x W_j) GR_j (W_j x D_j)
The host library counts its own multiply-adds; both numbers are stored.  The slab moves (gathers) are not counted.
Lattice: LX rings of kagome YC(2 CC) (default LX=7, CC=3: the 60-spin benchmark problem)."""
import sys, os, json, time, platform
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import ctypes
import numpy as np, edc_dmrg as ed, edc_dmrg_state as st

LX, CC = int(os.environ.get('LX', 7)), int(os.environ.get('CC', 3))
TOL_EXP, TOL_CLOSE, ALPHA = 3e-5, 1e-6, 1e-3
_p = ed._p


def engine_class(where):
    if where == 'host':
        return ed.DMRG, None
    if where == 'host2':                                               # the host engine with the Lanczos loop of the GPU back end (host2.py)
        import host2
        n = int(os.environ.get('NUMPY_BLAS_THREADS', 0)); print('BLAS threads:', host2.numpy_blas_threads(n), flush=True)
        return host2.DMRGHost2, None
    import cupy as cp, edc_dmrg_gpu3 as e3
    return e3.DMRGGPU3, cp


def machine(cp):
    out = dict(host=platform.node(), threads=dict(EDC_DMRG=os.environ.get('EDC_DMRG_THREADS'), OMP=os.environ.get('OMP_NUM_THREADS'),
                                                  OPENBLAS=os.environ.get('OPENBLAS_NUM_THREADS')), cores_allowed=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
    try:
        out['cpu'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        out['cpu'] = platform.processor()
    if cp is not None:
        try:
            out['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        except Exception:
            out['device'] = 'stand-in'
    return out


def work(model, p, r):
    """Matrix-product flops of one product with the effective Hamiltonian of layer r, and the sizes that set them."""
    n = model.n; hasL = model.site_ls[r] >= 0 and r > 0; hasR = model.site_rs[r] >= 0 and r < n-1
    DL, C, WL, DR, R, WR = (a.astype(np.float64) for a in (p.DL, p.C, p.WL, p.DR, p.R, p.WR)); parts = {}
    parts['LH M'] = 2*float(np.sum(DL*DL*C)) if r > 0 else 0.0; parts['GL SL'] = 2*float(np.sum(DL*WL*C)) if hasL else 0.0
    parts['U RH'] = 2*float(np.sum(R*DR*DR)) if r < n-1 else 0.0; parts['SR GR'] = 2*float(np.sum(R*WR*DR)) if hasR else 0.0
    f = sum(parts.values()); wl = DL*DL*C; wr = R*DR*DR
    mean_block = float((np.sum(wl*DL) + np.sum(wr*DR))/max(1.0, np.sum(wl) + np.sum(wr)))    # bond-sector size, weighted by the work it carries
    return dict(flops=f, parts=parts, labels_left=int(np.sum(p.DL > 0)), labels_right=int(np.sum(p.DR > 0)), bond_left=int(p.DL.sum()), bond_right=int(p.DR.sum()),
                largest_sector=int(max(p.DL.max(), p.DR.max())), work_weighted_sector=mean_block, layer_states=int(p.c.S))


class Replay:
    """One engine carried along the pass file."""
    def __init__(self, where, state, passf):
        self.where = where; self.cls, self.cp = engine_class(where); self.xp = self.cp if self.cp is not None else np
        self.f = np.load(passf, allow_pickle=False); self.head = json.loads(str(self.f['head']))
        self.lat = ed.kagome_cylinder(self.head['LX'], self.head['CC']); self.model = ed.Model(self.lat); self.state = state
        t = time.perf_counter(); self.eng = st.load_state(state, self.model, cls=self.cls); self.sync(); self.t_load = time.perf_counter()-t
        self.eng.noise = 0.0; self.n = self.eng.n

    def sync(self):
        if self.cp is not None:
            self.cp.cuda.Device().synchronize()

    def tensor(self, key):
        a = np.ascontiguousarray(self.f[key], dtype=np.float64)
        return self.cp.asarray(a) if self.cp is not None else a

    def at(self, r):
        """Bring the engine to 'centre at layer r' (call with r = 0, 1, 2, ... in order)."""
        eng = self.eng; md = self.model
        if r > 0:
            q = r-1; eng.bond[r].dims[:] = self.f[f'd{r}']
            eng.plan[q] = ed.Plan(md, q, eng.bond[q], eng.bond[q+1]); eng.plan[r] = ed.Plan(md, r, eng.bond[r], eng.bond[r+1])
            eng.M[q] = self.tensor(f'A{q}'); eng.M[r] = self.tensor(f'C{r}'); assert eng.M[q].size == eng.plan[q].size and eng.M[r].size == eng.plan[r].size
            if hasattr(eng, '_dev'):
                eng._dev = None
            eng._grow_left(q)
        self.sync()

    def restore(self, r):
        self.eng.M[r] = self.tensor(f'C{r}')
        if hasattr(self.eng, '_dev'):
            self.eng._dev = None

    # ---- one product
    def product(self, r):
        """A function v -> H_eff v for layer r (host: numpy in, numpy out; device: cupy in, cupy out, synchronised)."""
        eng = self.eng; p = eng.plan[r]; n = p.size
        if self.cp is None:
            U = np.empty(n); oU = np.empty(n); a = (p.ref, _p(eng.LH[r]), _p(eng.GL[r]), _p(eng.RH[r+1]), _p(eng.GR[r+1]), _p(eng.Q[r]), _p(eng.P[r+1]))
            keep = (eng.LH[r], eng.GL[r], eng.RH[r+1], eng.GR[r+1], eng.Q[r], eng.P[r+1], U, oU)

            def mv(v, _keep=keep):
                out = np.zeros(n); eng.lib.heff(*a, _p(v), _p(out), _p(U), _p(oU))
                return out
            return mv, {}
        import edc_dmrg_gpu as eg
        prof = {}; H = eg.Product(eng, r, prof)
        return H.apply, prof

    def level1(self, r, min_reps, min_seconds, max_reps):
        eng = self.eng; xp = self.xp; p = eng.plan[r]; n = p.size; row = dict(layer=r, size=n, **work(self.model, p, r))
        t = time.perf_counter(); mv, prof = self.product(r); self.sync(); row['setup_seconds'] = time.perf_counter()-t
        v = eng.M[r]/xp.linalg.norm(eng.M[r]); w = mv(v); self.sync()                       # first call: not timed
        if self.cp is None:                                                                  # the library's own count of one product
            c = (ctypes.c_double*2)(); eng.lib.counts(c, 1); w = mv(v); eng.lib.counts(c, 0); row['host_dense_multiply_adds'] = c[0]; row['host_slab_multiply_adds'] = c[1]
        for k in list(prof):
            prof[k] = 0.0
        ts = []; t0 = time.perf_counter()
        while len(ts) < max_reps and (len(ts) < min_reps or time.perf_counter()-t0 < min_seconds):
            t = time.perf_counter(); w = mv(v); self.sync(); ts.append(time.perf_counter()-t)
        z = np.random.default_rng(1000+r).standard_normal(n); wh = w.get() if hasattr(w, 'get') else w; vh = v.get() if hasattr(v, 'get') else v
        row.update(product_seconds_min=min(ts), product_seconds_median=float(np.median(ts)), reps=len(ts), vHv=float(vh @ wh), norm_Hv=float(np.linalg.norm(wh)), zHv=float(z @ wh),
                   gflops=row['flops']/min(ts)/1e9)
        if prof:
            row['device_parts_per_product'] = {k: prof[k]/len(ts) for k in prof}
        return row, wh

    # ---- one layer solve
    def level2(self, r, reps=2):
        eng = self.eng; ts = []; out = {}
        for _ in range(reps):
            self.restore(r); self.sync(); p0 = eng.stats['products']; before = dict(getattr(eng, 'prof', {}))
            t = time.perf_counter(); E = eng.solve(r, TOL_EXP); self.sync(); ts.append(time.perf_counter()-t)
            out = dict(solve_E=E, solve_products=eng.stats['products']-p0)
            if hasattr(eng, 'prof'):
                out['solve_device_parts'] = {k: eng.prof[k]-before.get(k, 0.0) for k in eng.prof if eng.prof[k]-before.get(k, 0.0) > 0}
        self.restore(r); out.update(solve_seconds_min=min(ts), solve_seconds_all=ts)
        return out

    # ---- sweeps from the state
    def level3(self):
        t = time.perf_counter(); eng = st.load_state(self.state, self.model, cls=self.cls); self.sync(); tl = time.perf_counter()-t; D = self.head['D']; rows = []
        for name, al, tol in (('expansion', ALPHA, TOL_EXP), ('closing', 0.0, TOL_CLOSE)):
            s0 = {k: eng.stats[k] for k in ('products', 'solves', 't_products', 't_svd', 't_env')}; before = dict(getattr(eng, 'prof', {}))
            t = time.perf_counter(); E = eng.sweep(D, al, tol); self.sync(); dt = time.perf_counter()-t
            row = dict(sweep=name, expansion=al, tolerance=tol, E=E, seconds=dt, products=eng.stats['products']-s0['products'], solves=eng.stats['solves']-s0['solves'],
                       solve=eng.stats['t_products']-s0['t_products'], moves=eng.stats['t_svd']-s0['t_svd'], environments=eng.stats['t_env']-s0['t_env'], largest_bond=max(eng.bond_sizes()))
            if hasattr(eng, 'prof'):
                row['device_parts'] = {k: eng.prof[k]-before.get(k, 0.0) for k in eng.prof}
            rows.append(row); print(f'  level 3, {name} sweep: E = {E:.10f} | {dt:8.2f} s | solve {row["solve"]:.2f}, moves {row["moves"]:.2f}, environments {row["environments"]:.2f} | '
                                    f'products {row["products"]}', flush=True)
        return dict(load_seconds=tl, sweeps=rows)


def make(D, nup, K, state, passf, where='gpu3', nsw=3):
    lat = ed.kagome_cylinder(LX, CC); model = ed.Model(lat); cls, cp = engine_class(where)
    if cp is None:
        import edc_dmrg_gpu_stub as eg
    else:
        import edc_dmrg_gpu as eg
    print(f'{lat.name}, N = {lat.N}, nup = {nup}, K = {K}, D = {D}: two warm-up sweeps and {nsw} at D with {getattr(cls, "version", "host")}', flush=True)
    t = time.perf_counter(); E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=cls, sweeps=nsw, log=print); trun = time.perf_counter()-t
    head = st.save_state(eng, state, D=D, sweeps_at_D=nsw, note='levels.py make'); del eng
    h = st.load_state(state, model, cls=ed.DMRG); n = h.n; arr = {}; t = time.perf_counter()
    for r in range(n):
        arr[f'C{r}'] = np.array(h.M[r], dtype=np.float64)
        if r < n-1:
            h.move_right(r, 10**9, 0.0); arr[f'A{r}'] = np.array(h.M[r], dtype=np.float64); arr[f'd{r+1}'] = h.bond[r+1].dims.astype(np.int32)
    E_end, nrm, res = st.energy(h, n-1)
    ph = dict(LX=LX, CC=CC, N=int(lat.N), D=D, nup=nup, K=K, sweeps_at_D=nsw, layers=n, E_state=head['energy'], E_after_pass=E_end, made_with=getattr(cls, 'version', 'host'),
              bond_sizes_state=head['bond_sizes'], bond_sizes_pass=[int(b.dims.sum()) for b in h.bond], run_seconds=trun, pass_seconds=time.perf_counter()-t,
              last_sweep_products=rows[-1]['products'], last_sweep_seconds=rows[-1]['seconds'])
    np.savez(passf, head=np.array(json.dumps(ph)), **arr)
    print(f'state: E = {head["energy"]:.12f}, bonds {head["bond_sizes"]}\npass:  E = {E_end:.12f} at the last layer (difference {E_end-head["energy"]:+.1e}), bonds {ph["bond_sizes_pass"]}; '
          f'{os.path.getsize(passf)/2**20:.0f} MiB', flush=True)


def run(where, state, passf, outf, levels='123'):
    R = Replay(where, state, passf); n = R.n; out = dict(engine=getattr(R.cls, 'version', 'host'), machine=machine(R.cp), problem=R.head, load_seconds=R.t_load, layers=[])
    print(f'{R.lat.name}, D = {R.head["D"]}, nup = {R.head["nup"]}: {out["engine"]} on {out["machine"].get("device", out["machine"]["cpu"])}, threads {out["machine"]["threads"]}; '
          f'state read (environments rebuilt) in {R.t_load:.2f} s', flush=True)
    gpu = R.cp is not None
    if gpu and ('1' in levels or '2' in levels):                                             # kernels compiled and libraries loaded before any clock starts
        R.eng.solve(0, TOL_EXP, maxit=2); R.restore(0)
    if '1' in levels or '2' in levels:
        for r in range(n):
            R.at(r); row = dict(layer=r)
            if '1' in levels:
                row, _ = R.level1(r, *((5, 0.3, 50) if gpu else (3, 1.0, 15)))
            if '2' in levels:
                row.update(R.level2(r))
            out['layers'].append(row)
            print(f'  layer {r:2d}: tensor {R.eng.plan[r].size:9d}' + (f' | product {row["product_seconds_min"]*1e3:9.3f} ms ({row["gflops"]:8.2f} GFLOP/s, {row["reps"]} reps)' if '1' in levels else '')
                  + (f' | solve {row["solve_seconds_min"]*1e3:9.2f} ms, {row["solve_products"]} products, E = {row["solve_E"]:.10f}' if '2' in levels else ''), flush=True)
        L = out['layers']
        if '1' in levels:
            out['sum_product_seconds'] = sum(x['product_seconds_min'] for x in L); out['sum_flops'] = sum(x['flops'] for x in L); out['gflops_all_layers'] = out['sum_flops']/out['sum_product_seconds']/1e9
            print(f'  one product of every layer: {out["sum_product_seconds"]*1e3:.2f} ms, {out["gflops_all_layers"]:.2f} GFLOP/s', flush=True)
        if '2' in levels:
            out['sum_solve_seconds'] = sum(x['solve_seconds_min'] for x in L); out['sum_solve_products'] = sum(x['solve_products'] for x in L)
            print(f'  the solve of every layer: {out["sum_solve_seconds"]:.3f} s, {out["sum_solve_products"]} products', flush=True)
    if '3' in levels:
        if gpu and not ('1' in levels or '2' in levels):
            R.eng.solve(0, TOL_EXP, maxit=2)
        del R.eng; out['level3'] = R.level3()
    json.dump(out, open(outf, 'w'), indent=1)


def check(state, passf, outf):
    H = Replay('host', state, passf); G = Replay('gpu3', state, passf); rows = []; worst = 0.0
    for r in range(H.n):
        H.at(r); G.at(r); a, wh = H.level1(r, 1, 0.0, 1); b, wg = G.level1(r, 1, 0.0, 1); d = float(np.linalg.norm(wg-wh)/np.linalg.norm(wh)); worst = max(worst, d)
        rows.append(dict(layer=r, size=a['size'], relative_difference=d, largest_difference=float(np.abs(wg-wh).max()), vHv_host=a['vHv'], vHv_device=b['vHv'], zHv_host=a['zHv'], zHv_device=b['zHv']))
        print(f'  layer {r:2d}: tensor {a["size"]:9d} | |w_device - w_host| / |w_host| = {d:.2e} | v.Hv host {a["vHv"]:.12f}, device {b["vHv"]:.12f}', flush=True)
    print(f'largest relative difference over the layers: {worst:.2e}', flush=True)
    json.dump(dict(problem=H.head, machine=machine(G.cp), worst=worst, layers=rows), open(outf, 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'make':
        make(int(a[1]), int(a[2]), int(a[3]), a[4], a[5], *(a[6:7]), *(int(x) for x in a[7:8]))
    elif a[0] == 'run':
        run(*a[1:6])
    elif a[0] == 'check':
        check(*a[1:4])
    else:
        sys.exit(__doc__)
