#!/usr/bin/env python3
"""Save a state of the layer DMRG, read it back, compare.  Runs with the host engine (no GPU) or a
GPU back end; with the numpy stand-in for cupy it checks the device path without a GPU.

    check_state.py [host|gpu1|gpu2|gpu3|gpu3r] [Lx c D nup K] [dir for the files]

1. run DMRG, save (centre at layer 0), read back into a fresh engine: energy, norm, labels and sizes of every bond,
   <S_z> of every layer, every tensor (bit for bit), and the environments through one more product;
2. the read-back engine continues: one closing sweep from the file gives the energy of one more sweep of the original;
3. the same state with the centre moved to the last layer (as after bench/kagome_gap/profiles.py), saved and read;
4. a file of a GPU engine read by the host engine and back (the file does not depend on the back end)."""
import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed, edc_dmrg_state as st

w = sys.argv[1] if len(sys.argv) > 1 else 'host'; Lx, c, D, nup, K = (int(v) for v in sys.argv[2:7]) if len(sys.argv) > 6 else (4, 3, 200, 17, 0)
d = sys.argv[7] if len(sys.argv) > 7 else tempfile.mkdtemp(prefix='edc_state_'); lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat)
if w == 'host':
    import edc_dmrg_gpu_stub as eg; cls = ed.DMRG
else:
    import edc_dmrg_gpu as eg
    if w == 'gpu1': cls = eg.DMRGGPU1b
    elif w == 'gpu2': import edc_dmrg_gpu2 as e2; cls = e2.DMRGGPU2
    elif w == 'gpu3': import edc_dmrg_gpu3 as e3; cls = e3.DMRGGPU3
    else: import edc_dmrg_gpu3r as e3r; cls = e3r.DMRGGPU3r
host = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a); ok = True


def report(name, value, limit):
    global ok
    good = value <= limit; ok = ok and good; print(f'  {name}: {value:.2e}  ({"ok" if good else "FAILED"}, limit {limit:.0e})', flush=True)


print(f'{lat.name}, N = {lat.N}, nup = {nup}, K = {K}, D = {D}, engine {getattr(cls, "version", "host")}')
E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=cls, log=None); f0 = os.path.join(d, f'state_{w}.npz')
t = time.perf_counter(); head = st.save_state(eng, f0, D=D, note='check_state'); ts = time.perf_counter()-t; sz0 = np.array(st.layer_sz(eng))
print(f'1. saved: centre {head["centre"]}, E = {head["energy"]:.12f} (run: {E:.12f}), norm {head["norm"]:.15f}, centre residual {head["centre_residual"]:.1e}, '
      f'{os.path.getsize(f0)/2**20:.1f} MiB in {ts:.2f} s; sum of layer S_z {sz0.sum():+.6f} (sector {nup-lat.N/2:+.1f})')
t = time.perf_counter(); e2_ = st.load_state(f0, model, cls=cls); tl = time.perf_counter()-t; E2, n2, r2 = st.energy(e2_, e2_.loaded['centre']); sz2 = np.array(st.layer_sz(e2_))
print(f'   read back in {tl:.2f} s: E = {E2:.12f}, norm {n2:.15f}')
report('energy, read back - saved', abs(E2-head['energy']), 1e-11); report('energy, saved - run', abs(head['energy']-E), 1e-9)
report('norm - 1', abs(n2-1), 1e-12); report('layer S_z, largest difference', float(np.abs(sz2-sz0).max()), 1e-12)
report('tensors, largest difference', max(float(np.abs(host(a)-host(b)).max()) for a, b in zip(eng.M, e2_.M)), 0.0)
report('bond labels and sizes differ', float(sum(not (np.array_equal(a.m, b.m) and np.array_equal(a.k, b.k) and np.array_equal(a.dims, b.dims)) for a, b in zip(eng.bond, e2_.bond))), 0.0)
print('   layer S_z: ' + ' '.join(f'{x:+.4f}' for x in sz2))
Ea = eng.sweep(D, 0.0, 1e-6); Eb = e2_.sweep(D, 0.0, 1e-6)
print(f'2. one more closing sweep: original {Ea:.12f}, from the file {Eb:.12f}'); report('difference', abs(Ea-Eb), 1e-9)
for r in range(eng.n-1):                                               # 3. centre to the last layer without truncation
    if hasattr(eng, '_dev'):
        eng.solve(r, 1e-6, maxit=1)                                    # v1 ... v3 move from the device product of a solve; one step leaves the tensor
    eng.move_right(r, 10**9, 0.0)
f1 = os.path.join(d, f'state_{w}_end.npz'); h1 = st.save_state(eng, f1); e3_ = st.load_state(f1, model, cls=cls); E3, n3, r3 = st.energy(e3_, e3_.loaded['centre'])
print(f'3. centre moved to layer {h1["centre"]}: saved E = {h1["energy"]:.12f}; read back E = {E3:.12f}')
report('energy, read back - saved', abs(E3-h1['energy']), 1e-11); report('energy against the sweep', abs(h1['energy']-Ea), 1e-9)
report('layer S_z against the state with the centre at layer 0', float(np.abs(np.array(st.layer_sz(e3_)) - np.array(st.layer_sz(e2_))).max()), 1e-9)
if w != 'host':
    e4 = st.load_state(f0, model, cls=ed.DMRG); E4, n4, r4 = st.energy(e4, 0); print(f'4. the file of {cls.version} read by the host engine: E = {E4:.12f}')
    report('energy, host engine - saved', abs(E4-head['energy']), 1e-10)
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
