#!/usr/bin/env python3
"""edc_dmrg_combine against full vectors (24 spins, the sector with two levels 0.00067 apart): four short DMRG runs that
mix the two levels; their overlap and Hamiltonian matrices from sums of the saved states must equal those of the
vectors on the spins; Ritz levels and the lowest Ritz state.    ritz_check.py [host|gpu] [dir]"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed, edc_dmrg_state as st, edc_dmrg_excited as ex, edc_dmrg_vector as vec, edc_dmrg_combine as cb
where = sys.argv[1] if len(sys.argv) > 1 else 'host'; d = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix='edc_ritz_')
if where == 'host':
    base = ed.DMRG
else:
    import edc_dmrg_gpu3r as e3r; base = e3r.DMRGGPU3r
lat = ed.kagome_cylinder(3, 3); model = ed.Model(lat); nup, K = 13, 1; cls = ex.excited(base)      # the sector with two levels 0.00067 apart
w, V, H, proj = vec.exact_states(lat, nup, K, k=5); print('exact levels', np.round(w, 8))
def run(seed, D, orth=(), sweeps=6):
    e = cls(model, nup, K, seed=seed)
    for q in orth: e.orthogonal_to(q, 4.0)
    for Dw, al, tl, nz in [(D//4, 1e-2, 3e-5, 1e-2), (D//2, 1e-2, 3e-5, 1e-2)] + [(D, a, 3e-5 if a > 0 else 1e-6, 0.0) for a in ed.ALPHAS[:sweeps]]:
        e.sweep(Dw, al, tl, noise=nz)
    return e
files = []; vs = []
for name, seed, D, sw in (('a', 0, 80, 3), ('b', 1, 80, 2), ('c', 2, 60, 4), ('d', 3, 100, 2)):     # deliberately short runs: mixtures of the two close levels
    e = run(seed, D, sweeps=sw); f = os.path.join(d, name + '.npz'); st.save_state(e, f); files.append(f); v = vec.to_vector(e); vs.append(v)
    we = np.abs(V.conj() @ v)**2; print(f'state {name}: E = {vec.measure(v, H)[0]:.8f}, weights on the exact levels ' + ' '.join(f'{x:.4f}' for x in we))
R = cb.subspace(model, files, cls=base, log=print)
Sv = np.array([[np.vdot(a, b) for b in vs] for a in vs]); Hv = np.array([[np.vdot(a, H @ b) for b in vs] for a in vs])
print('largest |imag| of the vector matrices', abs(Sv.imag).max(), abs(Hv.imag).max())
print('S: largest difference to the vectors', abs(np.array(R['S'])-Sv.real).max(), '; H:', abs(np.array(R['H'])-Hv.real).max(), '; sum-rule checks', R['checks'])
print('Ritz levels', np.round(R['levels'][:2], 8)); print('exact      ', np.round(w[:2], 8))
assert abs(np.array(R['S'])-Sv.real).max() < 1e-10 and abs(np.array(R['H'])-Hv.real).max() < 1e-9 and R['checks'] < 1e-9
eng = base(model, nup, K); E0 = cb.ritz_state(eng, files, [r[0] for r in R['vectors']]); v0 = vec.to_vector(eng); we = np.abs(V.conj() @ v0)**2
print(f'lowest Ritz state as an MPS: E = {E0:.8f} (Ritz {R["levels"][0]:.8f}), weights on the exact levels ' + ' '.join(f'{x:.4f}' for x in we), 'largest bond', max(eng.bond_sizes()))
assert abs(E0 - R['levels'][0]) < 1e-9 and R['levels'][0] > w[0] - 1e-9
print('ok')
