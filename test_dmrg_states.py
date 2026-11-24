#!/usr/bin/env python3
"""States of the layer DMRG (edc_dmrg_state, edc_dmrg_excited, edc_dmrg_vector) on a kagome cylinder small enough for
exact diagonalisation: YC6, two rings and a last layer, 18 spins, S_z = 0, momentum 0.  CPU engine, no GPU needed.

1. A state is saved and read back: energy, norm, <S_z> of every layer and the tensors themselves must be unchanged, a
   sweep continued from the file must give what the original engine gives, and a file for another sector is refused.
2. Four states of the same (S_z, K) sector, each found as the lowest state of H + w sum |earlier><earlier|, are written
   as vectors on the spins and measured with the Hamiltonian of the bond list (independent of the DMRG environments):
   each must sit on its own exact level (levels 2 and 3 are 0.013 apart), with the energy, the weight on that level and
   the overlaps with the earlier states inside the limits below.  The same at a truncating bond dimension."""
import os, tempfile
import numpy as np
import edc_dmrg as ed, edc_dmrg_state as st, edc_dmrg_excited as ex, edc_dmrg_vector as vec

lat = ed.kagome_cylinder(2, 3, tail=True); model = ed.Model(lat); N = lat.N; nup, K = N//2, 0; tol = 3e-5


def schedule(eng, D):
    for Dw, al, tl, nz in [(max(min(D, 64), D//4), 1e-2, tol, 1e-2), (max(min(D, 64), D//2), 1e-2, tol, 1e-2)] + [(D, al, tol if al > 0 else tol/30, 0.0) for al in ed.ALPHAS + [0.0]]:
        E = eng.sweep(Dw, al, tl, noise=nz)
    return E


# ---- 1. save and read back
eng = ed.DMRG(model, nup, K, seed=0); E = schedule(eng, 128); d = tempfile.mkdtemp(prefix='edc_states_'); f = os.path.join(d, 'state.npz')
head = st.save_state(eng, f, D=128); sz = np.array(st.layer_sz(eng)); back = st.load_state(f, model); E2, norm, res = st.energy(back, back.loaded['centre'])
print(f'saved: centre {head["centre"]}, E = {head["energy"]:.12f}; read back: E = {E2:.12f}, norm {norm:.15f}; layer S_z ' + ' '.join(f'{x:+.4f}' for x in sz))
assert head['centre'] == 0 and abs(head['energy']-E) < 1e-10 and abs(E2-head['energy']) < 1e-12 and abs(norm-1) < 1e-12
assert np.abs(np.array(st.layer_sz(back))-sz).max() < 1e-12 and abs(sz.sum()-(nup-N/2)) < 1e-10
assert all(np.array_equal(a, b) for a, b in zip(eng.M, back.M)) and all(np.array_equal(a.dims, b.dims) for a, b in zip(eng.bond, back.bond))
Ea, Eb = eng.sweep(128, 0.0, 1e-6), back.sweep(128, 0.0, 1e-6); print(f'one more sweep: original {Ea:.12f}, from the file {Eb:.12f}'); assert abs(Ea-Eb) < 1e-10
try:
    ex.excited(ed.DMRG)(model, nup+1, K).orthogonal_to(f, 4.0); raise SystemExit('a state of another sector was accepted')
except AssertionError:
    print('a file of another sector is refused')

# ---- 2. four states of one sector against the exact levels
w, V, H, proj = vec.exact_states(lat, nup, K, k=7); print('exact levels: ' + ' '.join(f'{x:.10f}' for x in w)); cls = ex.excited(ed.DMRG)
for D, dE, dW, dO in ((512, 1e-9, 1e-6, 1e-5), (128, 2e-4, 1e-4, 1e-4)):          # D, allowed energy error, missing weight, overlap
    engs = []; vs = []
    for p in range(4):
        e = cls(model, nup, K, seed=p)
        for q in engs:
            e.orthogonal_to(q, 4.0)
        schedule(e, D); v = vec.to_vector(e); Ev, res = vec.measure(v, H); we = np.abs(V.conj() @ v)**2; k = int(np.argmax(we)); ov = [abs(np.vdot(u, v)) for u in vs]
        own = [abs(x) for x in (e.last or {}).get('overlaps', [])]
        print(f'D = {D}, state {p}: E = {Ev:.10f}, exact level {k} {w[k]:.10f} (difference {Ev-w[k]:+.1e}), weight on it {we[k]:.8f}, whole-state residual {res:.1e}, '
              f'overlaps with earlier states ' + (' '.join(f'{x:.1e}' for x in ov) or '-'))
        assert k == p, 'the states do not come in the order of the exact levels'
        assert -1e-9 < Ev - w[k] < dE and 1 - we[k] < dW and all(x < dO for x in ov)
        assert all(abs(a-b) < 1e-9 for a, b in zip(ov, own)), "the engine's own overlaps differ from those of the vectors"
        assert p == 0 or abs(e.last['E'] - Ev) < 1e-9, "the engine's energy differs from <psi|H|psi> of the vector"
        engs.append(e); vs.append(v)
print('ok')
