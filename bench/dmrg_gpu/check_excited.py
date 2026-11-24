#!/usr/bin/env python3
"""Several states of one (S_z, K) sector with the layer DMRG, state by state against exact diagonalisation on a small
cylinder.  Host engine by default (no GPU needed); gpu3, gpu3r, gpu3g or gpu3gr for a device back end.

    check_excited.py Lx c tail(0|1) nup K D [states=3] [weight=4] [host|gpu3|gpu3r|gpu3g|gpu3gr] [out.json]

State 0 is ordinary DMRG; state p is DMRG for H + w sum_{q<p} |psi_q><psi_q| (edc_dmrg_excited), each from its own
random start.  Every state is then written as a vector on the spins (edc_dmrg_vector) and measured with the Hamiltonian
of the bond list, independently of the DMRG environments:
    E             <psi|H|psi> from the vector            (and the engine's own value E_penalised - w sum overlap^2)
    residual      |H psi - E psi| of the whole state
    local         residual of the centre tensor in its last layer solve (for H', the operator DMRG minimises)
    overlaps      with the states found before (from the vectors, and the engine's own contraction)
    exact         the exact level it belongs to and the weight |<exact_k|psi>|^2 on each of the lowest levels:
                  a state that mixes two close levels shows weight on both."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed, edc_dmrg_state as st, edc_dmrg_vector as vec, edc_dmrg_excited as ex

Lx, c, tail, nup, K, D = (int(v) for v in sys.argv[1:7]); ns = int(sys.argv[7]) if len(sys.argv) > 7 else 3; wt = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
where = sys.argv[9] if len(sys.argv) > 9 else 'host'; outf = sys.argv[10] if len(sys.argv) > 10 else None
lat = ed.kagome_cylinder(Lx, c, tail=bool(tail)); model = ed.Model(lat); N = lat.N
if where == 'host':
    base = ed.DMRG
elif where == 'gpu3':
    import edc_dmrg_gpu3 as e3; base = e3.DMRGGPU3
elif where == 'gpu3r':
    import edc_dmrg_gpu3r as e3r; base = e3r.DMRGGPU3r
elif where == 'gpu3g':
    from edc_dmrg_gpu3g import DMRGGPU3g as base
elif where == 'gpu3gr':
    from edc_dmrg_gpu3gr import DMRGGPU3gr as base
else:
    raise SystemExit(f'unknown engine {where}')
cls = ex.excited(base); tol = 3e-5
t = time.time(); w, V, H, proj = vec.exact_states(lat, nup, K, k=ns+3)
print(f'{lat.name}, N = {N}, S_z = {nup-N/2:+.1f}, K = {K}, D = {D}, weight {wt}, engine {cls.version}')
print('exact levels: ' + ' '.join(f'{x:.10f}' for x in w) + f'   ({time.time()-t:.0f} s)'); print('spacings:     ' + ' '.join(f'{x:.6f}' for x in np.diff(w)), flush=True)
vs = []; engs = []; out = dict(lattice=lat.name, N=N, nup=nup, K=K, D=D, weight=wt, engine=cls.version, exact=w.tolist(), states=[])
for p in range(ns):
    t = time.time(); eng = cls(model, nup, K, seed=p)
    for q in engs:
        eng.orthogonal_to(q, wt)
    sched = [(max(min(D, 64), D//4), 1e-2, tol, 1e-2), (max(min(D, 64), D//2), 1e-2, tol, 1e-2)] + [(D, al, tol if al > 0 else tol/30, 0.0) for al in ed.ALPHAS + [0.0, 0.0]]
    hist = [eng.sweep(Dw, al, tl, noise=nz) for Dw, al, tl, nz in sched]
    last = eng.last or dict(E=hist[-1], E_penalised=hist[-1], overlaps=[])
    v = vec.to_vector(eng); E, res = vec.measure(v, H); ov = [abs(np.vdot(u, v)) for u in vs]; we = np.abs(V.conj() @ v)**2; k = int(np.argmax(we))
    pens = [(s.w, st._host(s.phi(eng, 0))) for s in eng.prev]; E0, nrm, r0 = st.energy(eng, 0); x = np.asarray(st._host(eng.M[0])); x = x/np.linalg.norm(x)
    # local residual for H' = Heff + sum w phi phi^T at the centre: st.energy gives Heff x - E_H x; add the penalty part
    import edc_dmrg_gpu_stub  # noqa (keeps the host schedule importable on machines without cupy)
    loc = None
    if where == 'host':
        pl = eng.plan[0]; n = pl.size; hx = np.zeros(n); U = np.empty(n); oU = np.empty(n); _p = ed._p
        eng.lib.heff(pl.ref, _p(eng.LH[0]), _p(eng.GL[0]), _p(eng.RH[1]), _p(eng.GR[1]), _p(eng.Q[0]), _p(eng.P[1]), _p(x), _p(hx), _p(U), _p(oU))
        for ww, phi in pens:
            hx += ww*float(phi @ x)*phi
        loc = float(np.linalg.norm(hx - float(x @ hx)*x))
    print(f'state {p}: E = {E:.10f} (engine {last["E"]:.10f}), exact level {k}: {w[k]:.10f}, E - exact = {E-w[k]:+.1e}; whole-state residual {res:.1e}; '
          f'local residual {"%.1e" % loc if loc is not None else "n/a"}; {time.time()-t:.0f} s')
    print(f'         weight on the exact levels: ' + ' '.join(f'{x:.6f}' for x in we) + f'   (sum {we.sum():.6f})')
    if vs:
        print(f'         overlaps with the earlier states: vectors ' + ' '.join(f'{x:.1e}' for x in ov) + '; engine ' + ' '.join(f'{abs(x):.1e}' for x in last['overlaps']))
    sys.stdout.flush()
    out['states'].append(dict(E=E, E_engine=last['E'], E_penalised=last['E_penalised'], exact_level=k, residual=res, local_residual=loc, overlaps=ov, overlaps_engine=[abs(x) for x in last['overlaps']],
                              weights=we.tolist(), bond=max(eng.bond_sizes()), sweep_values=hist, seconds=time.time()-t))
    vs.append(v); engs.append(eng)
G = np.abs(np.array([[np.vdot(a, b) for b in vs] for a in vs])); np.fill_diagonal(G, 0.0); print(f'largest overlap between two of the states: {G.max():.1e}')
ok = all(abs(w[s['exact_level']] - w[i]) < 1e-9 for i, s in enumerate(out['states'])); print('exact levels in order (including degeneracies): ' + ('yes' if ok else 'NO: ' + str([s['exact_level'] for s in out['states']])))
if outf:
    json.dump(out, open(outf, 'w'), indent=1)
