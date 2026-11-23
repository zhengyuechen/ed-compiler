#!/usr/bin/env python3
"""Correlations in the lowest state of the zero-momentum sector of a kagome cluster too large for the full S_z basis
(36 sites), with the symmetric engine and translations only.

    corr_sym.py <out.npz> --cluster 2,2,-2,4 --nup 20 [--J2 0] [--device gpu] [--res 1e-7]

What translations alone allow (bench/corr_tools.py for the definitions):
  S^z_i and S^z_i S^z_j on bonds are diagonal in the S_z basis: their full matrices Gz, Gzz (every momentum) follow from
    the weights |psi|^2 of the representatives.
  S_i.S_j on bonds and the chirality change the state; only sums over all translations of one bond type (triangle
    kind) act inside the sector.  That gives the zero-momentum blocks:  Gb0[u, v] = <H_u H_v> / Nc for the six bond
    types u = (triangle kind, edge), eb0[u] = <H_u> / Nc, and Gx0[k, l] = <X_k X_l> / Nc for the two triangle kinds,
    with X_k = i [H_(k,0), H_(k,1)] the sum of the chiralities of the triangles of kind k.
The state must be the only lowest level of the sector (a fully symmetric state); 'asym' records how far the measured
matrices are from invariance under the point group."""
import sys, os, json, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import edc, edc_lattices as L, edc_sub, edc_hubbard_sym as hs
import corr_tools as ct


def bond_types(cl, geo):
    """Type (triangle kind, edge) = 3 kind + edge of every bond, checked to be constant along translations."""
    typ = np.array([3*geo['tri_kind'][a//3]+a % 3 for a in range(len(geo['bonds']))])
    sp = ct.OpSpace(cl, 'b', geo)
    for (perm, ij, name), img in zip(sp.group, sp.image):
        if name == 'E':
            assert np.array_equal(typ[img], typ), 'bond types are not translation classes'
    return typ


def run(cluster, nup, J2=0.0, device='gpu', rtol=1e-9, maxit=4000, chunk=1 << 20, res=1e-7, log=print):
    A = [int(x) for x in cluster.split(',')]
    cl = L.LatticeCluster('kagome', (A[0], A[1]), (A[2], A[3]))
    N, Nc = cl.N, cl.Nc
    geo = ct.geometry(cl)
    bonds, tris = geo['bonds'], geo['tris']
    typ = bond_types(cl, geo)
    t0 = time.perf_counter()
    e = edc_sub.SubEngine(cl.heisenberg(1.0, J2), nup, cl, (0.0, 0.0), 'X0', 0, device=device, point_ops=['E'], store_states=True)
    assert e.real and e.store
    xp = e.xp
    def sync():
        if device == 'gpu':
            xp.cuda.Stream.null.synchronize()
    log(f'{N} sites, N_up = {nup}, J2 = {J2}: {e.dim} states (translations only), set-up {time.perf_counter()-t0:.0f} s')
    # The replayed Ritz vector of a long run can be poor although the energy has converged: once copies of converged
    # levels have formed, the later Lanczos vectors depend on the last bits, and a replay that rounds a sum differently
    # builds another basis (ground_state_vector now checks and repairs this itself).  The run is restarted from
    # the vector until its residual |H psi - E psi| is below `res`; with the repaired routine one pass suffices.
    t0 = time.perf_counter()
    its, psi, passes = 0, None, []
    for attempt in range(8):
        E, it1, _ = e.ground_energy(tol=1e-12, maxit=maxit, rtol=rtol, v0=psi)
        new, Ec, r = hs.ground_state_vector(e)
        del psi
        psi = new; its += it1; passes.append((int(it1), float(r)))
        sync()
        log(f'  pass {attempt+1}: {it1} steps, E = {Ec:.10f}, residual {r:.1e}, {time.perf_counter()-t0:.0f} s so far')
        if r < res:
            break
    e._lanczos = {}
    t_l = time.perf_counter()-t0
    res_final = r
    _, dot = hs._vec_ops(xp, True)

    # ---- bond sums and chirality sums (zero momentum)
    t0 = time.perf_counter()
    ops = []
    for u in range(6):
        m = edc.SpinHalfModel(N)
        for a in np.flatnonzero(typ == u):
            m.heisenberg(bonds[a][0], bonds[a][1], 1.0)
        ops.append(e.operator(m))
    def apply(u, x):
        y = xp.zeros_like(x); ops[u](x, y); return y
    xi = []                                                        # at most seven vectors of the sector at a time
    for k in range(2):
        p0, p1 = apply(3*k, psi), apply(3*k+1, psi)
        a = apply(3*k, p1)                                         # H_(k,0) H_(k,1) psi
        del p1
        a -= apply(3*k+1, p0)                                      # - H_(k,1) H_(k,0) psi
        del p0
        xi.append(a)                                               # X_k psi = i xi_k
    Gx0 = np.array([[float(dot(xi[k], xi[l]).real) for l in range(2)] for k in range(2)])/Nc
    ex0 = [float(dot(psi, xi[k]).real) for k in range(2)]
    del xi, a
    phi = [apply(u, psi) for u in range(6)]
    Gb0 = np.array([[float(dot(phi[u], phi[v]).real) for v in range(6)] for u in range(6)])/Nc
    eb0 = np.array([float(dot(psi, phi[u]).real) for u in range(6)])/Nc
    del phi, ops
    sync()
    log(f'  bond and chirality sums: {time.perf_counter()-t0:.0f} s; sum of bond energies {eb0.sum()*Nc:.10f}')

    # ---- diagonal sets from the weights of the representatives
    t0 = time.perf_counter()
    st = e.dev['ST'] if device == 'gpu' else e.arrs['ST']
    sh = xp.asarray(np.array([e._new[i] for i in range(N)], np.uint64))
    bi, bj = xp.asarray(np.array([b[0] for b in bonds])), xp.asarray(np.array([b[1] for b in bonds]))
    mz, Gz = xp.zeros(N), xp.zeros((N, N))
    ezz, Gzz = xp.zeros(len(bonds)), xp.zeros((len(bonds), len(bonds)))
    one = xp.uint64(1)
    for c0 in range(0, e.dim, chunk):
        s = st[c0:c0+chunk]; p = psi[c0:c0+chunk]**2
        Z = ((s[:, None] >> sh[None, :]) & one).astype(xp.float64)-0.5
        mz += p @ Z; Gz += Z.T @ (p[:, None]*Z)
        Zb = Z[:, bi]*Z[:, bj]
        ezz += p @ Zb; Gzz += Zb.T @ (p[:, None]*Zb)
    get = (lambda a: a.get()) if device == 'gpu' else (lambda a: a)
    raw = dict(mz=get(mz), Gz=get(Gz), ezz=get(ezz), Gzz=get(Gzz))
    log(f'  diagonal sets: {time.perf_counter()-t0:.0f} s')

    G = ct.space_group(cl)
    out, asym = {}, {}
    for name, kind in (('mz', 'z'), ('Gz', 'z'), ('ezz', 'zz'), ('Gzz', 'zz')):
        sp = ct.OpSpace(cl, kind, geo, G)
        tr = [gi for gi, g in enumerate(G) if g[2] == 'E']
        only_t = np.zeros_like(raw[name])
        for gi in tr:                                              # exact: the state has zero momentum
            m = sp.image[gi]
            if raw[name].ndim == 2:
                only_t[np.ix_(m, m)] += raw[name]
            else:
                only_t[m] += raw[name]
        only_t /= len(tr)
        out[name] = sp.average(raw[name])
        asym[name] = float(np.max(np.abs(out[name]-only_t)))
    assert abs(out['mz'].sum()-(nup-N/2)) < 1e-8
    out.update(Gb0=Gb0, eb0=eb0, Gx0=Gx0, bond_type=typ, tri_kind=np.array(geo['tri_kind']))
    meta = dict(cluster=A, N=N, nup=nup, J2=J2, sector='G (translations only)', E=float(Ec), residual=float(res_final), steps=int(its), passes=passes,
                dim=int(e.dim), lanczos_seconds=t_l, asym=asym, chirality_one_point=ex0, bond_energy_sum=float(eb0.sum()*Nc))
    log(f'  departure from point-group invariance: {asym}')
    return out, meta


def embed(space, small, labels):
    """Matrix on the operators of `space` that equals small[u, v] / (number per class) between classes u and v
    (the zero-momentum block written in the full index)."""
    lab = np.asarray(labels)
    return small[np.ix_(lab, lab)]/np.bincount(lab)[0]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--cluster', default='2,2,-2,4'); ap.add_argument('--nup', type=int, required=True)
    ap.add_argument('--J2', type=float, default=0.0); ap.add_argument('--device', default='gpu')
    ap.add_argument('--rtol', type=float, default=1e-9); ap.add_argument('--maxit', type=int, default=4000)
    ap.add_argument('--res', type=float, default=1e-7)
    a = ap.parse_args()
    t0 = time.perf_counter()
    out, meta = run(a.cluster, a.nup, a.J2, a.device, a.rtol, a.maxit, res=a.res, log=lambda s: print(s, flush=True))
    meta['seconds'] = time.perf_counter()-t0
    np.savez(a.out, meta=json.dumps(meta), **out)
    print(f"wrote {a.out} ({meta['seconds']:.0f} s)")
