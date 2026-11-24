#!/usr/bin/env python3
"""Save a state of the layer DMRG and read it back: every layer tensor together with the symmetry
labels (number of up spins m, momentum k) and the sector sizes of every bond.  Works for the host engine (edc_dmrg.DMRG)
and the GPU back ends (v1 ... v3; tensors and environments on the device are copied to the host for the file).

    save_state(eng, path, **meta)                 one .npz file; the centre of the state is found from the tensors
    eng = load_state(path, model, cls=ed.DMRG)    an engine holding the state, environments rebuilt from the tensors
    canonical(eng)                                where the centre is, and how orthonormal the other tensors are
    energy(eng, centre)                           <psi|H|psi> / <psi|psi> from one product with the effective Hamiltonian
                                                  of the centre layer (no solve, the state is not changed)
    layer_sz(eng)                                 <S_z> of every layer, from host copies (the engine is not changed)

A state is an MPS in mixed canonical form: one centre layer c, the tensors left of it left-orthonormal, right of it
right-orthonormal.  After DMRG.sweep the centre is layer 0; after a pass that moves the centre to the end it is the
last layer.  The file stores the centre it finds; the reader rebuilds the left environments of layers < c and the right
environments of layers > c with the engine's own growth routines.

The file does not hold the lattice: the reader is given the model and refuses a file whose lattice (hash of the bond
list), anisotropy, sector or bond labels differ from it."""
import json, time, hashlib
import numpy as np
import edc_dmrg as ed

FORMAT = 1


def _host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def _lattice_hash(model):
    b = sorted((min(int(i), int(j)), max(int(i), int(j)), round(float(J), 12)) for i, j, J in model.lat.global_bonds())
    return hashlib.sha1(json.dumps([b, model.lat.c, [len(y) for y in model.lat.layers]]).encode()).hexdigest()[:16]


def canonical(eng, tol=1e-8):
    """dict(centre, left, right): left[r] / right[r] = largest deviation of layer r from left / right orthonormality.
    centre = the smallest c with every layer < c left-orthonormal and every layer > c right-orthonormal (None if none)."""
    n = eng.n; L = np.zeros(n); R = np.zeros(n)
    for r in range(n):
        p = eng.plan[r]; M = np.ascontiguousarray(_host(eng.M[r])); U = np.empty(p.size); eng.lib.m2u(p.ref, M.ctypes.data, U.ctypes.data)
        for i in range(p.nL):
            Di, Ci = int(p.DL[i]), int(p.C[i])
            if Di:
                A = M[p.Moff[i]:p.Moff[i+1]].reshape(Di, Ci); R[r] = max(R[r], float(np.abs(A @ A.T - np.eye(Di)).max()) if Ci else 1.0)
        for j in range(p.nR):
            Rj, Dj = int(p.R[j]), int(p.DR[j])
            if Dj:
                A = U[p.Uoff[j]:p.Uoff[j+1]].reshape(Rj, Dj); L[r] = max(L[r], float(np.abs(A.T @ A - np.eye(Dj)).max()) if Rj else 1.0)
    c = next((c for c in range(n) if np.all(L[:c] < tol) and np.all(R[c+1:] < tol)), None)
    return dict(centre=c, left=L.tolist(), right=R.tolist())


def energy(eng, centre=0):
    """(E, norm, local residual): E = <M|Heff|M>/<M|M> at the centre layer, norm = |M|, and |Heff M - E M| / |M| (the residual
    of the centre tensor alone; the residual of the whole state needs H on the whole state)."""
    r = centre; p = eng.plan[r]; n = p.size
    if hasattr(eng, 'tables'):                                         # GPU back end: the product on the device
        import cupy as cp, edc_dmrg_gpu as eg
        v = cp.asarray(eng.M[r]); nrm = float(cp.linalg.norm(v)); v = v/nrm; w = eg.Product(eng, r, {}).apply(v); E = float(v @ w)
        return E, nrm, float(cp.linalg.norm(w - E*v))
    v = np.ascontiguousarray(eng.M[r]); nrm = float(np.linalg.norm(v)); v = v/nrm; w = np.zeros(n); U = np.empty(n); oU = np.empty(n); _p = ed._p
    eng.lib.heff(p.ref, _p(eng.LH[r]), _p(eng.GL[r]), _p(eng.RH[r+1]), _p(eng.GR[r+1]), _p(eng.Q[r]), _p(eng.P[r+1]), _p(v), _p(w), _p(U), _p(oU)); E = float(v @ w)
    return E, nrm, float(np.linalg.norm(w - E*v))


def _rfac(A):
    """Square R with R^T R = A^T A (A: rows x cols, any shape): the R of a QR, padded with zero rows if A is wide."""
    R = np.linalg.qr(A, mode='r'); out = np.zeros((A.shape[1], A.shape[1])); out[:R.shape[0]] = R
    return out


def bond_weights(eng):
    """For every bond r = 0 ... n the weight of each of its labels in the state (they sum to 1).  With the centre on layer
    r the weights of its left (right) bond are the squared norms of the blocks of its tensor in the M (U) layout; the
    centre is carried through all layers on host copies (the factor R of each block goes to the neighbour; nothing is
    truncated, sizes stay).  The engine is not changed."""
    n = eng.n; c = canonical(eng)['centre']; assert c is not None, 'the tensors are not in a canonical form'
    T0 = np.array(_host(eng.M[c]), dtype=float); nrm2 = float(T0 @ T0); out = [None]*(n+1); lib = eng.lib
    blocks = lambda v, off, m: np.array([float(np.sum(v[off[i]:off[i+1]]**2))/nrm2 for i in range(m)])
    T = T0.copy()
    for r in range(c, n):                                              # centre at r, moving right
        p = eng.plan[r]; U = np.empty(p.size); lib.m2u(p.ref, T.ctypes.data, U.ctypes.data)
        if r == c:
            out[r] = blocks(T, p.Moff, p.nL)
        out[r+1] = blocks(U, p.Uoff, p.nR)
        if r == n-1:
            break
        pn = eng.plan[r+1]; Tn = np.ascontiguousarray(_host(eng.M[r+1]), dtype=float); nxt = np.zeros(pn.size)
        for j in range(p.nR):
            Rj, Dj, Cj = int(p.R[j]), int(p.DR[j]), int(pn.C[j])
            if Rj and Dj and Cj:
                nxt[pn.Moff[j]:pn.Moff[j+1]] = (_rfac(U[p.Uoff[j]:p.Uoff[j+1]].reshape(Rj, Dj)) @ Tn[pn.Moff[j]:pn.Moff[j+1]].reshape(Dj, Cj)).reshape(-1)
        T = nxt
    T = T0.copy()
    for r in range(c, 0, -1):                                          # centre at r, moving left
        p = eng.plan[r]; pp = eng.plan[r-1]; Tp = np.ascontiguousarray(_host(eng.M[r-1]), dtype=float); Up = np.empty(pp.size); lib.m2u(pp.ref, Tp.ctypes.data, Up.ctypes.data)
        new = np.zeros(pp.size)
        for i in range(p.nL):
            Di, Ci, Ri = int(p.DL[i]), int(p.C[i]), int(pp.R[i])
            if Di and Ci and Ri:
                new[pp.Uoff[i]:pp.Uoff[i+1]] = (Up[pp.Uoff[i]:pp.Uoff[i+1]].reshape(Ri, Di) @ _rfac(T[p.Moff[i]:p.Moff[i+1]].reshape(Di, Ci).T).T).reshape(-1)
        T = np.zeros(pp.size); lib.u2m(pp.ref, new.ctypes.data, T.ctypes.data, 0); out[r-1] = blocks(T, pp.Moff, pp.nL)
    return out


def layer_sz(eng):
    """<S_z> of every layer (sums to nup - N/2)."""
    w = bond_weights(eng); mleft = [float((eng.bond[r].m*w[r]).sum()) for r in range(eng.n+1)]
    return (np.diff(mleft) - np.array([len(y) for y in eng.model.lat.layers])/2.0).tolist()


def save_state(eng, path, **meta):
    """Write the state of eng to path (.npz).  meta: anything JSON can hold (energy, D, seed, start pattern, job ...)."""
    can = canonical(eng); c = can['centre']; assert c is not None, f'not a canonical state: left {max(can["left"]):.1e}, right {max(can["right"]):.1e}'
    md = eng.model; nup, K = int(eng.bond[eng.n].m[0]), int(eng.bond[eng.n].k[0]); E, nrm, res = energy(eng, c)
    head = dict(format=FORMAT, lattice=md.lat.name, N=int(md.lat.N), c=int(md.lat.c), layers=[len(y) for y in md.lat.layers], Delta=float(md.Delta), mod=int(md.mod),
                lattice_hash=_lattice_hash(md), nup=nup, K=K, centre=int(c), bond_sizes=[int(b.dims.sum()) for b in eng.bond], energy=E, norm=nrm,
                centre_residual=res, engine=getattr(eng, 'version', 'host'), saved=time.strftime('%Y-%m-%d %H:%M:%S'), meta=meta)
    arr = {'head': np.array(json.dumps(head))}
    for r in range(eng.n+1):
        b = eng.bond[r]; arr[f'm{r}'] = b.m.astype(np.int32); arr[f'k{r}'] = b.k.astype(np.int32); arr[f'd{r}'] = b.dims.astype(np.int32)
    for r in range(eng.n):
        arr[f'M{r}'] = np.array(_host(eng.M[r]), dtype=np.float64)
    np.savez(path, **arr)
    return head


def read_head(path):
    return json.loads(str(np.load(path, allow_pickle=False)['head']))


def load_state(path, model, cls=ed.DMRG, **kw):
    """An engine of class cls holding the saved state: same tensors, same labels, environments rebuilt.  The head of the
    file is in eng.loaded; eng.loaded['centre'] is the centre layer (a sweep of the engine expects the centre at layer 0:
    DMRG.sweep starts with a solve of layer 0 and needs the right environments, which are there only if centre == 0)."""
    f = np.load(path if str(path).endswith('.npz') else str(path) + '.npz', allow_pickle=False); head = json.loads(str(f['head']))
    assert head['format'] == FORMAT, 'unknown file format'
    for key, val in (('N', model.lat.N), ('c', model.lat.c), ('mod', model.mod), ('Delta', model.Delta), ('lattice_hash', _lattice_hash(model))):
        assert head[key] == val, f'the file is for another model: {key} {head[key]} != {val}'
    eng = cls(model, head['nup'], head['K'], **kw); n = eng.n; c = head['centre']
    dev_env = getattr(eng, '_env_dev', False); dev_M = type(eng.M[0]) is not np.ndarray     # v3 keeps the tensors on the device
    if dev_env or dev_M:
        import cupy as cp
    for r in range(n+1):
        b = eng.bond[r]; assert np.array_equal(b.m, f[f'm{r}']) and np.array_equal(b.k, f[f'k{r}']), f'labels of bond {r} differ from the file'
        b.dims[:] = f[f'd{r}']
    for r in range(n):
        p = eng.plan[r] = ed.Plan(model, r, eng.bond[r], eng.bond[r+1]); M = np.ascontiguousarray(f[f'M{r}'], dtype=np.float64)
        assert M.size == p.size, f'tensor of layer {r}: {M.size} numbers, the labels give {p.size}'
        eng.M[r] = cp.asarray(M) if dev_M else M
    z = (lambda: cp.zeros(1)) if dev_env else (lambda: np.zeros(1))
    eng.LH = [None]*(n+1); eng.GL = [None]*(n+1); eng.RH = [None]*(n+1); eng.GR = [None]*(n+1); eng.Q = [z() for _ in range(n+1)]; eng.P = [z() for _ in range(n+1)]
    eng.LH[0] = z(); eng.GL[0] = z(); eng.RH[n] = z(); eng.GR[n] = z()
    if hasattr(eng, '_dev'):
        eng._dev = None
    for r in range(n-1, c, -1):
        eng._grow_right(r)
    for r in range(c):
        eng._grow_left(r)
    eng.loaded = head
    return eng
