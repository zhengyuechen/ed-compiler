#!/usr/bin/env python3
"""Correlation matrices of the lowest level of one symmetry kind, in the full S_z basis (kagome clusters to 27 sites;
36 sites near saturation).

    corr_full.py <out.npz> --cluster 3,0,0,3 --nup 15 [--J2 0] [--sector G:A1] [--threads 4] [--seed 1]

The state is found by a Lanczos run that starts from a random vector projected on the chosen kind (momentum star and
irrep of its little group, e.g. G:A1, G:E2, K:A1, K:E; 'none' = no projection) and is projected again every 25 steps.
For an irrep of dimension > 1 the vector is one member of the level; the average over the space group of every matrix
below is then the average over the level.  Written (all averaged over the space group, bench/corr_tools.py):
  one-point values  mz (S^z_i), ezz (S^z_i S^z_j on bonds), eb (S_i.S_j on bonds)
  raw two-point matrices  Gz, Gzz, Gb, Gx = Re <O_a O_b>  (connected part: G - e e^T)
  SS[i, j] = <S_i . S_j> for every pair of sites."""
import sys, os, json, math, time, argparse, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import edc, edc_lattices as L
import corr_tools as ct

PD = ctypes.POINTER(ctypes.c_double)


class Basis:
    """States of the engine's S_z sector and the index of any state of the sector (Lin tables)."""

    def __init__(self, eng):
        self.st, self.L, self.dim = eng.st, eng.L, eng.dim
        self.T = self.st.dtype.type
        self.LA = self.L//2
        self.ja, self.jb = eng.ja.astype(np.int64), eng.jb.astype(np.int64)

    def index(self, s):
        T = self.T
        return self.ja[(s >> T(self.LA)).astype(np.intp)]+self.jb[(s & T((1 << self.LA)-1)).astype(np.intp)]

    def swap_index(self, i, j):
        """Index of the state with the spins of sites i and j exchanged."""
        T, st = self.T, self.st
        d = ((st >> T(i)) ^ (st >> T(j))) & T(1)
        return self.index(st ^ (d*T((1 << i) | (1 << j))))

    def perm_index(self, p):
        """Index of g s for every basis state s, where g moves the spin of site a to site p[a]."""
        T, st = self.T, self.st
        gs = np.zeros_like(st)
        byte = np.arange(256, dtype=np.uint64)
        for c in range((self.L+7)//8):
            tab = np.zeros(256, np.uint64)
            for bi in range(8):
                a = 8*c+bi
                if a < self.L:
                    tab |= ((byte >> np.uint64(bi)) & np.uint64(1)) << np.uint64(p[a])
            gs |= tab.astype(st.dtype)[((st >> T(8*c)) & T(255)).astype(np.intp)]
        return self.index(gs)


def projector(cl, basis, sector):
    """v -> projection of v on the states of one kind 'star:irrep' (all members of the star, all rows of the irrep)."""
    if sector in (None, 'none'):
        return lambda v: v
    sname, ir = sector.split(':')
    st = next(s for s in ct.stars(cl) if s['name'] == sname)
    chi = st['table'][ir]
    def key(k):
        return tuple(round((float(x) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for x in k)
    ks = {}
    for o in cl.point_ops():
        k2 = cl.act_k(o, st['k'])
        ks.setdefault(key(k2), k2)
    assert len(ks) == st['size']
    G = ct.space_group(cl)
    dt = np.uint32 if basis.dim < 2**32 else np.int64
    trans, point = [], []
    for perm, (i, j), name in G:
        if name == 'E':
            w = sum(np.exp(1j*(k[0]*i+k[1]*j)) for k in ks.values())/cl.Nc
            assert abs(w.imag) < 1e-12
            if abs(w.real) > 1e-14:
                trans.append((w.real, basis.perm_index(perm).astype(dt)))
        if (i, j) == (0, 0) and name in st['little']:
            c = chi[st['cls'][name]]*chi['E']/len(st['little'])
            if c:
                point.append((c, basis.perm_index(perm).astype(dt)))
    def apply(v):
        a = np.zeros_like(v)
        for w, idx in trans:
            a += w*v[idx]
        b = np.zeros_like(v)
        for c, idx in point:
            b += c*a[idx]
        return b
    return apply


def lowest(eng, project, seed=1, rtol=1e-9, maxit=3000, every=25, log=None):
    """Lowest level reachable from a projected random start: (psi, E, |H psi - E psi|, steps)."""
    from scipy.linalg import eigh_tridiagonal
    n = eng.dim
    def p(a):
        return a.ctypes.data_as(PD)
    v0 = project(np.random.default_rng(seed).standard_normal(n)); v0 /= np.linalg.norm(v0)
    if n < 20000:
        every = 1                       # small kinds are exhausted in a few steps; keep every vector inside
    ab = np.zeros(2)
    def run(m_stop, coef):
        v, vp, w = v0.copy(), np.zeros(n), np.empty(n)
        A, B, bprev, psi = [], [], 0.0, (np.zeros(n) if coef is not None else None)
        for it in range(1, m_stop+1):
            if psi is not None:
                psi += coef[it-1]*v
            eng.lib.lanczos_step(0, p(v), p(vp), p(w), bprev, p(ab))
            A.append(ab[0]); B.append(ab[1]); bprev = ab[1]
            vp, v, w = v, w, vp
            if it % every == 0:
                v[:] = project(v); vp[:] = project(vp)
            if coef is None and it >= 2 and (it % 10 == 0 or n < 20000 or ab[1] < 1e-8):   # ab[1] ~ 0: the kind is exhausted
                e, U = eigh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, 0))
                r = abs(B[-1]*U[-1, 0])
                if log:
                    log(f'    step {it}: E = {e[0]:.12f}, estimate of the residual {r:.1e}')
                if r < rtol or ab[1] < 1e-8:
                    return np.array(A), np.array(B), None
        return np.array(A), np.array(B), psi
    # A long run without reorthogonalisation leaves the Ritz vector with a residual of 1e-6 or so although the estimate
    # is far smaller; restarting from that vector removes it in a few tens of steps.
    total = 0
    for attempt in range(6):
        A, B, _ = run(maxit, None)
        m = len(A); total += m
        _, U = eigh_tridiagonal(A, B[:m-1], select='i', select_range=(0, 0))
        _, _, psi = run(m, U[:, 0])
        psi = project(psi); psi /= np.linalg.norm(psi)
        hp = eng.matvec(psi)
        E = float(psi @ hp)
        res = float(np.linalg.norm(hp-E*psi))
        if log:
            log(f'    pass {attempt+1}: {m} steps, E = {E:.12f}, residual {res:.1e}')
        if res < 20*rtol:
            break
        v0 = psi
    return psi, E, res, total


def gram_stream(gen, n, psi, block=8):
    """G[a, b] = gen(a) . gen(b) and e[a] = psi . gen(a), holding `block` vectors at a time."""
    G, e = np.zeros((n, n)), np.zeros(n)
    for a0 in range(0, n, block):
        blk = np.stack([gen(a) for a in range(a0, min(n, a0+block))])
        e[a0:a0+len(blk)] = blk @ psi
        for b in range(a0, n):
            vb = blk[b-a0] if b < a0+len(blk) else gen(b)
            G[a0:a0+len(blk), b] = blk @ vb
            G[b, a0:a0+len(blk)] = G[a0:a0+len(blk), b]
    return G, e


def measure(cl, basis, psi, geo=None, chunk=1 << 18, log=None):
    """Raw (not yet symmetrised) one- and two-point values of the four operator sets in the real vector psi."""
    geo = geo or ct.geometry(cl)
    N, st, T = cl.N, basis.st, basis.T
    bonds, tris = geo['bonds'], geo['tris']
    bi, bj = np.array([b[0] for b in bonds]), np.array([b[1] for b in bonds])
    t0 = time.perf_counter()
    mz, Gz = np.zeros(N), np.zeros((N, N))
    ezz, Gzz = np.zeros(len(bonds)), np.zeros((len(bonds), len(bonds)))
    sh = np.arange(N).astype(st.dtype)
    for c0 in range(0, basis.dim, chunk):
        s = st[c0:c0+chunk]; p = psi[c0:c0+chunk]**2
        Z = ((s[:, None] >> sh[None, :]) & T(1)).astype(np.float64)-0.5
        mz += p @ Z; Gz += Z.T @ (p[:, None]*Z)
        Zb = Z[:, bi]*Z[:, bj]
        ezz += p @ Zb; Gzz += Zb.T @ (p[:, None]*Zb)
    if log:
        log(f'    diagonal sets: {time.perf_counter()-t0:.0f} s'); t0 = time.perf_counter()
    Gb, eb = gram_stream(lambda a: 0.5*psi[basis.swap_index(*bonds[a])]-0.25*psi, len(bonds), psi)
    if log:
        log(f'    bonds: {time.perf_counter()-t0:.0f} s'); t0 = time.perf_counter()
    def xi(a):
        i, j, k = tris[a]
        ij, jk = basis.swap_index(i, j), basis.swap_index(j, k)
        return 0.25*(psi[jk][ij]-psi[ij][jk])
    Gx, ex = gram_stream(xi, len(tris), psi)
    assert np.max(np.abs(ex)) < 1e-10, 'the chirality of a real state must vanish'
    if log:
        log(f'    triangles: {time.perf_counter()-t0:.0f} s'); t0 = time.perf_counter()
    SS = np.zeros((N, N))
    for i in range(N):
        SS[i, i] = 0.75
        for j in range(i+1, N):
            SS[i, j] = SS[j, i] = 0.5*float(psi @ psi[basis.swap_index(i, j)])-0.25
    if log:
        log(f'    all pairs: {time.perf_counter()-t0:.0f} s')
    return dict(mz=mz, Gz=Gz, ezz=ezz, Gzz=Gzz, eb=eb, Gb=Gb, Gx=Gx, SS=SS)


def symmetrise(cl, raw, geo=None):
    geo = geo or ct.geometry(cl)
    G = ct.space_group(cl)
    sp = {k: ct.OpSpace(cl, k, geo, G) for k in ('z', 'zz', 'b', 'x')}
    out = {}
    for name, kind in (('mz', 'z'), ('Gz', 'z'), ('SS', 'z'), ('ezz', 'zz'), ('Gzz', 'zz'), ('eb', 'b'), ('Gb', 'b'), ('Gx', 'x')):
        out[name] = sp[kind].average(raw[name])
    return out


def run(cluster, nup, J2=0.0, sector='none', threads=4, seed=1, rtol=1e-9, log=print):
    A = [int(x) for x in cluster.split(',')]
    cl = L.LatticeCluster('kagome', (A[0], A[1]), (A[2], A[3]))
    t0 = time.perf_counter()
    eng = edc.Engine(cl.heisenberg(1.0, J2), nup, threads=threads)
    basis = Basis(eng)
    log(f'{cl.N} sites, N_up = {nup}, J2 = {J2}, kind {sector}: {eng.dim} states, set-up {time.perf_counter()-t0:.0f} s')
    t0 = time.perf_counter()
    proj = projector(cl, basis, sector)
    log(f'  projector tables: {time.perf_counter()-t0:.0f} s'); t0 = time.perf_counter()
    psi, E, res, steps = lowest(eng, proj, seed=seed, rtol=rtol, log=log if eng.dim > 10**6 else None)
    log(f'  E = {E:.10f}, residual {res:.1e}, {steps} steps, {time.perf_counter()-t0:.0f} s')
    geo = ct.geometry(cl)
    raw = measure(cl, basis, psi, geo, log=log)
    out = symmetrise(cl, raw, geo)
    N = cl.N
    assert abs(out['mz'].sum()-(nup-N/2)) < 1e-9
    e1 = float(out['eb'].sum())
    if J2 == 0:
        assert abs(e1-E) < 1e-7*max(1.0, abs(E)), (e1, E)
    meta = dict(cluster=A, N=N, nup=nup, J2=J2, sector=sector, E=E, residual=res, steps=steps, dim=eng.dim, seed=seed,
                bond_energy_sum=e1)
    return out, meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--cluster', default='3,0,0,3'); ap.add_argument('--nup', type=int, required=True)
    ap.add_argument('--J2', type=float, default=0.0); ap.add_argument('--sector', default='none')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--rtol', type=float, default=1e-9)
    a = ap.parse_args()
    t0 = time.perf_counter()
    out, meta = run(a.cluster, a.nup, a.J2, a.sector, a.threads, a.seed, a.rtol, log=lambda s: print(s, flush=True))
    meta['seconds'] = time.perf_counter()-t0
    np.savez(a.out, meta=json.dumps(meta), **out)
    print(f"wrote {a.out} ({meta['seconds']:.0f} s)")
