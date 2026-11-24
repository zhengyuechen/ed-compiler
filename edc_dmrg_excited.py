#!/usr/bin/env python3
"""A further state of the same (S_z, K) sector with the layer DMRG: DMRG for

    H' = H + sum_p w_p |psi_p><psi_p|

with psi_p states found before (saved with edc_dmrg_state, or engines still in memory).  For w_p larger than the distance
of the wanted level from psi_p's level, the lowest state of H' is the lowest state of H orthogonal to the psi_p (to the
accuracy of the psi_p).  The same construction as the `weight` of excited-state DMRG in ITensorMPS.jl.

    cls = excited(ed.DMRG)            # or excited(e3r.DMRGGPU3r), excited(e3.DMRGGPU3)
    eng = cls(model, nup, K, seed=1)
    eng.orthogonal_to('state0.npz', weight=4.0)      # before the first sweep; several calls for several states
    eng.sweep(D, alpha, tol) ...                      # as before; the value a sweep returns is the lowest value of H'
    eng.last = dict(E_penalised, E, overlaps)         # after every solve: E = <psi|H|psi> = E_penalised - sum_p w_p overlap_p^2

What is added to the engine: for each psi_p the overlap environments of the current state with it, per bond label
    OL[r]_i (D_i x D0_i)  left of layer r        OR[r]_i (D_i x D0_i)  right of bond r
(D: sizes of the current state, D0: sizes of psi_p; both states carry the same labels because they are in the same
sector).  They grow with the Hamiltonian environments (after every move).  At the centre layer r
    phi_i[:, slab s] = OL[r]_i  M0_i[:, slab s]  OR[r+1]_j^T          (j = right label of slab s of left label i)
is psi_p written in the current basis, <psi_p|psi> = phi . M, and the product of the layer solve gets + w_p phi (phi . v).
The tensors of psi_p are used as stored (any canonical form of a normalised state gives the same contraction)."""
import copy
import numpy as np
import edc_dmrg as ed
import edc_dmrg_state as st


def _slabs(p):
    """(i, s, j, first column, width) of every slab of the M layout of plan p (cached on the plan)."""
    out = getattr(p, '_slab_list', None)
    if out is None:
        ii, ss = np.nonzero((p.cst >= 0) & (p.DL[:, None] > 0)); out = p._slab_list = [(int(i), int(s), int(p.mr[i, s]), int(p.cst[i, s]), int(p.cw[i, s])) for i, s in zip(ii, ss)]
    return out


class Stored:
    """One state to stay orthogonal to: its tensors (M layout, on the host or the device of the engine), its plans, its
    overlap environments with the engine's current state."""
    def __init__(self, eng, source, weight, xp, host=False):
        self.w = float(weight); n = eng.n; self.xp = xp; self.host = host     # host=True: tensors kept in host memory, one layer at a time on the device
        if isinstance(source, str):
            f = np.load(source if source.endswith('.npz') else source + '.npz', allow_pickle=False); head = st.json.loads(str(f['head'])); md = eng.model
            for key, val in (('N', md.lat.N), ('c', md.lat.c), ('mod', md.mod), ('Delta', md.Delta), ('lattice_hash', st._lattice_hash(md)),
                             ('nup', int(eng.bond[n].m[0])), ('K', int(eng.bond[n].k[0]))):
                assert head[key] == val, f'the stored state is for another model or sector: {key} {head[key]} != {val}'
            dims = [f[f'd{r}'] for r in range(n+1)]; T = [np.ascontiguousarray(f[f'M{r}'], dtype=np.float64) for r in range(n)]
            for r in range(n+1):
                assert np.array_equal(eng.bond[r].m, f[f'm{r}']) and np.array_equal(eng.bond[r].k, f[f'k{r}'])
            self.name = source; self.head = head
        else:                                                          # an engine: a copy of its tensors now
            assert st.canonical(source)['centre'] is not None, 'the other engine is not in a canonical form'
            dims = [b.dims.copy() for b in source.bond]; T = [np.array(st._host(m), dtype=np.float64) for m in source.M]; self.name = 'engine'; self.head = None
        nrm = 1.0
        self.bond = []
        for r in range(n+1):
            b = copy.copy(eng.bond[r]); b.dims = np.array(dims[r], dtype=eng.bond[r].dims.dtype); self.bond.append(b)
        self.plan = [ed.Plan(eng.model, r, self.bond[r], self.bond[r+1]) for r in range(n)]
        for r in range(n):
            assert T[r].size == self.plan[r].size
        self.M = [np.ascontiguousarray(t) if host else xp.asarray(t) for t in T]; self.OL = [None]*(n+1); self.OR = [None]*(n+1)
        self.OL[0] = [xp.ones((int(a), int(b))) if a and b else xp.zeros((int(a), int(b))) for a, b in zip(eng.bond[0].dims, self.bond[0].dims)]
        self.OR[n] = [xp.ones((int(a), int(b))) if a and b else xp.zeros((int(a), int(b))) for a, b in zip(eng.bond[n].dims, self.bond[n].dims)]

    def _blocks(self, p, M, p0, r):
        """Pairs (A, B, i, j, slab) over the slabs present in both states: A = M_i[:, slab] of the current tensor
        (D_i x D_j), B = M0_i[:, slab] of the stored one (D0_i x D0_j)."""
        M0 = self.xp.asarray(self.M[r]) if self.host else self.M[r]; s0 = {(i, s): (c0, w) for i, s, j, c0, w in _slabs(p0)}
        for i, s, j, c0, w in _slabs(p):
            q = s0.get((i, s))
            if q is None or not w or not q[1]:
                continue
            Di, Ci = int(p.DL[i]), int(p.C[i]); D0i, C0i = int(p0.DL[i]), int(p0.C[i])
            yield (M[int(p.Moff[i]):int(p.Moff[i+1])].reshape(Di, Ci)[:, c0:c0+w], M0[int(p0.Moff[i]):int(p0.Moff[i+1])].reshape(D0i, C0i)[:, q[0]:q[0]+q[1]], i, j, (c0, w))

    def grow_left(self, eng, r, M):                                    # OL[r+1] from OL[r] and the (left-orthonormal) tensor of layer r
        xp = self.xp; p = eng.plan[r]; p0 = self.plan[r]; new = [xp.zeros((int(a), int(b))) for a, b in zip(p.DR, p0.DR)]; OL = self.OL[r]
        for A, B, i, j, _ in self._blocks(p, M, p0, r):
            new[j] += A.T @ (OL[i] @ B)
        self.OL[r+1] = new

    def grow_right(self, eng, r, M):                                   # OR[r] from OR[r+1] and the (right-orthonormal) tensor of layer r
        xp = self.xp; p = eng.plan[r]; p0 = self.plan[r]; new = [xp.zeros((int(a), int(b))) for a, b in zip(p.DL, p0.DL)]; OR = self.OR[r+1]
        for A, B, i, j, _ in self._blocks(p, M, p0, r):
            new[i] += A @ (OR[j] @ B.T)
        self.OR[r] = new

    def phi(self, eng, r):                                             # the stored state in the current basis of the centre layer r (M layout)
        xp = self.xp; p = eng.plan[r]; p0 = self.plan[r]; out = xp.zeros(p.size); OL = self.OL[r]; OR = self.OR[r+1]; M = out
        for A, B, i, j, (c0, w) in self._blocks(p, M, p0, r):
            A[...] = OL[i] @ B @ OR[j].T                               # A is a view into out
        return out


_CACHE = {}


def excited(base, maxit=40, host_stored=False):
    """The class `base` (edc_dmrg.DMRG or a GPU back end with the tensors on the device: v3, v3r) with the penalty terms.
    maxit: largest number of Krylov vectors per layer solve (40 as in the engine; a smaller cap lowers the peak device
    memory, 40 vectors of the largest tensor dominate it, but more solves then end at the cap).
    host_stored: keep the tensors of the stored states in host memory and move one layer at a time to the device when an
    overlap environment or the stored state at the centre is formed (saves the device memory of the stored states, about
    one state per orthogonality term, for one transfer of a layer tensor per use).  Results are the same."""
    key = (base, maxit, host_stored)
    if key in _CACHE:
        return _CACHE[key]
    device = hasattr(base, 'version') and base.version.startswith('gpu')

    class Excited(base):
        version = getattr(base, 'version', 'host') + '+orth'

        def __init__(self, *a, **k):
            self.prev = []; self.last = None
            super().__init__(*a, **k)
            if device:
                import cupy as cp
                assert type(self.M[0]) is not np.ndarray, 'GPU back ends with the tensors on the host (v1, v2) are not supported here: use v3'
                self._xp = cp
            else:
                self._xp = np

        def orthogonal_to(self, source, weight):
            """Add a state to stay orthogonal to.  Call with the centre at layer 0 (a new engine, or after a sweep)."""
            assert st.canonical(self)['centre'] == 0, 'call orthogonal_to with the centre at layer 0'
            s = Stored(self, source, weight, self._xp, host=self.host_stored)
            for r in range(self.n-1, 0, -1):
                s.grow_right(self, r, self.M[r])
            self.prev.append(s)
            return s

        def overlaps(self, r=0):
            """<psi_p|psi> for every stored state, with the centre at layer r."""
            v = self.M[r]/float((self.M[r] @ self.M[r])**0.5)
            return [float(s.phi(self, r) @ v) for s in self.prev]

        def _grow_left(self, r):
            super()._grow_left(r)
            for s in getattr(self, 'prev', []):
                s.grow_left(self, r, self.M[r])

        def _grow_right(self, r):
            super()._grow_right(r)
            for s in getattr(self, 'prev', []):                        # (the first growths run inside the base constructor, before any state is added)
                s.grow_right(self, r, self.M[r])

        def _after(self, r, E, pens):
            v = self.M[r]; ov = [float(phi @ v) for w, phi in pens]
            self.last = dict(layer=r, E_penalised=E, E=E - sum(w*o*o for (w, phi), o in zip(pens, ov)), overlaps=ov)

        if device:
            def solve(self, r, tol=1e-6, maxit=None):
                maxit = self.maxit if maxit is None else maxit
                if not self.prev:
                    return super().solve(r, tol, maxit)
                pens = [(s.w, s.phi(self, r)) for s in self.prev]
                product_cls = self.product_cls

                class Penalised(product_cls):
                    def apply(product, v):
                        out = super().apply(v)
                        for weight, phi in pens:
                            out += (weight * float(phi @ v)) * phi
                        return out

                self.product_cls = Penalised
                try:
                    E = super().solve(r, tol, maxit)
                finally:
                    self.product_cls = product_cls
                self._after(r, E, pens)
                return E
        else:
            def solve(self, r, tol=1e-6, maxit=None):
                maxit = self.maxit if maxit is None else maxit                    # edc_dmrg.DMRG.solve with the penalty in the product
                if not self.prev:
                    return super().solve(r, tol, maxit)
                import time
                p = self.plan[r]; lib = self.lib; n = p.size; U = np.empty(n); oU = np.empty(n); _p = ed._p
                LH, GL, RH, GR, QL, PR = (_p(a) for a in (self.LH[r], self.GL[r], self.RH[r+1], self.GR[r+1], self.Q[r], self.P[r+1]))
                pens = [(s.w, s.phi(self, r)) for s in self.prev]

                def mv(v):
                    out = np.zeros(n); lib.heff(p.ref, LH, GL, RH, GR, QL, PR, _p(v), _p(out), _p(U), _p(oU)); self.stats['products'] += 1
                    for w, phi in pens:
                        out += (w*float(phi @ v))*phi
                    return out
                t0 = time.perf_counter(); v = self.M[r]/np.linalg.norm(self.M[r])
                if self.noise > 0:
                    xi = self.rng.standard_normal(n); v = v + (self.noise/np.linalg.norm(xi))*xi; v /= np.linalg.norm(v)
                # as edc_dmrg.DMRG.solve: Krylov vectors as rows of one array, re-orthogonalisation w -= (V w) V
                m = min(maxit, n); V = np.empty((m, n)); V[0] = v; al = []; be = []; E = None; y = np.array([1.0]); k = 1
                for it in range(m):
                    w = mv(V[it]); a = float(V[it] @ w); al.append(a)
                    w -= a*V[it]
                    if be:
                        w -= be[-1]*V[it-1]
                    w -= (V[:k] @ w) @ V[:k]
                    b = float(np.linalg.norm(w))
                    ev, Sv = np.linalg.eigh(np.diag(al) + np.diag(be, 1) + np.diag(be, -1)); E = float(ev[0]); y = Sv[:, 0]
                    if b*abs(y[-1]) < tol*max(1.0, abs(E)) or b < 1e-13 or k == n or k == m:
                        break
                    be.append(b); V[k] = w/b; k += 1
                x = y[:k] @ V[:k]; self.M[r] = x/np.linalg.norm(x)
                self.stats['solves'] += 1; self.stats['t_products'] += time.perf_counter()-t0
                self._after(r, E, pens)
                return E

    Excited.__name__ = f'Excited_{base.__name__}'; Excited.maxit = maxit; Excited.host_stored = host_stored; _CACHE[key] = Excited
    return Excited


def state_overlap(eng, source):
    """<source|state of eng>, source = a file of edc_dmrg_state or another engine; the centre of eng must be at layer 0.
    Both states normalised: the value is the overlap of two unit vectors (sign included)."""
    assert st.canonical(eng)['centre'] == 0, 'the centre of the engine must be at layer 0'
    if type(eng.M[0]) is not np.ndarray:
        import cupy as xp
    else:
        xp = np
    s = Stored(eng, source, 0.0, xp)
    for r in range(eng.n-1, 0, -1):
        s.grow_right(eng, r, eng.M[r])
    v = eng.M[0]
    return float(s.phi(eng, 0) @ v)/float((v @ v)**0.5)
