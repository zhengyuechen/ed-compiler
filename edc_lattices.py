#!/usr/bin/env python3
"""Periodic clusters of the square, triangular, honeycomb and kagome lattices for the ED engines.

Coordinates are lattice coordinates (x, y) = x a1 + y a2 of the Bravais lattice (square: a1 = (1, 0), a2 = (0, 1);
hexagonal lattices: a1 = (1, 0), a2 = (1/2, sqrt3/2)); basis sites carry fractional offsets chosen so that the point
group acts about the origin (hexagon centres for honeycomb and kagome).  A cluster is the Bravais lattice modulo the
integer vectors A1, A2 (columns of M).

Interface used by edc_gpu_lean.GPULean and edc_k (same as edc_lattice.Cluster):
  N, n1, m, nblocks, tx, ty (site permutations for one step along a1, a2), point_ops(), point_perm(name), momenta(),
  little_group(k), irrep_table(k) -> {name: {op: character}} (all 1D irreps of the little group), group(k, irrep, ops).
Momenta are lattice-reduced: k = (k1, k2) with phase k1 per step along a1 and k2 per step along a2.
Sites are numbered sublattice-major and along tx orbits: index = (s m + j) n1 + i for basis site s at cell tx^i ty^j 0,
so tx is a rotation by one bit inside each of the nblocks = nb m blocks of n1 bits."""
import cmath, math, itertools
from fractions import Fraction as Fr
import numpy as np
import edc

S3 = math.sqrt(3)
KINDS = {
    'square': dict(cart=np.array([[1.0, 0.0], [0.0, 1.0]]).T, basis=[(Fr(0), Fr(0))],
                   gens={'C4': ((0, -1), (1, 0)), 'Mx': ((-1, 0), (0, 1))}, order=4),
    'triangular': dict(cart=np.array([[1.0, 0.0], [0.5, S3/2]]).T, basis=[(Fr(0), Fr(0))],
                       gens={'C6': ((0, -1), (1, 1)), 'My': ((1, 1), (0, -1))}, order=6),
    'honeycomb': dict(cart=np.array([[1.0, 0.0], [0.5, S3/2]]).T, basis=[(Fr(1, 3), Fr(1, 3)), (Fr(2, 3), Fr(2, 3))],
                      gens={'C6': ((0, -1), (1, 1)), 'My': ((1, 1), (0, -1))}, order=6),
    'kagome': dict(cart=np.array([[1.0, 0.0], [0.5, S3/2]]).T, basis=[(Fr(1, 2), Fr(0)), (Fr(0), Fr(1, 2)), (Fr(1, 2), Fr(1, 2))],
                   gens={'C6': ((0, -1), (1, 1)), 'My': ((1, 1), (0, -1))}, order=6),
}


def _point_group(kind):
    """All point operations as integer matrices (lattice coordinates), named Rn (rotation by n steps of 2 pi/order)
    and Sn (Rn followed by the generating mirror)."""
    d = KINDS[kind]
    rname = 'C4' if d['order'] == 4 else 'C6'
    mname = [k for k in d['gens'] if k != rname][0]
    R, Mi = np.array(d['gens'][rname]), np.array(d['gens'][mname])
    ops = {}
    P = np.eye(2, dtype=np.int64)
    for n in range(d['order']):
        ops[f'R{n}'] = P.copy()
        ops[f'S{n}'] = (P @ Mi).astype(np.int64)
        P = (R @ P).astype(np.int64)
    ops['E'] = ops.pop('R0')
    return ops


def _root12(v):
    """The exact 12th root of unity nearest to v (characters of 1D irreps of C6v/C4v subgroups are 12th roots)."""
    m = round(cmath.phase(v)*12/(2*math.pi)) % 12
    return {0: 1+0j, 3: 1j, 6: -1+0j, 9: -1j}.get(m, cmath.exp(2j*math.pi*m/12))


class LatticeCluster:
    def __init__(self, kind, A1, A2):
        self.kind = kind
        d = KINDS[kind]
        self.cart, self.basis = d['cart'], d['basis']
        self.nb = len(self.basis)
        self.M = np.array([A1, A2], dtype=np.int64).T
        self.Nc = int(abs(round(np.linalg.det(self.M))))
        self.N = self.Nc*self.nb
        self.Minv = np.linalg.inv(self.M.astype(float))
        n1 = 1
        while self.red(n1, 0) != (0, 0):
            n1 += 1
        self.n1, self.m = n1, self.Nc//n1
        self.nblocks = self.nb*self.m
        cells = {}
        for j in range(self.m):
            for i in range(n1):
                c = self.red(i, j)
                assert c not in cells
                cells[c] = j*n1+i
        assert len(cells) == self.Nc
        self.cells = cells
        self.cell_of = {v: k for k, v in cells.items()}
        self.tx = [self.site(*self._shift(t, 1, 0)) for t in range(self.N)]
        self.ty = [self.site(*self._shift(t, 0, 1)) for t in range(self.N)]
        self.ops = _point_group(kind)

    # ---------------------------------------------------------- geometry
    def red(self, x, y):
        f = self.Minv @ np.array([x, y], float)
        fl = np.floor(f+1e-9)
        r = np.array([x, y])-self.M @ fl.astype(np.int64)
        return int(r[0]), int(r[1])

    def site(self, s, x, y):
        """Index of basis site s in the cell at lattice coordinates (x, y)."""
        return s*self.Nc+self.cells[self.red(x, y)]

    def where(self, t):
        s, c = divmod(t, self.Nc)
        x, y = self.cell_of[c]
        return s, x, y

    def _shift(self, t, dx, dy):
        s, x, y = self.where(t)
        return s, x+dx, y+dy

    def position(self, t):
        """Fractional lattice coordinates of site t (cell + basis offset)."""
        s, x, y = self.where(t)
        return (x+self.basis[s][0], y+self.basis[s][1])

    def cartesian(self, t):
        p = self.position(t)
        return self.cart @ np.array([float(p[0]), float(p[1])])

    def _site_at(self, px, py):
        """Site at fractional lattice coordinates (exact Fractions)."""
        for s, (bx, by) in enumerate(self.basis):
            fx, fy = px-bx, py-by
            if fx.denominator == 1 and fy.denominator == 1:
                return self.site(s, int(fx), int(fy))
        raise ValueError('not a lattice site')

    def bonds(self, shell):
        """Pairs (a, b), a < b, at the shell-th smallest distance (1 = nearest neighbours), with periodic images."""
        L = 3
        dists = set()
        imgs = [self.M @ np.array([u, v]) for u in range(-L, L+1) for v in range(-L, L+1)]
        pos = [self.cartesian(t) for t in range(self.N)]
        cand = {}
        for a in range(self.N):
            for b in range(self.N):
                for I in imgs:
                    dvec = pos[b]+self.cart @ I-pos[a]
                    dd = round(float(np.hypot(*dvec)), 9)
                    if dd > 1e-9:
                        dists.add(dd)
                        if a < b:
                            cand.setdefault(dd, set()).add((a, b))
        shells = sorted(dists)
        return sorted(cand.get(shells[shell-1], set()))

    def heisenberg(self, *J):
        """Heisenberg model with couplings J[0] (nearest), J[1] (next shell), ... ."""
        m = edc.SpinHalfModel(self.N)
        for sh, Jv in enumerate(J, start=1):
            if Jv:
                for a, b in self.bonds(sh):
                    m.heisenberg(a, b, Jv)
        return m

    # ---------------------------------------------------------- symmetry
    def point_ops(self):
        """Point operations that map the cluster onto itself (and the lattice with its basis onto itself)."""
        ok = []
        for name, R in self.ops.items():
            if all(self.red(*(R @ self.M[:, c])) == (0, 0) for c in range(2)):
                try:
                    self.point_perm(name)
                    ok.append(name)
                except ValueError:
                    pass
        return ok

    def point_perm(self, name):
        R = self.ops[name]
        perm = []
        for t in range(self.N):
            px, py = self.position(t)
            qx = int(R[0, 0])*px+int(R[0, 1])*py; qy = int(R[1, 0])*px+int(R[1, 1])*py
            perm.append(self._site_at(qx, qy))
        assert sorted(perm) == list(range(self.N))
        return perm

    def momenta(self):
        """All allowed lattice-reduced momenta k = 2 pi M^{-T} n mod 2 pi."""
        G = 2*math.pi*np.linalg.inv(self.M.astype(float)).T
        ks = {}
        for a in range(-self.Nc, self.Nc+1):
            for b in range(-self.Nc, self.Nc+1):
                k = G @ np.array([a, b], float)
                key = tuple(round((v % (2*math.pi))/(2*math.pi)*self.Nc*4) % (self.Nc*4) for v in k)
                ks.setdefault(key, tuple(float(v % (2*math.pi)) for v in k))
                if len(ks) == self.Nc:
                    return sorted(ks.values())
        return sorted(ks.values())

    def act_k(self, name, k):
        """Image of a lattice-reduced momentum under a point operation: k -> R^{-T} k."""
        R = self.ops[name]
        return np.linalg.inv(R.astype(float)).T @ np.array(k, float)

    def little_group(self, k):
        out = []
        for name in self.point_ops():
            d = (self.act_k(name, k)-np.array(k))/(2*math.pi)
            if np.allclose(d, np.round(d), atol=1e-9):
                out.append(name)
        return sorted(out, key=lambda o: (o != 'E', o))

    def _mul(self, a, b):
        P = self.ops[a] @ self.ops[b]
        for n, R in self.ops.items():
            if np.array_equal(R, P):
                return n
        raise ValueError

    def irrep_table(self, k, allowed=None):
        """All one-dimensional irreps of the little group of k (or of its subgroup made of the `allowed` operations),
        as {name: {op: character}}; names 'X0', 'X1', ... with 'X0' the trivial irrep."""
        H = [o for o in self.little_group(k) if allowed is None or o in allowed]
        gens = []
        span = {'E'}
        for o in H:
            if o not in span:
                gens.append(o)
                span = {'E'}
                frontier = ['E']
                while frontier:
                    a = frontier.pop()
                    for g in gens:
                        c = self._mul(g, a)
                        if c not in span:
                            span.add(c); frontier.append(c)
        roots = [cmath.exp(2j*math.pi*q/12) for q in range(12)]
        found = []
        for vals in itertools.product(roots, repeat=len(gens)):
            chi = {'E': 1.0+0j}
            for g, v in zip(gens, vals):
                chi[g] = v
            frontier = list(chi)
            ok = True
            while frontier and ok:
                a = frontier.pop()
                for g in gens:
                    c = self._mul(g, a)
                    v = chi[g]*chi[a]
                    if c in chi:
                        if abs(chi[c]-v) > 1e-9:
                            ok = False; break
                    else:
                        chi[c] = v; frontier.append(c)
            if not ok or len(chi) != len(H):
                continue
            if all(abs(chi[self._mul(a, b)]-chi[a]*chi[b]) < 1e-9 for a in H for b in H):
                key = tuple(round(chi[o].real, 9)+1j*round(chi[o].imag, 9) for o in H)
                if key not in [f[0] for f in found]:
                    found.append((key, {o: _root12(chi[o]) for o in H}))
        found.sort(key=lambda f: (any(abs(v-1) > 1e-9 for v in f[0]), [(-v.real, -v.imag) for v in f[0]]))
        return {f'X{i}': f[1] for i, f in enumerate(found)}

    def rotations(self):
        return ['E']+[o for o in self.ops if o.startswith('R')]

    def group(self, k, irrep='X0', ops=None, allowed=None):
        table = self.irrep_table(k, allowed)[irrep]
        ops = ops if ops is not None else [o for o in self.little_group(k) if o in table]
        assert ops[0] == 'E'
        perms, chars = [], []
        for name in ops:
            P = self.point_perm(name)
            for j in range(self.m):
                for i in range(self.n1):
                    perm = []
                    for s in range(self.N):
                        t = P[s]
                        for _ in range(j):
                            t = self.ty[t]
                        for _ in range(i):
                            t = self.tx[t]
                        perm.append(t)
                    perms.append(perm)
                    chars.append(complex(table[name])*cmath.exp(-1j*(k[0]*i+k[1]*j)))
        assert len({tuple(p) for p in perms}) == len(perms)
        return perms, chars, ops
