#!/usr/bin/env python3
"""Square-lattice clusters with periodic vectors a1, a2 (any tilt), their J1-J2 models, translation generators,
point groups, momenta and one-dimensional irreps.

Sites are numbered along the orbits of the one-step translation tx: site index j * n1 + i is the point tx^i ty^j (0),
0 <= i < n1 = order of tx, 0 <= j < m = N / n1.  With this numbering tx is a rotation inside blocks of n1 bits, which the
compiler writes as two shift-and-mask operations.
"""
import cmath, math
import numpy as np
import edc

# point operations of the square lattice as integer matrices, with names
D4 = {'E': ((1, 0), (0, 1)), 'C4': ((0, -1), (1, 0)), 'C2': ((-1, 0), (0, -1)), 'C4^3': ((0, 1), (-1, 0)),
      'Mx': ((-1, 0), (0, 1)), 'My': ((1, 0), (0, -1)), 'Md': ((0, 1), (1, 0)), "Md'": ((0, -1), (-1, 0))}
# characters of the one-dimensional irreps of D4 and C4 (restrictions are used for subgroups)
IRREPS = {
    'A1': {'E': 1, 'C4': 1, 'C2': 1, 'C4^3': 1, 'Mx': 1, 'My': 1, 'Md': 1, "Md'": 1},
    'A2': {'E': 1, 'C4': 1, 'C2': 1, 'C4^3': 1, 'Mx': -1, 'My': -1, 'Md': -1, "Md'": -1},
    'B1': {'E': 1, 'C4': -1, 'C2': 1, 'C4^3': -1, 'Mx': 1, 'My': 1, 'Md': -1, "Md'": -1},
    'B2': {'E': 1, 'C4': -1, 'C2': 1, 'C4^3': -1, 'Mx': -1, 'My': -1, 'Md': 1, "Md'": 1},
    'E+': {'E': 1, 'C4': 1j, 'C2': -1, 'C4^3': -1j},          # C4 only
    'E-': {'E': 1, 'C4': -1j, 'C2': -1, 'C4^3': 1j},
    # the two irreps of the D2 subgroups {E, C2, Mx, My} and {E, C2, Md, Md'} with C2 = -1 (no C4 in the little group)
    'Bx': {'E': 1, 'C2': -1, 'Mx': 1, 'My': -1, 'Md': 1, "Md'": -1},
    'By': {'E': 1, 'C2': -1, 'Mx': -1, 'My': 1, 'Md': -1, "Md'": 1},
}


class Cluster:
    def __init__(self, a1, a2):
        self.M = np.array([a1, a2], dtype=np.int64).T            # columns are the periodic vectors
        self.N = int(abs(round(np.linalg.det(self.M))))
        self.Minv = np.linalg.inv(self.M.astype(float))
        n1 = 1
        while self.red(n1, 0) != (0, 0):
            n1 += 1
        self.n1, self.m = n1, self.N//n1
        self.coord = [self.red(i, j) for j in range(self.m) for i in range(n1)]
        self.index = {c: k for k, c in enumerate(self.coord)}
        assert len(self.index) == self.N
        self.tx = [self.site(x+1, y) for (x, y) in self.coord]
        self.ty = [self.site(x, y+1) for (x, y) in self.coord]

    def red(self, x, y):
        f = self.Minv @ np.array([x, y], float)
        fl = np.floor(f+1e-9)
        r = np.array([x, y])-self.M @ fl.astype(np.int64)
        return int(r[0]), int(r[1])

    def site(self, x, y):
        return self.index[self.red(x, y)]

    def j1j2(self, J1=1.0, J2=0.0):
        m = edc.SpinHalfModel(self.N)
        for s, (x, y) in enumerate(self.coord):
            m.heisenberg(s, self.site(x+1, y), J1)
            m.heisenberg(s, self.site(x, y+1), J1)
            if J2:
                m.heisenberg(s, self.site(x+1, y+1), J2)
                m.heisenberg(s, self.site(x+1, y-1), J2)
        return m

    def point_ops(self):
        """Names of the D4 operations that map the cluster onto itself."""
        ok = []; seen = set()
        for name, R in D4.items():
            R = np.array(R)
            if all(self.red(*(R @ self.M[:, c])) == (0, 0) for c in range(2)):
                perm = tuple(self.site(*(R @ np.array(c))) for c in self.coord)
                if perm in seen:                              # a chain (one row): several operations move the sites alike; keep one
                    continue
                seen.add(perm); ok.append(name)
        return ok

    def point_perm(self, name):
        R = np.array(D4[name])
        return [self.site(*(R @ np.array(c))) for c in self.coord]

    def momenta(self):
        """All allowed k = 2 pi M^{-T} n, reduced to [0, 2 pi)^2, as (kx, ky) pairs."""
        G = 2*math.pi*np.linalg.inv(self.M.astype(float)).T
        ks = {}
        for a in range(-self.N, self.N+1):
            for b in range(-self.N, self.N+1):
                k = G @ np.array([a, b], float)
                key = (round((k[0] % (2*math.pi))/(2*math.pi)*self.N*4) % (self.N*4),
                       round((k[1] % (2*math.pi))/(2*math.pi)*self.N*4) % (self.N*4))
                ks.setdefault(key, (k[0] % (2*math.pi), k[1] % (2*math.pi)))
                if len(ks) == self.N:
                    return sorted(ks.values())
        return sorted(ks.values())

    def little_group(self, k):
        """Point operations that leave k invariant modulo reciprocal vectors of the cluster."""
        G = 2*math.pi*np.linalg.inv(self.M.astype(float)).T
        out = []
        for name in self.point_ops():
            Rk = np.array(D4[name], float) @ np.array(k)
            n = np.linalg.solve(G, Rk-np.array(k))       # (Rk - k) in units of the cluster reciprocal basis
            if np.allclose(n, np.round(n), atol=1e-9) and np.allclose(np.round(n) % 1, 0):
                n2 = np.linalg.solve(2*math.pi*np.eye(2), Rk-np.array(k))   # must be a reciprocal vector of Z^2
                if np.allclose(n2, np.round(n2), atol=1e-9):
                    out.append(name)
        return out

    def group(self, k, irrep='A1', ops=None):
        """Full list of site permutations g = T(i ex + j ey) o P and characters chi(g) = chi_P(P) e^{-i k.(i ex + j ey)}."""
        ops = ops if ops is not None else [o for o in self.little_group(k) if o in IRREPS[irrep]]
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
                    chars.append(complex(IRREPS[irrep][name])*cmath.exp(-1j*(k[0]*i+k[1]*j)))
        assert len({tuple(p) for p in perms}) == len(perms)
        return perms, chars, ops


def tilted(n, m_):
    """Tilted square cluster with periodic vectors (n, m) and (-m, n); N = n^2 + m^2."""
    return Cluster((n, m_), (-m_, n))
