#!/usr/bin/env python3
"""Equal-time correlations of local operators on a kagome cluster and their decomposition by lattice symmetry.

Four sets of local operators O_a (cluster = edc_lattices.LatticeCluster('kagome', A1, A2), N sites):
  'z'   S^z_i                 on every site                                  (N of them)
  'zz'  S^z_i S^z_j           on every nearest-neighbour bond                (2N)
  'b'   S_i . S_j             on every nearest-neighbour bond                (2N)
  'x'   S_i . (S_j x S_k)     on every elementary triangle, anticlockwise    (2N/3)

For a state (or the average over a degenerate level) the connected matrix

    C_ab = Re <O_a O_b> - <O_a> <O_b>

is invariant under the space group of the cluster.  A real pattern c (sum_a c_a^2 = 1) has the variance
c.C.c = <(sum_a c_a (O_a - <O_a>))^2>, and the eigenvectors of C are the patterns that fluctuate independently.
They are sorted by momentum and by the irreducible representation of the little group (modes()).  A state with
long-range order in a pattern has one eigenvalue that grows in proportion to the number of operators; without order
every eigenvalue stays of the size of the on-site variance, the mean eigenvalue trace(C)/n.

Names.  Momentum: G (little group C6v), K (C3v), M (C2v), Q (one mirror only: the star of six of the 27- and 36-site
clusters; 'Y' in Raikos et al. 2026).  Irreps: C6v A1 A2 B1 B2 E1 E2 with B1 even under the three mirrors whose axis
passes through sites of the central hexagon (axis along a lattice vector) and B2 even under the other three; C3v
A1 A2 E; C2v A1 A2 B1 B2 = (C2, mirror along the momentum, mirror across it) = (+,+,+), (+,-,-), (-,+,-), (-,-,+);
one mirror: e, o.  The chirality is a pseudoscalar: a mirror maps it to minus itself, and the labels include that sign."""
import math
import numpy as np


# ------------------------------------------------------------------ geometry
def geometry(cl):
    """Sites, nearest-neighbour bonds and elementary triangles with unwrapped Cartesian coordinates.
    Returns dict(pos (N,2) wrapped positions, bonds [(i, j)], bond_xy [(2,2) end points], tris [(i, j, k)] anticlockwise,
    tri_xy [(3,2)], tri_kind [0 or 1])."""
    assert cl.kind == 'kagome'
    C = cl.cart
    def xy(s, X, Y):
        return C @ np.array([X+float(cl.basis[s][0]), Y+float(cl.basis[s][1])])
    tris, tri_xy, tri_kind = [], [], []
    for c in range(cl.Nc):
        x, y = cl.cell_of[c]
        for kind, members in enumerate((((0, x, y), (1, x, y), (2, x, y)), ((0, x, y+1), (1, x+1, y), (2, x, y)))):
            P = np.array([xy(*m) for m in members])
            cen = P.mean(axis=0)
            order = np.argsort(np.arctan2(P[:, 1]-cen[1], P[:, 0]-cen[0]))
            tris.append(tuple(cl.site(*members[o]) for o in order))
            tri_xy.append(P[order]); tri_kind.append(kind)
    bonds, bond_xy, seen = [], [], {}
    for t, P in zip(tris, tri_xy):
        for u in range(3):
            i, j = t[u], t[(u+1) % 3]
            key = (min(i, j), max(i, j))
            assert key not in seen, 'a nearest-neighbour pair is joined twice on this cluster'
            seen[key] = len(bonds); bonds.append(key)
            bond_xy.append(np.array([P[u], P[(u+1) % 3]]) if i < j else np.array([P[(u+1) % 3], P[u]]))
    assert sorted(bonds) == sorted(cl.bonds(1)), 'triangle edges are not the nearest-neighbour bonds'
    pos = np.array([cl.cartesian(t) for t in range(cl.N)])
    return dict(pos=pos, bonds=bonds, bond_xy=bond_xy, tris=tris, tri_xy=tri_xy, tri_kind=tri_kind)


def cart_op(cl, name):
    return cl.cart @ cl.ops[name].astype(float) @ np.linalg.inv(cl.cart)


def classify(cl, name):
    """('E'|'C2'|'C3'|'C6', None) for rotations, ('s', axis angle in degrees mod 180) for mirrors."""
    R = cart_op(cl, name)
    if np.linalg.det(R) > 0:
        ang = round(math.degrees(math.atan2(R[1, 0], R[0, 0]))) % 360
        return {0: 'E', 180: 'C2', 120: 'C3', 240: 'C3', 60: 'C6', 300: 'C6'}[ang], None
    return 's', (0.5*math.degrees(math.atan2(R[1, 0], R[0, 0]))) % 180.0


def space_group(cl):
    """Every element as (site permutation p with p[a] = image of site a, lattice translation (i, j), point operation)."""
    out = []
    for name in cl.point_ops():
        P = cl.point_perm(name)
        for j in range(cl.m):
            for i in range(cl.n1):
                perm = []
                for s in range(cl.N):
                    t = P[s]
                    for _ in range(j):
                        t = cl.ty[t]
                    for _ in range(i):
                        t = cl.tx[t]
                    perm.append(t)
                out.append((perm, (i, j), name))
    assert len({tuple(g[0]) for g in out}) == len(out)
    return out


class OpSpace:
    """One set of local operators with the action of the space group on its index: image[g][a] and a sign."""

    def __init__(self, cl, kind, geo=None, group=None):
        geo = geo or geometry(cl)
        self.cl, self.kind, self.geo = cl, kind, geo
        self.group = group or space_group(cl)
        if kind == 'z':
            self.items = [(i,) for i in range(cl.N)]
        elif kind in ('zz', 'b'):
            self.items = list(geo['bonds'])
        elif kind == 'x':
            self.items = list(geo['tris'])
        else:
            raise ValueError(kind)
        self.n = len(self.items)
        index = {frozenset(it): a for a, it in enumerate(self.items)}
        assert len(index) == self.n
        self.image, self.sign = [], []
        for perm, _, name in self.group:
            self.image.append(np.array([index[frozenset(perm[s] for s in it)] for it in self.items]))
            det = np.linalg.det(cl.ops[name].astype(float))
            self.sign.append(-1.0 if (kind == 'x' and det < 0) else 1.0)

    def average(self, A):
        """Mean of D(g) A D(g)^T over the space group (A a matrix or a vector of one-point values)."""
        A = np.asarray(A, float)
        out = np.zeros_like(A)
        for m, s in zip(self.image, self.sign):
            if A.ndim == 2:
                out[np.ix_(m, m)] += A
            else:
                out[m] += s*A
        return out/len(self.image)

    def rep(self, gi):
        """Matrix D(g): D e_a = sign e_{g(a)}."""
        D = np.zeros((self.n, self.n))
        D[self.image[gi], np.arange(self.n)] = self.sign[gi]
        return D


# ------------------------------------------------------------------ irreducible representations
TABLES = {
    12: ('G', {'A1': dict(E=1, C2=1, C3=1, C6=1, ss=1, sb=1), 'A2': dict(E=1, C2=1, C3=1, C6=1, ss=-1, sb=-1),
               'B1': dict(E=1, C2=-1, C3=1, C6=-1, ss=1, sb=-1), 'B2': dict(E=1, C2=-1, C3=1, C6=-1, ss=-1, sb=1),
               'E1': dict(E=2, C2=-2, C3=-1, C6=1, ss=0, sb=0), 'E2': dict(E=2, C2=2, C3=-1, C6=-1, ss=0, sb=0)}),
    6: ('K', {'A1': dict(E=1, C3=1, s=1), 'A2': dict(E=1, C3=1, s=-1), 'E': dict(E=2, C3=-1, s=0)}),
    4: ('M', {'A1': dict(E=1, C2=1, sl=1, st=1), 'A2': dict(E=1, C2=1, sl=-1, st=-1),
              'B1': dict(E=1, C2=-1, sl=1, st=-1), 'B2': dict(E=1, C2=-1, sl=-1, st=1)}),
    2: ('Q', {'e': dict(E=1, s=1), 'o': dict(E=1, s=-1)}),
    1: ('k', {'a': dict(E=1)}),
}


def k_cart(cl, k):
    """Cartesian wave vector of a lattice-reduced momentum, shortest representative."""
    B = 2*math.pi*np.linalg.inv(cl.cart).T
    q0 = np.linalg.inv(cl.cart).T @ np.array(k, float)
    best = min((q0+B @ np.array([u, v], float) for u in range(-2, 3) for v in range(-2, 3)), key=lambda q: round(float(q @ q), 9))
    return best


def stars(cl):
    """Momentum stars: list of dict(name, k (representative), size, little (operations), cls {op: class key}, table)."""
    def key(k):
        return tuple(round((float(v) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for v in k)
    ops = cl.point_ops()
    seen, out = set(), []
    for k in cl.momenta():
        if key(k) in seen:
            continue
        orbit = {key(cl.act_k(o, k)) for o in ops}
        seen |= orbit
        lg = cl.little_group(k)
        name, table = TABLES[len(lg)]
        q = k_cart(cl, k)
        cls = {}
        for o in lg:
            c, ang = classify(cl, o)
            if c == 's':
                if len(lg) == 12:
                    c = 'ss' if abs(((ang+30) % 60)-30) < 1e-6 else 'sb'
                elif len(lg) == 4:
                    ax = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
                    c = 'sl' if abs(ax[0]*q[1]-ax[1]*q[0]) < 1e-6 else 'st'
            cls[o] = c
        assert set(cls.values()) == set(next(iter(table.values())).keys()), (name, cls)
        out.append(dict(name=name, k=tuple(float(v) for v in k), size=len(orbit), little=lg, cls=cls, table=table))
    assert sum(s['size'] for s in out) == cl.Nc
    return out


def modes(space, C, tol=1e-9):
    """Eigenvalues of the invariant matrix C by momentum star and irrep.
    Returns a list of dict(star, irrep, dim (states per eigenvalue = star size x irrep dimension), values (descending),
    pattern (real, normalised eigenvector of the largest value, for drawing))."""
    cl, n = space.cl, space.n
    C = np.asarray(C, float)
    by = {}
    for gi, (_, ij, name) in enumerate(space.group):
        by.setdefault(name, []).append((gi, ij))
    D = {name: [(space.rep(gi), ij) for gi, ij in lst] for name, lst in by.items()}
    out, total = [], 0
    for st in stars(cl):
        k = st['k']
        Pq = sum(np.exp(1j*(k[0]*i+k[1]*j))*Dg for Dg, (i, j) in D['E'])/cl.Nc
        for ir, chi in st['table'].items():
            d = chi['E']
            Pp = np.zeros((n, n), complex)
            for o in st['little']:
                Do = next(Dg for Dg, ij in D[o] if ij == (0, 0))
                Pp += chi[st['cls'][o]]*Do
            P = Pq @ (Pp*(d/len(st['little'])))
            assert np.allclose(P @ P, P, atol=1e-9), 'not a projector'
            rank = int(round(np.trace(P).real))
            total += rank*st['size']
            if rank == 0:
                continue
            Mx = P.conj().T @ C @ P
            w, U = np.linalg.eigh((Mx+Mx.conj().T)/2)
            w, U = w[::-1][:rank], U[:, ::-1][:, :rank]
            assert all(abs(w[i]-w[i-i % d]) < tol*max(1.0, abs(w[0])) for i in range(rank)), 'degeneracy does not match the irrep'
            u = U[:, 0]
            u = u*np.exp(-1j*np.angle(u[np.argmax(np.abs(u))]))
            pat = u.real/np.linalg.norm(u.real)
            out.append(dict(star=st['name'], irrep=ir, dim=st['size']*d, values=[float(x) for x in w[::d]],
                            pattern=[float(x) for x in pat]))
    assert total == n, (total, n)
    return out


def summary(space, C):
    """Trace, mean eigenvalue and the modes sorted by their largest eigenvalue."""
    ms = modes(space, C)
    tr = float(np.trace(C))
    assert abs(sum(m['dim']*sum(m['values']) for m in ms)-tr) < 1e-8*max(1.0, abs(tr))
    ms.sort(key=lambda m: -m['values'][0])
    return dict(n=space.n, trace=tr, mean=tr/space.n, modes=ms)


# ------------------------------------------------------------------ patterns of a given symmetry
def kind_projectors(space):
    """{'G:A1': P, ...}: real symmetric projectors on all patterns of one kind (every member of the star, every row)."""
    cl, n = space.cl, space.n
    D0 = {name: space.rep(gi) for gi, (_, ij, name) in enumerate(space.group) if ij == (0, 0)}
    DT = [(space.rep(gi), ij) for gi, (_, ij, name) in enumerate(space.group) if name == 'E']
    def key(k):
        return tuple(round((float(v) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for v in k)
    out = {}
    for st in stars(cl):
        k = st['k']
        Pq = sum(np.exp(1j*(k[0]*i+k[1]*j))*Dg for Dg, (i, j) in DT)/cl.Nc
        members = {}
        for o in cl.point_ops():
            members.setdefault(key(cl.act_k(o, k)), o)
        for ir, chi in st['table'].items():
            Pp = sum(chi[st['cls'][o]]*D0[o] for o in st['little'])*(chi['E']/len(st['little']))
            P = Pq @ Pp
            tot = sum(D0[o] @ P @ D0[o].T for o in members.values())
            assert np.max(np.abs(tot.imag)) < 1e-9
            if round(np.trace(tot).real) > 0:
                out[f"{st['name']}:{ir}"] = tot.real
    assert abs(sum(np.trace(P) for P in out.values())-n) < 1e-8
    return out


def content(space, pattern, kinds=None):
    """Weight of a pattern in every kind (sums to |pattern|^2)."""
    kinds = kinds or kind_projectors(space)
    c = np.asarray(pattern, float)
    return {kd: float(c @ P @ c) for kd, P in kinds.items() if c @ P @ c > 1e-10}


def element(space, name, centre=(0, 0)):
    """Index of the space-group element 'point operation `name` about the point `centre`' (lattice coordinates, exact
    fractions), e.g. a threefold rotation about a triangle centre (1/3, 1/3)."""
    from fractions import Fraction as Fr
    cl = space.cl
    R = cl.ops[name]
    c = (Fr(centre[0]), Fr(centre[1]))
    perm = []
    for t in range(cl.N):
        p = cl.position(t)
        d = (Fr(p[0])-c[0], Fr(p[1])-c[1])
        perm.append(cl._site_at(int(R[0, 0])*d[0]+int(R[0, 1])*d[1]+c[0], int(R[1, 0])*d[0]+int(R[1, 1])*d[1]+c[1]))
    lookup = {tuple(g[0]): gi for gi, g in enumerate(space.group)}
    return lookup[tuple(perm)]


def subgroup(space, gens):
    """Indices of the elements generated by the elements `gens`."""
    perms = [space.group[g][0] for g in range(len(space.group))]
    lookup = {tuple(p): gi for gi, p in enumerate(perms)}
    ident = lookup[tuple(range(space.cl.N))]
    H, frontier = {ident}, [ident]
    while frontier:
        a = frontier.pop()
        for g in gens:
            pa, pg = perms[a], perms[g]
            c = lookup[tuple(pg[pa[s]] for s in range(space.cl.N))]
            if c not in H:
                H.add(c); frontier.append(c)
    return sorted(H)


def best_invariant(space, C, H):
    """The pattern of largest variance among those invariant under the subgroup H (element indices) and orthogonal to
    the patterns invariant under the whole group: (variance, pattern, number of independent patterns)."""
    PH = sum(space.rep(g) for g in H)/len(H)
    PG = sum(space.rep(g) for g in range(len(space.group)))/len(space.group)
    Q = PH-PG
    w, U = np.linalg.eigh(Q @ np.asarray(C, float) @ Q)
    return float(w[-1]), U[:, -1], int(round(np.trace(Q)))


def symmetry_classes(space):
    """Subgroups that stand for kinds of crystal: (label, element indices), built from a unit cell (the translations
    that keep it) and the point operations kept about a hexagon centre or a triangle centre.
    Cells: 27 sites (translations of the 3 x 3 superlattice; the whole 27-site cluster), 9 sites (sqrt3 x sqrt3) and
    the 3-site cell of the lattice.  A cell that does not fit the cluster is left out."""
    from fractions import Fraction as Fr
    cl = space.cl
    names = cl.point_ops()
    cls = {n: classify(cl, n) for n in names}
    ms = [n for n in names if cls[n][0] == 's' and abs(((cls[n][1]+30) % 60)-30) < 1e-6]      # axis through sites of the hexagon
    mb = [n for n in names if cls[n][0] == 's' and n not in ms]                                # axis through bond centres
    trans = [(gi, g[1]) for gi, g in enumerate(space.group) if g[2] == 'E']
    def keep(ks):
        return [gi for gi, (i, j) in trans if all(abs(np.exp(1j*(k[0]*i+k[1]*j))-1) < 1e-9 for k in ks)]
    G = 2*math.pi*np.linalg.inv(cl.M.astype(float)).T
    def is_momentum(k):
        d = np.linalg.solve(G, np.array(k, float))
        return np.allclose(d, np.round(d), atol=1e-9)
    third = [(2*math.pi*a/3, 2*math.pi*b/3) for a in range(3) for b in range(3)]               # momenta of the 3 x 3 superlattice
    kk = [(4*math.pi/3, 2*math.pi/3), (2*math.pi/3, 4*math.pi/3)]                              # the two K points: (2 b1 + b2)/3, (b1 + 2 b2)/3
    cells = []
    if all(is_momentum(k) for k in third):
        cells.append(('27-site cell', keep(third)))
    if all(is_momentum(k) for k in kk):
        cells.append(('9-site cell', keep(kk)))
    cells.append(('3-site cell', [gi for gi, _ in trans]))
    hexc, tri = (0, 0), (Fr(1, 3), Fr(1, 3))
    def el(n, c=hexc):
        return element(space, n, c)
    points = [('no point symmetry', []), ('one mirror through sites', [el(ms[0])]), ('one mirror through bond centres', [el(mb[0])]),
              ('C2', [el('R3')]), ('C2 and two mirrors', [el('R3'), el(ms[0])]),
              ('C3 about a hexagon', [el('R2')]), ('C3 about a triangle', [el('R2', tri)]),
              ('C3 and mirrors through sites, hexagon', [el('R2'), el(ms[0])]),
              ('C3 and mirrors through bond centres, hexagon', [el('R2'), el(mb[0])]),
              ('C3 and mirrors, triangle', [el('R2', tri), el(mb[0], tri)]),
              ('C6 about a hexagon', [el('R1')]), ('C6 and all mirrors', [el('R1'), el(ms[0])])]
    out = []
    for cn, T in cells:
        for pn, gens in points:
            out.append((f'{cn}, {pn}', subgroup(space, list(T)+gens)))
    return out


def class_table(space, C, kinds=None):
    """For every symmetry class: order of the subgroup, number of independent patterns (beyond the uniform one), the
    largest variance of such a pattern, and the kinds it is made of."""
    kinds = kinds or kind_projectors(space)
    rows, seen = [], {}
    for label, H in symmetry_classes(space):
        val, pat, npat = best_invariant(space, C, H)
        if npat == 0:
            continue
        rows.append(dict(label=label, order=len(H), patterns=npat, value=val,
                         content={k: round(w, 4) for k, w in sorted(content(space, pat, kinds).items(), key=lambda x: -x[1])[:3]}))
    return rows
