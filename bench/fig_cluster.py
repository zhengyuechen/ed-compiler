#!/usr/bin/env python3
"""Picture of a kagome cluster of edc_lattices: a, the sites and nearest-neighbour bonds inside the Wigner-Seitz cell of
the cluster's superlattice (centred on a hexagon of the lattice; open circles lie on the boundary and are the same
site as the open circle on the opposite edge), periodic copies in grey, the two spanning vectors, and for one site its
four nearest (J1) and four second (J2) neighbours; b, the momenta of the cluster in the Brillouin zone.
Usage: fig_cluster.py <out.png> [A1x,A1y,A2x,A2y]   (default 2,2,-2,4: the 36-site cluster 36d)"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import edc_lattices as L, edc_measure as em

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 8,
                     'legend.fontsize': 7, 'legend.frameon': False, 'savefig.dpi': 300})
out = sys.argv[1]
a = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else '2,2,-2,4').split(',')]
cl = L.LatticeCluster('kagome', (a[0], a[1]), (a[2], a[3])); N = cl.N
C = np.array(cl.cart, float)                                     # columns: lattice vectors a1, a2
A1, A2 = C @ np.array([a[0], a[1]], float), C @ np.array([a[2], a[3]], float)
pos = np.array([cl.cartesian(t) for t in range(N)], float)
dnn = min(np.linalg.norm(pos[i]-pos[j]) for i, j in cl.bonds(1) if np.linalg.norm(pos[i]-pos[j]) > 1e-9)
shifts = [n1*A1+n2*A2 for n1 in range(-3, 4) for n2 in range(-3, 4)]
allpts = np.array([p+s for p in pos for s in shifts]); owner = np.array([t for t in range(N) for s in shifts])
# centre: a point with six sites at the nearest-neighbour distance (a hexagon of the lattice) closest to the centroid
cands = [0.5*(p+q) for p in pos for q in allpts if abs(np.linalg.norm(p-q)-2*dnn) < 1e-6]
cands = [c for c in cands if sum(abs(np.linalg.norm(allpts-c, axis=1)-dnn) < 1e-6) == 6]
c0 = min(cands, key=lambda c: (round(np.linalg.norm(c-pos.mean(axis=0)), 6), round(c[0], 6), round(c[1], 6)))
allpts -= c0


def ws_class(p, eps=1e-7):
    """1 inside the Wigner-Seitz cell of the superlattice, 2 on its boundary, 0 outside."""
    d0 = np.linalg.norm(p)
    dm = min(np.linalg.norm(p-s) for s in shifts if np.linalg.norm(s) > 1e-9)
    return 1 if d0 < dm-eps else (2 if d0 < dm+eps else 0)


cls = np.array([ws_class(p) for p in allpts])
# every site of the cluster is once inside, or two or three times on the boundary (opposite edges are identified)
assert (cls == 1).sum()+len(set(owner[cls == 2])) == N, ((cls == 1).sum(), len(set(owner[cls == 2])))
inside = cls > 0
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axs[0]
R = 1.25*np.linalg.norm(A1)
near = np.linalg.norm(allpts, axis=1) < R
idx = np.where(near)[0]
P = allpts[idx]; ins = inside[idx]
for i, j in itertools.combinations(range(len(P)), 2):
    if abs(np.linalg.norm(P[i]-P[j])-dnn) < 1e-6:
        both = ins[i] and ins[j]
        ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], '-', color='#222222' if both else '#c8c8c8', lw=1.1 if both else 0.6, zorder=2 if both else 1)
ax.plot(P[~ins][:, 0], P[~ins][:, 1], 'o', color='#c8c8c8', ms=2.6, zorder=2)
cl_ = cls[idx]
ax.plot(P[cl_ == 1][:, 0], P[cl_ == 1][:, 1], 'o', color='#222222', ms=4.0, zorder=4)
ax.plot(P[cl_ == 2][:, 0], P[cl_ == 2][:, 1], 'o', color='white', mec='#222222', mew=1.0, ms=4.0, zorder=4)
# Wigner-Seitz boundary: hexagon with corners at the circumcentres of neighbouring superlattice triangles
nbr = sorted((s for s in shifts if 1e-9 < np.linalg.norm(s) < 1.01*np.linalg.norm(A1)), key=lambda s: math.atan2(s[1], s[0]))
if len(nbr) == 6:
    corners = [(nbr[k]+nbr[(k+1) % 6])/3 for k in range(6)]
    ax.add_patch(plt.Polygon(corners, closed=True, fill=False, ec='#1f4e9c', lw=0.9, ls='--', zorder=3))
for v in (A1, A2):
    ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='#1f4e9c', lw=0.9, shrinkA=0, shrinkB=0), zorder=5)
# one site with its J1 and J2 neighbours
target = -0.20*A1-0.12*A2
t0 = min(np.where(cl_ == 1)[0], key=lambda i: (round(np.linalg.norm(P[i]-target), 3), round(P[i][0], 3), round(P[i][1], 3)))
d = np.linalg.norm(P-P[t0], axis=1)
second = sorted({round(x, 6) for x in d if x > dnn+1e-6})[0]
for sel, col, lab in ((abs(d-dnn) < 1e-6, '#b5482a', '$J_1$'), (abs(d-second) < 1e-6, '#2a8a5a', '$J_2$')):
    for q in P[sel]:
        ax.plot([P[t0][0], q[0]], [P[t0][1], q[1]], '-', color=col, lw=1.6, zorder=3, solid_capstyle='round')
    ax.plot([], [], '-', color=col, lw=1.6, label=lab)
ax.plot(*P[t0], 'o', color='#b5482a', mec='white', ms=5.6, mew=0.8, zorder=6)
ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.93*R, 0.93*R); ax.set_ylim(-0.88*R, 0.88*R)
ax.legend(loc='lower left', handlelength=1.4, bbox_to_anchor=(0.0, 0.0))
# momenta
ax = axs[1]
b1, b2 = em.reciprocal(cl)
Minv = np.linalg.inv(np.array([[a[0], a[1]], [a[2], a[3]]], float))     # rows: A_i in units of a_j
G = [g1*b1+g2*b2 for g1 in range(-2, 3) for g2 in range(-2, 3)]
ks = []
for m1 in range(-2*cl.Nc, 2*cl.Nc+1):
    for m2 in range(-2*cl.Nc, 2*cl.Nc+1):
        coef = Minv @ np.array([m1, m2], float)                           # k = coef . (b1, b2), k.A_i = 2 pi m_i
        k = coef[0]*b1+coef[1]*b2
        d0 = np.linalg.norm(k)
        if all(d0 <= np.linalg.norm(k-g)+1e-9 for g in G if np.linalg.norm(g) > 1e-9):
            ks.append(k)
kk = []
for k in ks:                                                              # one representative per equivalence class
    if not any(any(np.allclose(k-q, g, atol=1e-7) for g in G) for q in kk):
        kk.append(k)
assert len(kk) == cl.Nc, len(kk)
gn = sorted((g for g in G if np.linalg.norm(g) > 1e-9), key=lambda g: np.linalg.norm(g))[:6]
gn = sorted(gn, key=lambda g: math.atan2(g[1], g[0]))
bz = [(gn[k]+gn[(k+1) % 6])/3 for k in range(6)]
ax.add_patch(plt.Polygon(bz, closed=True, fill=False, ec='#222222', lw=0.8))
Kn = np.linalg.norm(bz[0]); Mn = np.linalg.norm(gn[0])/2
seen = set()
for k in ks:                                                              # draw every copy on the zone boundary too
    n = np.linalg.norm(k)
    if n < 1e-9:
        kind, col, mk = '$\\Gamma$', '#1f4e9c', 'o'
    elif abs(n-Kn) < 1e-7:
        kind, col, mk = 'K', '#b5482a', 'o'
    elif abs(n-Mn) < 1e-7 and any(abs(np.dot(k, g)/np.linalg.norm(g)-Mn) < 1e-7 for g in gn):
        kind, col, mk = 'M', '#2a8a5a', 's'
    else:
        kind, col, mk = 'Q', '#8a8a8a', 'D'
    ax.plot(k[0], k[1], mk, color=col, ms=5.0, mec='white', mew=0.6, zorder=3, label=None if kind in seen else kind)
    seen.add(kind)
ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.25*Kn, 1.25*Kn); ax.set_ylim(-1.3*Kn, 1.3*Kn)
h, l = ax.get_legend_handles_labels(); order = [l.index(x) for x in ('$\\Gamma$', 'K', 'M', 'Q') if x in l]
ax.legend([h[i] for i in order], [l[i] for i in order], loc='upper right', bbox_to_anchor=(1.12, 1.02), handletextpad=0.1)
for a_, lab in zip(axs, 'ab'):
    a_.text(0.0, 0.98, lab, transform=a_.transAxes, fontweight='bold', fontsize=9)
fig.tight_layout(); fig.savefig(out)
print('written', out, '| sites', N, 'cells', cl.Nc, '| |A1| = %.4f, |A2| = %.4f, angle %.1f deg' % (np.linalg.norm(A1), np.linalg.norm(A2), math.degrees(math.acos(np.dot(A1, A2)/np.linalg.norm(A1)/np.linalg.norm(A2)))),
      '| momenta', len(kk), {k_: sum(1 for k in kk if (np.linalg.norm(k) < 1e-9) == (k_ == 'G') and (abs(np.linalg.norm(k)-Kn) < 1e-7) == (k_ == 'K')) for k_ in ('G', 'K')})
