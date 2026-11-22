#!/usr/bin/env python3
"""General test: a parameter scan from a few ground states (reduced basis / eigenvector continuation).
J1-J2 Heisenberg model on an Lx x Ly torus, S_z engine, sector K = (0,0), A1, flip +1; H(J2) = H1 + J2 H2.
Training points are added greedily where the residual of the reduced solution is largest; every other point of the
scan costs no product at all (H1 and H2 applied once to each training vector).  Compared with a Lanczos run per point.
Usage: reduced_basis_scan.py Lx Ly npoints [tolerance]"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, scipy.linalg as sl, edc_lattice as el, edc_sub
from edc_hubbard_sym import ground_state_vector
Lx, Ly, npts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-8
cl = el.Cluster((Lx, 0), (0, Ly)); grid = np.linspace(0.05, 1.0, npts)
e = edc_sub.SubEngine(cl.j1j2(1.0, 0.5), cl.N//2, cl, (0.0, 0.0), 'A1', 1, device='cpu'); n = int(e.dim); products = [0]
def H12(v):
    y1 = np.zeros(n); y2 = np.zeros(n)
    e.retune(cl.j1j2(1.0, 1.0)); e.matvec(v, y1); e.retune(cl.j1j2(1.0, 2.0)); e.matvec(v, y2); products[0] += 2
    return 2*y1-y2, y2-y1                                         # H1 v, H2 v
exact = []; its = 0
for j2 in grid:                                                  # reference: one Lanczos run per point
    e.retune(cl.j1j2(1.0, float(j2))); r = e.ground_energy(tol=1e-12); exact.append(float(r[0])); its += int(r[1])
exact = np.array(exact)
V, HV1, HV2 = [], [], []; train = []; train_its = 0
def add(j2):
    global train_its
    e.retune(cl.j1j2(1.0, float(j2))); r = e.ground_energy(tol=1e-12); train_its += int(r[1]); psi = np.array(ground_state_vector(e)[0], dtype=float)
    for u in V:
        psi -= np.dot(u, psi)*u
    psi /= np.linalg.norm(psi); h1, h2 = H12(psi); V.append(psi); HV1.append(h1); HV2.append(h2); train.append(float(j2))
def reduced():
    Vm = np.array(V); A1 = Vm @ np.array(HV1).T; A2 = Vm @ np.array(HV2).T; E = []; R = []
    for j2 in grid:
        w, c = sl.eigh((A1+A1.T)/2+j2*(A2+A2.T)/2); c0 = c[:, 0]
        res = c0 @ (np.array(HV1)+j2*np.array(HV2))-w[0]*(c0 @ Vm); E.append(w[0]); R.append(np.linalg.norm(res))
    return np.array(E), np.array(R)
add(grid[0]); add(grid[-1]); hist = []
while True:
    E, R = reduced(); err = np.abs(E-exact); hist.append((len(V), float(err.max()), float(R.max())))
    Rm = R.copy(); Rm[[int(np.argmin(np.abs(grid-t))) for t in train]] = 0.0       # trained points are not candidates again
    if err.max() < tol or len(V) >= 30 or Rm.max() == 0.0:
        break
    add(grid[int(np.argmax(Rm))])
print(json.dumps(dict(torus=f'{Lx}x{Ly}', dim=n, points=npts, lanczos_products_all_points=its, training_vectors=len(V), training_points=sorted(train),
                      training_products=train_its+products[0], largest_energy_error=float(err.max()), largest_residual=float(R.max()),
                      history=[(a, float('%.2e' % b), float('%.2e' % c)) for a, b, c in hist])))
