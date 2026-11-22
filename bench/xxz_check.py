# XXZ in the model language against a dense diagonalisation without any lattice symmetry (4x4 torus, S_z = 0 and 1).
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.getcwd()); import edc_cli
Lx = Ly = 4; N = 16
bonds = [(x+Lx*y, (x+1) % Lx+Lx*y) for x in range(Lx) for y in range(Ly)]+[(x+Lx*y, x+Lx*((y+1) % Ly)) for x in range(Lx) for y in range(Ly)]
def dense_E0(nup, delta):
    st = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(N), nup)], dtype=np.int64); st.sort(); n = len(st); H = np.zeros((n, n))
    for i, j in bonds:
        bi, bj = (st >> i) & 1, (st >> j) & 1; H[np.arange(n), np.arange(n)] += delta*np.where(bi == bj, 0.25, -0.25)
        m = np.flatnonzero(bi != bj); H[m, np.searchsorted(st, st[m] ^ ((1 << i) | (1 << j)))] += 0.5
    return float(np.linalg.eigvalsh(H)[0])
worst = 0
for delta in (0.4, 1.0, 1.7):
    for sz in (0, 1):
        spec = edc_cli.parse(f"lattice rect Lx=4 Ly=4\ncouplings J1=1 Delta={delta}\nsector Sz={sz} all\nsolve device=cpu\n")
        r = edc_cli.run(spec); E = min((x['E'] if not isinstance(x['E'], list) else x['E'][0]) for x in r['sectors'] if x['E'] is not None)
        ref = dense_E0(8+sz, delta); worst = max(worst, abs(E-ref)); print('Delta %.1f Sz %d: engine %.10f dense %.10f' % (delta, sz, E, ref))
print('largest difference %.1e' % worst)
