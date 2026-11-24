#!/usr/bin/env python3
"""Smallest use of the GPU back end, version 1: one state of a kagome YC cylinder on one GPU, its
energy and the magnetisation of every layer.  Nearest-neighbour couplings only (no J2 in this back end).

    OPENBLAS_NUM_THREADS=1 python example_v1.py Lx c D nup K [tail]      e.g.  example_v1.py 7 3 1200 31 0

Run with OPENBLAS_NUM_THREADS=1: with more BLAS threads the small host factorisations between device calls are many
times slower on some hosts.  check_small.py compares this back end with exact diagonalisation on an
18-spin cylinder (lowest S_z = 0 and S_z = 1 levels and their difference); run it once on the machine first:
    python check_small.py out.json gpu1"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, edc_dmrg as ed, edc_dmrg_gpu as eg


def layer_sz(eng, D, nup):
    """<S_z> of every layer from the weights of the bond labels (the label of a bond is the number of up spins to its
    left).  Moves the centre from the first layer to the last without truncating; call it after the run."""
    n = eng.n; mleft = []
    for r in range(n):
        p = eng.plan[r]; b = eng.bond[r]; M = eng.M[r]; w = np.array([np.sum(M[p.Moff[i]:p.Moff[i+1]]**2) for i in range(p.nL)])
        mleft.append(float((b.m*w).sum()/w.sum()))
        if r < n-1:
            eng.move_right(r, D, 0.0)
    mleft.append(float(nup)); sizes = np.array([len(y) for y in eng.model.lat.layers])
    return np.diff(mleft) - sizes/2.0


if __name__ == '__main__':
    Lx, c, D, nup, K = (int(v) for v in sys.argv[1:6]); tail = len(sys.argv) > 6 and sys.argv[6] == 'tail'
    lat = ed.kagome_cylinder(Lx, c, tail=tail); model = ed.Model(lat)                 # layers: rings of 2c spins, layers of c spins
    E, eng, rows = eg.run(model, D, nup=nup, K=K, cls=eg.DMRGGPU1b)                  # two warm-up sweeps, nine at D; prints every sweep
    print(f'{lat.name}: N = {lat.N}, S_z = {nup - lat.N/2:g}, K = {K}, D = {D}: E = {E:.10f}, {rows[-1]["elapsed"]:.1f} s')
    print('<S_z> per layer:', ' '.join(f'{x:+.3f}' for x in layer_sz(eng, D, nup)))
