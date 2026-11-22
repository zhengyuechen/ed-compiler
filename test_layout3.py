#!/usr/bin/env python3
"""Three-layout product (edc_layout3): against the full S_z = 0 space and against the sub engine; with a GPU also the
kernels with conversions on the fly against the table version. `test_layout3.py gpu` adds the GPU part."""
import sys
import numpy as np
import edc_cli as cli, edc_layout3 as l3

SPEC = "lattice kagome a1={a1} a2={a2}\ncouplings J1=1\nsector Sz=0 k=0,0 irrep=X0 flip=+1\nsolve engine={e} device={d}\n"
run = lambda **kw: cli.run(cli.parse(SPEC.format(**kw)))

lay = l3.Layout3((3, 0), (0, 2)); x = np.random.default_rng(1).standard_normal(lay.dim)
d = np.linalg.norm(lay.matvec_np(x)-lay.matvec_full(x))/np.linalg.norm(x)
assert d < 1e-13, d
a = run(a1='3,0', a2='0,2', e='layout3', d='cpu'); b = run(a1='3,0', a2='0,2', e='sub', d='cpu')
assert a['dim'] == b['dim'] == 2102 and abs(a['E']-b['E']) < 1e-9, (a['E'], b['E'])
for bad in (dict(a1='2,0', a2='0,2'),):                       # this cluster has rotations that exchange the sublattices
    try:
        run(e='layout3', d='cpu', **bad); raise AssertionError('accepted a cluster it does not cover')
    except ValueError:
        pass
msg = f'layout3: 18 sites against the full space {d:.1e}, energy against the sub engine {abs(a["E"]-b["E"]):.1e}, unsupported cluster refused'
if 'gpu' in sys.argv[1:]:
    import cupy as cp
    fly = l3.Layout3Fly((3, 0), (0, 2)); fly.to_gpu(); perm = lay.cidx_entry[lay._entry(2, *fly.rep_states())]
    xg = cp.asarray(x[perm]); yg = cp.empty_like(xg); fly.matvec_gpu(xg, yg); yt = lay.matvec_np(x)
    dg = float(np.linalg.norm(cp.asnumpy(yg)-yt[perm])/np.linalg.norm(yt)); assert dg < 1e-13, dg
    g = run(a1='3,0', a2='0,2', e='layout3', d='gpu'); assert abs(g['E']-b['E']) < 1e-9, (g['E'], b['E'])
    msg += f'; GPU kernels against the table version {dg:.1e}, GPU energy {abs(g["E"]-b["E"]):.1e}'
print(msg + ': ok')
