#!/usr/bin/env python3
"""DMRG engine (edc_dmrg.py) against exact energies of small cylinders, square and kagome, in several momentum and
magnetisation sectors, and through the language.  Exact values: product basis with a momentum projector
(edc_dmrg.exact_levels; the square ones were first obtained with explore/dmrg/exact_k.py)."""
import json, subprocess, sys, os
import edc_dmrg as ed

SQUARE = [  # Lx, Ly, K, J2, D, exact, allowed error
    (3, 4, 0, 0.0, 64, -7.5301371690, 2e-9), (3, 4, 1, 0.0, 64, -5.4305983466, 2e-9), (3, 4, 2, 0.0, 64, -6.8717242318, 2e-9),
    (2, 6, 0, 0.0, 64, -7.0132460189, 2e-9), (2, 6, 2, 0.0, 64, -5.3369552407, 2e-9),
    (6, 4, 0, 0.0, 400, -15.7284509715, 1e-7), (6, 4, 1, 0.0, 400, -13.5839032684, 2e-5), (6, 4, 0, 0.5, 400, -12.4399615002, 1e-7),
    (4, 6, 0, 0.0, 256, -15.0253709741, 5e-4), (4, 6, 3, 0.0, 256, -14.6385503476, 2e-3)]
for Lx, Ly, K, J2, D, exact, allow in SQUARE:
    E, eng, info = ed.ground_state(ed.Model(ed.square_cylinder(Lx, Ly, J2=J2)), D, K=K)
    err = E-exact
    print(f'square {Lx} x {Ly}, K = {K}, J2 = {J2}, D = {D}: E = {E:.10f}, error {err:.1e} (allowed {allow:.0e}), {info["time"]:.2f} s')
    assert -1e-9 < err < allow, 'energy outside the allowed window (it must also not fall below the exact value)'
m1 = ed.Model(ed.square_cylinder(6, 4)); m0 = ed.Model(ed.square_cylinder(6, 4), momentum=False)
E1 = ed.ground_state(m1, 100)[0]; E0 = ed.ground_state(m0, 100)[0]
print(f'square 6 x 4, D = 100, labels (S_z, k): {E1:.8f}; labels S_z only: {E0:.8f}')
assert abs(E1-E0) < 2e-3 and E1 > -15.7284509715-1e-9 and E0 > -15.7284509715-1e-9

# kagome: the layer construction is the lattice built from the unit cell, and every (S_z, momentum) sector is exact
for Lx, c, Delta, nups in ((3, 2, 1.0, (8, 7, 5, 4)), (3, 2, 0.7, (8, 5)), (2, 3, 1.0, (7, 6, 4)), (2, 4, 0.7, (10, 7)), (4, 2, 1.0, (11, 8))):
    lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat, Delta=Delta); cells = ed.kagome_cylinder_cells(Lx, c)
    for nup in nups:
        ex = ed.exact_levels(lat.global_bonds(), lat.N, nup, Delta, tperm=lat.translation(), c=c)
        assert abs(min(ex.values())-ed.exact_levels(cells, lat.N, nup, Delta)) < 1e-8, 'layer construction differs from the unit-cell construction'
        worst = 0.0
        for K, e in ex.items():
            E = ed.ground_state(model, 400, nup=nup, K=K)[0]; worst = max(worst, abs(E-e))
            assert abs(E-e) < 2e-8, f'kagome {Lx} rings, c = {c}, nup = {nup}, K = {K}: {E} against {e}'
        print(f'{lat.name}, Delta = {Delta}, {nup} up spins: {len(ex)} momentum sectors, largest error {worst:.1e}')
# near saturation the kagome flat band gives exact eigenstates that no coupling term connects to the ground state
# (a magnon with alternating signs inside one ring); every start must still reach the lowest level
lat = ed.kagome_cylinder(2, 3, tail=True); model = ed.Model(lat)
for nup in (17, 16):
    ex = ed.exact_levels(lat.global_bonds(), lat.N, nup, 1.0, tperm=lat.translation(), c=3)
    for K, e in ex.items():
        for seed in (0, 1, 2):
            assert abs(ed.ground_state(model, 100, nup=nup, K=K, seed=seed)[0]-e) < 1e-8, (nup, K, seed)
print(f'{lat.name}: one and two flipped spins, three starts each, every momentum: exact')
rows = ed.magnetisation(model, 200, sz2=(0, 6, 16, 18))
assert abs(min(rows[1][1].values())-(-6.23677714)) < 1e-7 and abs(min(rows[2][1].values())-4.5) < 1e-8 and abs(min(rows[3][1].values())-7.5) < 1e-9
here = os.path.dirname(os.path.abspath(__file__))
for name, exact, allow in (('dmrg_cylinder_6x4_k.edc', -11.0792516158, 2e-5), ('dmrg_kagome_yc6.edc', None, None)):
    out = json.loads(subprocess.check_output([sys.executable, os.path.join(here, 'edc_cli.py'), os.path.join(here, 'examples', name)]))
    print('language:', out['lattice'], '| Sz', out['Sz'], '| k', out['k'], '| E =', out['energy'])
    if exact is not None:
        assert -1e-9 < out['energy']-exact < allow
    else:
        lat = ed.kagome_cylinder(2, 3); ex = ed.exact_levels(lat.global_bonds(), lat.N, 6, 0.7, tperm=lat.translation(), c=3)
        assert abs(out['energy']-ex[1]) < 2e-8, (out['energy'], ex)
print('test_dmrg: all checks passed')
