#!/usr/bin/env python3
"""The end exchange of a saved state (edc_dmrg_mirror) against the permutation of the whole vector, on kagome cylinders
small enough for vectors: YC4 with 3 rings (16 spins), YC6 with 3 rings (24 spins) and YC4 with 4 rings (22 spins, an
even number of rings: the exchange needs no shift around the cylinder), momentum 0 and 1, a partly
converged (asymmetric) state and a converged ground state.  CPU engine, no GPU needed.

For every case: the image written as a vector must equal exp(i pi t K / c) P v (t = 2 x the shift), with P the site permutation of the end
exchange applied to the vector v of the source state (G|psi> = P v = exp(-i pi t K / c) |image>); its energy (from the engine's environments) must equal the
source's; a converged ground state whose exact level is single must be its own image up to a sign (overlap +-1 within 1e-3).  On
YC4 at K = 1 (K = -1 there) levels come in pairs (the two lowest are exactly degenerate at 16 spins) and the parity
checks are skipped.  The exact
eigenvectors carry their parity too: on YC6, S_z = 1, K = 1 the two lowest levels (0.00067 apart) are the even and the
odd member of one mirror pair.  A cylinder whose ends differ (a last single layer) has no end exchange and must be
refused."""
import numpy as np
import edc_dmrg as ed, edc_dmrg_state as st, edc_dmrg_vector as vec, edc_dmrg_mirror as mi


def permuted(v, codes, perm):
    t = np.zeros(len(codes), dtype=np.int64)
    for a, b in enumerate(perm):
        t |= ((codes >> a) & 1) << b
    w = np.zeros_like(v); w[np.searchsorted(codes, t)] = v
    return w


for Lx, c in ((3, 2), (3, 3), (4, 2)):
    lat = ed.kagome_cylinder(Lx, c); model = ed.Model(lat); N = lat.N; s, perm = mi.end_exchange(lat); tt = int(2*s)
    for nup, K in ((N//2, 0), (N//2+1, 1)):
        codes = vec.sector_states(N, nup); _, H = vec.sector_hamiltonian(lat, nup); w4, V4 = vec.exact_states(lat, nup, K, k=4)[:2]; wx, Vx = w4[:2], V4[:2]
        par = [complex(np.vdot(u, np.exp(1j*np.pi*tt*K/c)*permuted(u, codes, perm))) for u in Vx]
        print(f'YC{2*c} N={N} S_z={nup-N/2:+.0f} K={K}: exact levels {wx[0]:.10f} {wx[1]:.10f} {w4[2]:.10f}, mirror parities ' + ' '.join(f'{p.real:+.6f}' for p in par))
        # K = -K (YC4, K = 1): the reflection y -> -y keeps the sector too and does not commute with the end exchange, levels come
        # in pairs (exact at 16 spins), and Lanczos may return one member or a mixture: no parity check there
        paired = K != 0 and (2*K) % c == 0
        single = wx[1]-wx[0] > 1e-8 and not paired                     # the lowest level is not degenerate
        for i, p in enumerate(par):                                    # a level of its own has parity +-1 (a degenerate one need not)
            if not paired and (i == 0 or w4[i]-w4[i-1] > 1e-8) and w4[i+1]-w4[i] > 1e-8:
                assert abs(p.imag) < 1e-8 and abs(abs(p.real)-1) < 1e-8
        for label, sweeps in (('partly converged', 1), ('converged', 8)):
            eng = ed.DMRG(model, nup, K, seed=3)
            for t in range(sweeps):
                eng.sweep(16 if sweeps == 1 else 256, 1e-2 if t < sweeps-2 else 0.0, 1e-8 if sweeps > 1 else 1e-2, noise=1e-2 if t == 0 else 0.0)
            v = vec.to_vector(eng); E0 = vec.measure(v, H)[0]; img = mi.mirror(eng, model); w = vec.to_vector(img)
            target = np.exp(1j*np.pi*tt*K/c)*permuted(v, codes, perm); err = float(np.abs(w - target).max())
            Ei, _, _ = st.energy(img, img.mirror_of['centre']); Ew = vec.measure(w, H)[0]; ov = complex(np.vdot(v, w))
            sz = np.array(st.layer_sz(eng)); szi = np.array(st.layer_sz(img))
            print(f'YC{2*c} Lx={Lx} N={N} S_z={nup-N/2:+.0f} K={K} {label}: shift {s}; |image - e^(i pi t K/c) P v| = {err:.1e}; E {E0:.12f}, image {Ew:.12f} '
                  f'(environments {Ei:.12f}); <v|image> = {ov.real:+.6f}{ov.imag:+.6f}i; layer S_z reversed: {np.abs(szi - sz[::-1]).max():.1e}')
            assert err < 1e-10 and abs(Ew-E0) < 1e-10 and abs(Ei-E0) < 1e-9 and np.abs(szi - sz[::-1]).max() < 1e-10
            if label == 'converged' and single:
                assert abs(abs(ov) - 1) < 1e-3, 'a converged ground state should be its own image up to a phase'   # D = 256, 8 sweeps: on YC6 K = 1 the odd partner 0.00067 above keeps 2e-4
        if (Lx, c, K) == (3, 3, 1):
            assert par[0].real*par[1].real < 0, 'the close pair should be one even and one odd state'
try:
    mi.end_exchange(ed.kagome_cylinder(2, 3, tail=True)); raise SystemExit('a cylinder with different ends was accepted')
except ValueError:
    print('a cylinder with different ends is refused')
print('ok')
