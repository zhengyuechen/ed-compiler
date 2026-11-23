#!/usr/bin/env python3
"""The Ritz vector of a long Lanczos run when the replay does not round as the run did.
Kagome 27 sites, N_up = 15, J2 = 0.05, k = 0, translations only (1,931,652 states): 199 steps at rtol = 1e-8, by which
time the highest level has copies in the tridiagonal matrix.  The replay with other dot products (numpy instead of the
threaded sums of the generated library) gives a vector with a residual of 1e-2; ground_state_vector has to notice and
repair it.  Also: the library's dot product returns the same bits on every call."""
import sys, ctypes
import numpy as np
from scipy.linalg import eigh_tridiagonal
import edc_lattices as L, edc_hubbard_sym as hs, edc_sub

cl = L.LatticeCluster('kagome', (3, 0), (0, 3)); m = cl.heisenberg(1.0, 0.05)
e = edc_sub.SubEngine(m, 15, cl, (0.0, 0.0), 'X0', 0, device='cpu', point_ops=['E'])
E, it, _ = e.ground_energy(tol=1e-12, rtol=1e-8)
A, B = np.array(e._lanczos['A']), np.array(e._lanczos['B']); top = eigh_tridiagonal(A, B[:len(A)-1], select='i', select_range=(len(A)-2, len(A)-1))[0]
assert abs(top[1]-top[0]) < 1e-9, 'expected a copy of the highest level in this run'
rec = dict(e._lanczos)
psi, Er, res = hs.ground_state_vector(e); assert res < 1e-6 and abs(Er-E) < 1e-10, (res, Er-E)


class OtherRounding:                                              # the same engine without the library's vector routines
    def __init__(s, e): s.__dict__['e'] = e
    def __getattr__(s, k):
        if k == 'lib': raise AttributeError(k)
        return getattr(s.e, k)
    def __setattr__(s, k, v): setattr(s.e, k, v)


h = OtherRounding(e)
e._lanczos = dict(rec); bad = hs.ground_state_vector(h, refine=0)
e._lanczos = dict(rec); good = hs.ground_state_vector(h)
assert bad[2] > 1e-4, f'the replay with other rounding was expected to fail without the check (residual {bad[2]:.1e})'
assert good[2] < 1e-6 and abs(good[1]-E) < 1e-10, (good[2], good[1]-E)
x = np.random.default_rng(0).standard_normal(2*e.dim if not e.real else e.dim); P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
e.lib.vdot_re.restype = ctypes.c_double; d = {e.lib.vdot_re(P(x), P(x), ctypes.c_int64(x.size)).hex() for _ in range(40)}
assert len(d) == 1, 'the dot product of the library is not reproducible'
print(f'test_replay: {it} steps, copy of the highest level present; replay with other rounding: residual {bad[2]:.1e} without the check, '
      f'{good[2]:.1e} with it (energy error {good[1]-E:+.1e}); library dot product reproducible: ok')
