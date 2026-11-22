"""Shared model definitions for the benchmarks (same lattices in edc and in the XDiag scripts)."""
import edc

def build(name):
    kind, *a = name.split(':')
    if kind == 'ring':        # ring:L[:sym]
        L = int(a[0]); m = edc.ring(L); g = edc.ring_translations(L)
    elif kind == 'j1j2':      # j1j2:L:J2[:sym]
        L = int(a[0]); m = edc.ring(L, 1.0, float(a[1])); g = edc.ring_translations(L); a = a[1:]
    elif kind == "squarepg":  # squarepg:L  (translations x D4, A1)
        L = int(a[0]); m = edc.square(L, L); g = edc.square_space_group(L); a = [a[0], "sym"]
    elif kind == "square":    # square:LxxLy[:sym]
        Lx, Ly = map(int, a[0].split('x')); m = edc.square(Lx, Ly); g = edc.torus_translations(Lx, Ly)
    else:
        raise ValueError(name)
    sym = len(a) > 1 and a[1] == 'sym'
    return m, (g if sym else None)
