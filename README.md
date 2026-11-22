# ed-compiler

An operator compiler for exact diagonalisation. A model is written as terms on sites; the compiler finds the structure
of the bond list (families of bonds that act as a few shift-and-mask operations on the bit string), of the symmetry
group and of the Hilbert space, emits specialised C/OpenMP or CUDA code at run time, compiles it and drives a Lanczos
iteration with it.

## Contents

| File | What |
|---|---|
| `edc.py` | spin-1/2 models (exchange, zz, field, any bonds), Sz sectors, site-permutation groups at momentum zero; matrix-free, row-blocked and stored kernels; autotune; C/OpenMP |
| `edc_k.py` | translation symmetry at any momentum (complex), C/OpenMP |
| `edc_gpu.py` | CUDA versions: table engine (setup on the GPU), any momentum, memory-lean engines (bitmap rank, representatives on the fly), rectangle space groups |
| `edc_lattice.py` | square-lattice clusters with any periodic vectors (tilted included), point groups, momenta, 1D irreps, J1-J2 models |
| `edc_gpu_lean.py` | memory-lean GPU engine for any cluster and sector (momentum, 1D irrep, spin flip), table-based representative search, two-vector Lanczos, compiled observables |
| `edc_lattices.py` | triangular, honeycomb, kagome (and square) clusters; point groups C4v/C6v; all 1D irreps of any little group |
| `edc_sub.py` | spin engine with sublattice-table representatives and a compact index; C/OpenMP and CUDA; several GPUs of one node; other symmetric operators on the same tables (`operator`) |
| `edc_measure.py` | spin structure factor S(Q), M²(Q) of a sector's ground state on any lattice (Cartesian positions, star of Q) |
| `edc_cli.py`, `examples/` | plain-text model description (`lattice`, `model`, `couplings`, `bond`, `sector [all]`, `solve`, `measure`) for all lattices and engines |
| `edc_family.py` | H_A + λ H_B at many λ in one pass (parameter scans) |
| `edc_hubbard.py`, `edc_hubbard_gpu.py` | Fermi-Hubbard with fixed N↑, N↓, species-product kernel |
| `edc_hubbard_sym.py` | Fermi-Hubbard with translations, point group (1D irreps), any momentum and spin flip; representatives found species by species; one generated source for C/OpenMP and CUDA |
| `test_*.py`, `check_dispersion.py` | correctness checks (exact values, dense diagonalisation, independent projected Lanczos) |
| `bench/` | benchmark drivers (edc, XDiag, QuSpin), figure scripts |
| `vendor/libomp` | OpenMP runtime used on macOS (Homebrew bottle 23.1.6) |

## Running

macOS: Python 3.12 with numpy and scipy; clang from the Command Line Tools; the OpenMP runtime in `vendor/` is loaded
automatically. Linux (RIS): gcc with `-fopenmp`; CuPy (`cupy-cuda12x[ctk]`) for the GPU engines. Compiled libraries are
cached in `~/.cache/edc` (override with `EDC_CACHE`).

```bash
python test_edc.py
```

```bash
python bench/e2e_edc.py square:4x8:sym 8
```

```bash
python edc_cli.py examples/square4x4_j1j2.edc
```
