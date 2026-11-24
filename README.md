# ed-compiler

An operator compiler for exact diagonalisation. A model is written as terms on sites; the compiler finds the structure
of the bond list (families of bonds that act as a few shift-and-mask operations on the bit string), of the symmetry
group and of the Hilbert space, emits specialised C/OpenMP or CUDA code at run time, compiles it and drives a Lanczos
iteration with it.

## Quick start

Python 3.9 or later and a C compiler with OpenMP: clang from the Command Line Tools on macOS (`xcode-select --install`;
the OpenMP runtime comes with this package), gcc on Linux.

```bash
git clone https://github.com/zhengyuechen/ed-compiler
```

```bash
python -m pip install ./ed-compiler
```

This installs the command `edc`, which works from any directory. A model is a few lines of text:

```bash
cat > kagome12.edc <<'EOF'
lattice kagome a1=2,0 a2=0,2
couplings J1=1
sector Sz=0 k=0,0 irrep=X2 flip=+1
solve device=cpu
EOF
```

```bash
edc kagome12.edc
```

It prints one JSON line. The first run compiles the kernel (6 s on an Apple M2 Pro); later runs take it from
`~/.cache/edc` (0.3 s):

```json
{"lattice": "kagome", "N": 12, "model": "heisenberg", "device": "cpu", "engine": "sub", "nup": 6, "k_over_pi": [0.0, 0.0], "irrep": "X2", "flip": 1, "dim": 10, "E": -5.4448752169718535, "iterations": 10, "seconds": 0.23}
```

| | |
|---|---|
| Lattice | kagome, 12 sites: 2 × 2 unit cells of 3 sites |
| Boundary | periodic in both directions: a torus with periods 2 a1 and 2 a2 (a1, a2 at 60°) |
| Hamiltonian | H = J1 Σ S_i · S_j over the 24 nearest-neighbour bonds, spin 1/2, J1 = 1 |
| Sector | S_z = 0, momentum (0, 0), irrep X2 (−1 under the sixfold rotation, +1 under the mirror S0), even under spin flip: 10 states |
| Reference | E0 = −5.444 875 216 — Läuchli, Sudan, Sørensen, PRB 83, 212401 (2011), Table 1, N = 12 |
| Tolerance | \|E − E0\| < 1e-9, one unit of the last published digit; `seconds` and the last digits of `E` vary |

The same number comes out of a dense diagonalisation of the 924 states with S_z = 0 that uses none of this code:
−5.444 875 216 972 (`test_quickstart.py`, agreement to 1e-10). With `sector Sz=0 all` in place of the sector line every
symmetry block of S_z = 0 is solved; the lowest level is the same and the block sizes add up to 924.

`edc --help` prints the model language; `examples/` holds more models (square J1-J2, scans, Hubbard, DMRG on cylinders).

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
| `edc_layout3.py` | kagome, nearest-neighbour Heisenberg, sector k = 0 / trivial irrep / spin-flip even on clusters with inversion only: one product through three layouts of the vector (the spectator sublattice of each bond family is fixed first); `solve engine=layout3`; 2.3 times the on-the-fly sub engine on one H100 at 36 sites |
| `edc_glayout.py` | spin-1/2 models on kagome, triangular and square clusters, any symmetry block (momentum, 1D irrep, spin flip, any S_z; XXZ, further-neighbour bonds): the product through layouts of the vector. The sites are split into parts that the group permutes (three sublattices, or the four sublattices of a 2 x 2 cell); the bonds of a pair of parts act as one operator on every column of the remaining (spectator) parts, and the result is folded back with one lookup per pair of the orbit. `solve engine=layout`; C/OpenMP and CUDA; same interface as `edc_sub.SubEngine`. One H100, 36 to 40 sites: 3 to 37 times the on-the-fly sub engine, as fast as its stored connectivity with a fifth of the memory; with less device memory it stores fewer tables (`fold=`, `expand=`) |
| `edc_hubbard.py`, `edc_hubbard_gpu.py` | Fermi-Hubbard with fixed N↑, N↓, species-product kernel |
| `edc_hubbard_sym.py` | Fermi-Hubbard with translations, point group (1D irreps), any momentum and spin flip; representatives found species by species; one generated source for C/OpenMP and CUDA |
| `test_*.py`, `check_dispersion.py` | correctness checks (exact values, dense diagonalisation, independent projected Lanczos) |
| `bench/` | benchmark drivers (edc, XDiag, QuSpin), figure scripts |
| `vendor/libomp` | OpenMP runtime used on macOS (Homebrew bottle 23.1.6) |

## GPU DMRG on kagome cylinders

Install the GPU dependencies with `python -m pip install '.[gpu]'` on a CUDA 12 system.
From the repository root, run the fixed-rule GPU solver and its small-cylinder check:

```sh
OPENBLAS_NUM_THREADS=1 python bench/dmrg_gpu/check_small.py check.json gpu3
OPENBLAS_NUM_THREADS=1 python bench/dmrg_gpu/bench60.py 1200 gpu3 30 0 energy.json
```

Use `gpu3s` to benchmark the optional block factorisation selector. The `edc` model-file
DMRG command uses the CPU backend. The following commands profile CuPy allocations and
measure magnetisation and bond energies on a five-ring YC8 cylinder:

```sh
python bench/dmrg_gpu/mem_profile.py 5 4 2400 28 3 mem.json
python bench/kagome_gap/kg_run.py . 5 4 800 yc8.json sz=0,1
```

Memory profiling changes execution time. Use a separate run for timing.
The experimental total-spin and momentum implementation is in `bench/dmrg_su2/`.

## Running

macOS: Python 3.12 with numpy and scipy; clang from the Command Line Tools; the OpenMP runtime in `vendor/` is loaded
automatically. Linux: gcc with `-fopenmp`; CuPy (`cupy-cuda12x[ctk]`) for the GPU engines. Compiled libraries are
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
