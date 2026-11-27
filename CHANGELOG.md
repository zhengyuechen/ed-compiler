# Changelog

Selected changes, newest first. Dates are commit dates. Add new entries above the most recent dated section while a change is unreleased.

## Unreleased

- Added `check_package.py`: the declared module list against the files (both directions), and every declared module imported from a directory holding nothing else, or with `--installed` from the installed package. It reports the ten GPU modules as needing the `[gpu]` extra instead of failing on their `cupy` import.
- Added a [contributor guide](AGENTS.md) with entry points, reference implementations and focused checks. Claude Code loads the same guide through `CLAUDE.md`.

## 2026-11-24

- Added grouped GPU DMRG products and the mirror-state subspace solvers. Grouped and excited-state solves use the engine instance's product. See `c1d7858` and [the two-level regression](test_dmrg_grouped_states.py).

## 2026-11-23

- Added GPU layer DMRG, saved-state reload and additional states in one symmetry sector. See `d26c330`, `e44f762` and [the saved-state checks](test_dmrg_states.py).
- Added the installable `edc` command and a tested kagome quick start (`c4f819e`).

## 2026-11-22

- Added the layout engine for symmetry blocks on kagome, triangular and square lattices (`18e054f`). See [the layout comparison](test_glayout.py).
