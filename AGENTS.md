# Working on ed-compiler

## Find the implementation to extend

Start with the [README](README.md) for the user entry point and the [changelog](CHANGELOG.md) for changes to existing behavior. Follow one real call from a CLI example or benchmark through the solver selection to the kernel. Read the matching test and the most recent change to that path. The host implementation gives a numerical reference; the production GPU implementation shows which work is cached, batched or kept on device.

| If you are changing | Follow this path | Check against |
|---|---|---|
| Model syntax or dispatch | [examples](examples/) → [edc_cli.py](edc_cli.py) → the selected engine | [CLI tests](test_cli.py), [12-site quick start](test_quickstart.py) |
| ED iteration or vector reconstruction | [Lanczos and replay](edc_hubbard_sym.py), their callers, and [the lean GPU engine](edc_gpu_lean.py) | [Replay tests](test_replay.py), [two-vector GPU checks](test_2v_gpu.py) |
| Finite-cylinder DMRG or grouped products | [60-site driver](bench/dmrg_gpu/bench60.py) → [host DMRG](edc_dmrg.py) or [ProductG](edc_dmrg_gpu3g.py) | [Small-system checks](bench/dmrg_gpu/check_small.py), [saved-state tests](test_dmrg_states.py) |

For example, before changing a grouped product, find where the driver selects it, then find who constructs and reuses its plan:

```sh
rg -n 'ProductG|DMRGGPU3g' edc_dmrg*.py bench/dmrg_gpu/
git log -5 --oneline -- edc_dmrg_gpu3g.py
```

## Make and check the change

- Check the branch and working files before editing. Keep unrelated changes and other contributors' staged files intact. If the index and working tree disagree unexpectedly, compare the affected files directly with `git show HEAD:path` before drawing conclusions.
- Extend the existing call path and reuse its layout, symmetry labels and solver interfaces. For a new GPU operation, inspect the fast engine as well as the host reference. If another implementation fits this problem better, record the correctness comparison and timing that led to the choice.
- Exercise the actual entry point on a small known case. For solver changes, compare energies and relevant sectors or residuals. For speed or memory claims, compare the same state, parameters and hardware and record the command and code revision.
- Add an entry to [CHANGELOG.md](CHANGELOG.md) for a change to behavior, solver selection, defaults, reproducibility or the developer workflow. State what a caller will observe and point to the test or commit. Distinguish completed checks from queued checks.
