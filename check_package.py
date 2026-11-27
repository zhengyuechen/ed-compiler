#!/usr/bin/env python3
"""Check the package the way a user meets it: a set of files equal to the declared manifest, imported with nothing
else from this checkout on the path — or, with --installed, whatever `pip install` actually left behind.

    check_package.py [--installed] [--verbose]

1. `pyproject.toml`'s `py-modules` against the `edc*.py` files of this directory, in both directions.  A declared
   module with no file is a broken package; a file that is not declared is a module users do not get, which is how
   ten of the engines here went missing from a release once.
2. The declared modules copied alone into a temporary directory, with `vendor/` (the OpenMP runtime `edc.py` loads on
   macOS), and every one of them imported there in a fresh interpreter.  A module that needs a sibling the manifest
   does not declare fails at this step, and only at this step: in the checkout it imports fine.
3. `--installed`: the same import sweep over the modules next to the installed `edc`, which is what a user has after
   `pip install ed-compiler`.

The ten GPU modules import CuPy at module level and need the `[gpu]` extra (`pip install 'ed-compiler[gpu]'`).  A
`ModuleNotFoundError: cupy` for one of them is reported as GPU-only, not as a failure; any other error is.

Exit status is 0 only when the manifest and the import sweep agree with the files.  Needs no network and no build
backend: two seconds, safe to run in a pre-commit hook or before tagging a release.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def declared(path='pyproject.toml'):
    """The py-modules list of pyproject.toml, in file order.  Parsed rather than imported: no tomllib on 3.9, and the
    file is two tables with a known shape."""
    text = open(os.path.join(HERE, path)).read()
    m = re.search(r'py-modules\s*=\s*\[(.*?)\]', text, re.S)
    if not m:
        raise SystemExit('check_package: pyproject.toml has no py-modules list')
    return re.findall(r'"([^"]+)"', m.group(1))


def on_disk():
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HERE, 'edc*.py')))


def import_in(folder, modules, verbose=False):
    """Import each module in its own interpreter with `folder` on the path and nothing else.  Returns (ok, gpu_only,
    failures)."""
    env = dict(os.environ, PYTHONPATH=folder, PYTHONDONTWRITEBYTECODE='1',
               EDC_CACHE=os.path.join(folder, '.edc_cache'))
    ok, gpu, bad = [], [], []
    for m in modules:
        r = subprocess.run([sys.executable, '-c', 'import '+m], cwd=folder, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            ok.append(m)
            if verbose:
                print('  ok      ', m)
            continue
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
        if 'No module named' in tail and ("'cupy'" in tail or "'cupyx'" in tail):
            gpu.append(m)
            if verbose:
                print('  gpu-only', m)
        else:
            bad.append((m, tail))
            print('  FAILED  ', m, ':', tail)
    return ok, gpu, bad


def need_deps():
    """The sweep imports every module in a fresh interpreter of this same Python, so numpy and scipy have to be there.
    Say so instead of reporting 33 import failures."""
    missing = [m for m in ('numpy', 'scipy') if subprocess.run([sys.executable, '-c', 'import '+m], stdout=subprocess.DEVNULL,
                                                              stderr=subprocess.DEVNULL).returncode != 0]
    if missing:
        raise SystemExit('check_package: this Python has no ' + ' and '.join(missing)
                         + '; run it with the interpreter the package is installed in (or with --installed after pip install)')


def main():
    ap = argparse.ArgumentParser(description='manifest and import check for the published package')
    ap.add_argument('--installed', action='store_true', help='check the installed package instead of this checkout')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    need_deps()
    if a.installed:
        sys.path[:] = [p for p in sys.path if p and os.path.realpath(p) != os.path.realpath(HERE)]
        try:
            import edc
        except ImportError:
            raise SystemExit('check_package: no installed edc found from this Python (the script\'s own directory is '
                             'not on the path in this mode)')
        root = os.path.dirname(os.path.abspath(edc.__file__))
        mods = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(root, 'edc*.py')))
        print('installed package:', root, '--', len(mods), 'modules')
        ok, gpu, bad = import_in(root, mods, a.verbose)
        print(json.dumps(dict(installed=True, root=root, modules=len(mods), importable=len(ok),
                              gpu_only=gpu, failures=[m for m, _ in bad])))
        return 1 if bad else 0

    want, have = sorted(declared()), on_disk()
    problems = []
    for m in sorted(set(want)-set(have)):
        problems.append('declared but no file: '+m)
    for m in sorted(set(have)-set(want)):
        problems.append('file not declared: '+m)
    print('manifest:', len(want), 'declared,', len(have), 'files' + ('' if not problems else ' -- '+'; '.join(problems)))

    tmp = tempfile.mkdtemp(prefix='edc-pkg-')
    try:
        for m in want:
            shutil.copy2(os.path.join(HERE, m+'.py'), os.path.join(tmp, m+'.py'))
        for d in ('vendor', 'edc_vendor'):
            if os.path.isdir(os.path.join(HERE, d)):
                shutil.copytree(os.path.join(HERE, d), os.path.join(tmp, d))
                break
        ok, gpu, bad = import_in(tmp, want, a.verbose)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('import sweep:', len(ok), 'importable,', len(gpu), 'gpu-only (need the [gpu] extra),', len(bad), 'failed')
    if gpu:
        print('  gpu-only:', ' '.join(gpu))
    print(json.dumps(dict(declared=len(want), files=len(have), importable=len(ok), gpu_only=gpu,
                          failures=[m for m, _ in bad], manifest_problems=problems)))
    return 1 if (bad or problems) else 0


if __name__ == '__main__':
    sys.exit(main())
