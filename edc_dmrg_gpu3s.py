#!/usr/bin/env python3
"""A small selector for the GPU layer DMRG: in a move between layers, which route factors a block.

Version 3 (edc_dmrg_gpu3.DMRGGPU3) uses one fixed rule: a block A (m x n) with n >= 48 and m >= 4 n goes through the
device (Cholesky QR 2 there, the small square factor decomposed on the host), every other block goes to the host for
an ordinary SVD.  DMRGGPU3s keeps everything else of version 3 and chooses, per shape class, among three routes:
    host   the block is copied to the host, LAPACK SVD, vectors copied back
    devh   Cholesky QR 2 with the tall products on the device, Cholesky and the SVD of the n x n factor on the host
    devd   all on the device (Cholesky QR 2 with cuSOLVER, eigenproblem of C C^T)
Shape class: (floor(log2 n), floor(log2(m / n))) for n >= 16 and m >= n; smaller or wide blocks stay on the host and
are not searched (the bound of the search).

The choice is measured in the run itself, on its own blocks: while a class is undecided, the blocks of that class in
one move go together through one route, timed to the point where the vectors and the carried factor are ready on the
device (so the transfers of a route are inside its time), and the result is used like any other.  Two timed batches
per route, the smaller time per unit of work (sum of m n^2) counts.  The host route is skipped for a class when its
rate in the nearest measured class predicts more than three times the best device route (an ordinary SVD of the
heaviest blocks would cost seconds).  So the tuning costs only the difference between a tried route and the best
one, and it is inside the total time of the run that tunes.
The table is stored as JSON (DMRG_SELECT_CACHE=path) with a key of the device, the CPU, the thread setting, the
library versions and the source; a later run with the same key starts with the choices and tunes only classes it
has not seen.  Stored times are a record of the tuning run, not a promise about the next one."""
import os, json, math, time, hashlib, platform
import numpy as np
import cupy as cp
import edc_dmrg_gpu3 as e3

ROUTES = ('devh', 'devd', 'host')


def _cpu():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _key(trials, min_cols, prune):
    h = hashlib.sha1()
    for f in ('edc_dmrg_gpu3.py', 'edc_dmrg_gpu3s.py', 'edc_dmrg_factor.py'):
        h.update(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), 'rb').read())
    return dict(device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cpu=_cpu(), blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                numpy=np.__version__, cupy=getattr(cp, '__version__', 'stand-in'), source=h.hexdigest()[:16], trials=trials, min_cols=min_cols, prune=prune)


class Selector:
    def __init__(self, path=None, trials=2, min_cols=16, prune=3.0):
        self.path = path; self.trials = trials; self.min_cols = min_cols; self.prune = prune; self.key = _key(trials, min_cols, prune)
        self.table = {}; self.loaded = 0; self.status = 'no cache file given'
        self.trial_seconds = 0.0; self.trial_excess = 0.0; self.trial_batches = 0; self.trial_blocks = 0; self._log = []
        if path:
            self.status = 'miss: no file'
            if os.path.exists(path):
                d = json.load(open(path))
                if d.get('key') == self.key:
                    self.table = {tuple(int(x) for x in k.split(',')): v for k, v in d['table'].items()}
                    self.loaded = sum(1 for v in self.table.values() if v['choice']); self.status = f'hit: {self.loaded} classes'
                else:
                    self.status = 'miss: key differs (' + ', '.join(k for k in self.key if d.get('key', {}).get(k) != self.key[k]) + ')'

    def bucket(self, shape):
        m, n = shape
        if n < self.min_cols or m < n:
            return None
        return (int(math.log2(n)), min(6, int(math.log2(m/n))))

    def _near_host_rate(self, b):
        c = [(abs(k[0]-b[0]) + abs(k[1]-b[1]), min(v['rates']['host'])) for k, v in self.table.items() if v['rates']['host']]
        return min(c)[1] if c else None

    def route(self, b):
        """(route, timed): the decided route of class b, or the next route to time."""
        e = self.table.setdefault(b, dict(choice=None, rates={r: [] for r in ROUTES}, skipped=[]))
        if e['choice']:
            return e['choice'], False
        for r in ('devh', 'devd'):
            if len(e['rates'][r]) < self.trials:
                return r, True
        if len(e['rates']['host']) < self.trials and 'host' not in e['skipped']:
            pred = self._near_host_rate(b); best = min(min(e['rates'][r]) for r in ('devh', 'devd'))
            if e['rates']['host'] or pred is None or pred <= self.prune*best:
                return 'host', True
            e['skipped'].append('host')
        e['choice'] = min((min(v), r) for r, v in e['rates'].items() if v)[1]
        return e['choice'], False

    def record(self, b, r, seconds, work, nblocks):
        self.table[b]['rates'][r].append(seconds/work); self.trial_seconds += seconds; self.trial_batches += 1; self.trial_blocks += nblocks
        self._log.append((b, seconds, work))

    def report(self):
        for b, s, w in self._log:                                    # time above what the finally best route of the class needs
            best = min(min(v) for v in self.table[b]['rates'].values() if v); self.trial_excess += max(0.0, s - best*w)
        self._log = []
        return dict(status=self.status, classes=len(self.table), decided=sum(1 for v in self.table.values() if v['choice']),
                    timed_batches=self.trial_batches, timed_blocks=self.trial_blocks, seconds_in_timed_batches=self.trial_seconds,
                    seconds_above_best_route=self.trial_excess,
                    choices={f'{2**k[0]}<=n<{2**(k[0]+1)}, {2**k[1]}<=m/n' + (f'<{2**(k[1]+1)}' if k[1] < 6 else ''): v['choice'] or 'undecided' for k, v in sorted(self.table.items())})

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            json.dump(dict(key=self.key, note='rates: seconds per unit of work (m n^2) in the run that tuned; a record, not a promise',
                           table={f'{k[0]},{k[1]}': v for k, v in sorted(self.table.items())}), open(self.path, 'w'), indent=1)


class DMRGGPU3s(e3.DMRGGPU3):
    version = 'gpu-v3s'

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.selector = Selector(os.environ.get('DMRG_SELECT_CACHE'))

    def _ready(self, route, blocks):
        """One route for a few blocks, each carried to the point where U (m x n) and s V^T (n x n) are on the device."""
        out = {}
        for q, A in blocks.items():
            r = (self._factor_device({q: A}) if route == 'devd' else self._factor_host({q: A}) if route == 'devh' else self._host_svds({q: A}))[q]
            if r[0] == 'host':
                _, s, Us, Vh = r; out[q] = ('ready', s, cp.asarray(Us), cp.asarray(s[:, None]*Vh))
            elif r[0] == 'devh':
                _, s, Q, Uc, Vh = r; out[q] = ('ready', s, Q @ cp.asarray(Uc), cp.asarray(s[:, None]*Vh))
            else:
                _, s, Q1, T2, Uc, C = r; out[q] = ('ready', s, Q1 @ (T2 @ Uc), Uc.T @ C)
        return out

    def _factor_all(self, blocks):
        sel = self.selector; groups = {r: {} for r in ROUTES}; timed = {}
        for q, A in blocks.items():
            b = sel.bucket(A.shape)
            if b is None:
                groups['host'][q] = A; continue
            r, t = sel.route(b)
            if t:
                timed.setdefault((b, r), {})[q] = A
            else:
                groups[r][q] = A
        res = self._host_svds(groups['host']); res.update(self._factor_host(groups['devh'])); res.update(self._factor_device(groups['devd']))
        for (b, r), blk in timed.items():
            cp.cuda.Device().synchronize(); t = time.perf_counter(); out = self._ready(r, blk); cp.cuda.Device().synchronize(); dt = time.perf_counter()-t
            sel.record(b, r, dt, float(sum(A.shape[0]*A.shape[1]**2 for A in blk.values())), len(blk)); res.update(out)
            self.prof['mv_timed_batches'] = self.prof.get('mv_timed_batches', 0.0) + dt
        return res

    def _assemble(self, res, new, carry):
        out, kept = super()._assemble({q: v for q, v in res.items() if v[0] != 'ready'}, new, carry); kd = []
        for q, v in res.items():
            k, c = int(new[q]), int(carry[q])
            if v[0] != 'ready' or not k:
                continue
            Cm = v[3][:k, :c] if c else None
            if Cm is not None:
                kd.append(cp.sum(Cm*Cm))
            out[q] = (v[2][:, :k], Cm)
        if kd:
            kept += float(cp.sum(cp.stack(kd)))
        return out, kept
