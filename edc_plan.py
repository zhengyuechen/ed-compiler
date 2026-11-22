#!/usr/bin/env python3
"""edc_plan — measure, then choose (the planner).

The fastest way to run a product depends on the machine and on the sector: the number of threads (a CPU with fast and slow
cores is not always helped by one thread per core), the GPU launch block, how many vectors share one pass over the stored
connectivity.  plan() times the candidates on the engine itself, with its own tables, sets the best and returns the table;
the result is cached per machine and sector.  Profile splits a run into products and everything else, so that a slow step
outside the products is seen (an inner product ten times slower than the matrix product went unnoticed).
"""
import os, sys, json, time, ctypes, hashlib, platform, subprocess
import numpy as np


def _omp():
    import edc
    lib = getattr(edc, '_OMP', None)
    if lib is None:
        for name in ('libgomp.so.1', 'libomp.so', 'libiomp5.so'):
            try:
                lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL); break
            except OSError:
                pass
    return lib


def set_threads(n):
    _omp().omp_set_num_threads(int(n))


def get_threads():
    return int(_omp().omp_get_max_threads())


def cores():
    """(fast cores, all cores) of this machine."""
    if sys.platform == 'darwin':
        q = lambda k: int(subprocess.run(['sysctl', '-n', k], capture_output=True, text=True).stdout or 0)
        p, n = q('hw.perflevel0.physicalcpu'), q('hw.ncpu')       # chips with three kinds of cores: count them all
        if p and n:
            return p, n
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return n, n


def machine(device):
    if device == 'gpu':
        import cupy as cp
        d = cp.cuda.Device()
        try:
            name = cp.cuda.runtime.getDeviceProperties(d.id)['name'].decode()
        except Exception:
            name = 'gpu'
        return f'{name} ({d.mem_info[1]/2**30:.0f} GiB)'
    name = platform.processor() or platform.machine()
    if sys.platform == 'darwin':
        name = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() or name
    return f'{name} ({cores()[1]} cores)'


def _sync(eng):
    if eng.device == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()


def _best(eng, fn, reps):
    best = np.inf
    for _ in range(reps+1):
        _sync(eng); t = time.perf_counter(); fn(); _sync(eng)
        best = min(best, time.perf_counter()-t)
    return best


def _cache_file():
    d = os.environ.get('EDC_CACHE', os.path.expanduser('~/.cache/edc'))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'plans.json')


def _key(eng, vectors):
    import edc_sub
    import math
    sig = repr((edc_sub.structure_key(eng._m2), int(round(math.log2(max(int(eng.dim), 1)))), bool(eng.real), getattr(eng, 'stored_format', 0),
                int(vectors)))                                  # sectors of the same kind and size class share a plan
    return machine(eng.device)+' | '+hashlib.sha1(sig.encode()).hexdigest()[:16]


def apply(eng, p):
    """Set a plan on an engine."""
    if eng.device == 'cpu' and p.get('threads'):
        set_threads(p['threads'])
    if eng.device == 'gpu' and p.get('block'):
        eng.block = int(p['block'])
        if getattr(eng, '_st', None) is None:
            eng.grid = (max(int(eng.dim), 1)+eng.block-1)//eng.block
    eng.plan = p
    return p


def plan(eng, vectors=1, reps=3, cache=True, log=None):
    """Time the candidates for the product of this engine and set the fastest.
    vectors: how many vectors the workload carries at once (FTLM chains, scan points); with stored connectivity the number
    K of vectors per pass is chosen too.  Returns dict(threads | block, K, seconds_per_vector, default_seconds, table)."""
    key = _key(eng, vectors)
    if cache and os.path.exists(_cache_file()):
        try:
            known = json.load(open(_cache_file()))
            if key in known:
                return apply(eng, dict(known[key], cached=True))
        except Exception:
            pass
    xp, n = eng.xp, int(eng.dim)
    dt = np.float64 if eng.real else np.complex128
    x = xp.ones(n, dt); y = xp.empty_like(x)                     # the time of a product does not depend on the values
    one = lambda: eng.matvec(x, y)
    table = []
    if eng.device == 'cpu':
        fast, every = cores(); default = get_threads()
        cands = sorted({c for c in (fast, every, default, max(1, every-1), every+2) if c >= 1})
        for c in cands:
            set_threads(c); table.append(dict(threads=c, K=1, seconds_per_vector=_best(eng, one, reps)))
        base = next(r['seconds_per_vector'] for r in table if r['threads'] == default)
        best = min(table, key=lambda r: r['seconds_per_vector']); p = dict(threads=best['threads'])
        set_threads(best['threads'])
    else:
        default = int(eng.block)
        for b in (64, 128, 256, 512, 1024):
            eng.block = b
            if getattr(eng, '_st', None) is None:
                eng.grid = (max(n, 1)+b-1)//b
            table.append(dict(block=b, K=1, seconds_per_vector=_best(eng, one, reps)))
        base = next(r['seconds_per_vector'] for r in table if r['block'] == default)
        best = min(table, key=lambda r: r['seconds_per_vector']); p = dict(block=best['block'])
        apply(eng, p)
    p.update(K=1, seconds_per_vector=best['seconds_per_vector'])
    st = getattr(eng, '_st', None)
    if vectors > 1 and st is not None and st.get('compact'):
        vs = 8 if eng.real else 16
        if eng.device == 'gpu':
            import cupy as cp
            free = cp.cuda.runtime.memGetInfo()[0]
        else:
            free = 0.5*os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        del x, y
        for K in (2, 4, 8, 16):
            if K > vectors or 2.3*n*K*vs > free:
                break
            X = xp.ones((n, K), dt); Y = xp.empty_like(X)
            f = eng.batched(K)
            t = _best(eng, lambda: f(X, Y), reps)/K
            table.append(dict(p, K=K, seconds_per_vector=t))
            if t < p['seconds_per_vector']:
                p.update(K=K, seconds_per_vector=t)
            del X, Y, f
    p.update(default_seconds=base, gain=base/p['seconds_per_vector'], table=table, machine=machine(eng.device), dim=n)
    if log:
        log(f'plan: {json.dumps({k: v for k, v in p.items() if k != "table"})}')
    if cache:
        try:
            known = json.load(open(_cache_file())) if os.path.exists(_cache_file()) else {}
            known[key] = p; json.dump(known, open(_cache_file(), 'w'))
        except Exception:
            pass
    return apply(eng, p)


class Profile:
    """with Profile(eng) as pr: ...   then pr.report(): seconds and calls inside the products, share of the whole."""

    def __init__(self, eng):
        self.eng, self.calls, self.inside = eng, 0, 0.0

    def __enter__(self):
        eng = self.eng; self._mv = eng.matvec
        def timed(x, y, accumulate=False):
            _sync(eng); t = time.perf_counter(); r = self._mv(x, y, accumulate); _sync(eng)
            self.inside += time.perf_counter()-t; self.calls += 1
            return r
        eng.matvec = timed
        _sync(eng); self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        _sync(self.eng); self.total = time.perf_counter()-self.t0
        del self.eng.matvec                                      # back to the method of the class
        return False

    def report(self):
        share = self.inside/self.total if self.total > 0 else 0.0
        out = dict(seconds=self.total, products=self.calls, seconds_in_products=self.inside, share_in_products=share)
        if self.calls > 20 and share < 0.6:
            out['warning'] = 'less than 60 % of the time is spent in the products: look at the steps around them'
        return out
