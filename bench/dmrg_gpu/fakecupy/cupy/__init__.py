"""A numpy stand-in for the part of cupy that the GPU back end uses, for checks on a machine without a GPU.

    PYTHONPATH=bench/dmrg_gpu/fakecupy python ...      (then `import cupy` finds this package)

* "Device" arrays are a numpy subclass.  cupy.asarray of a host array copies (as on a GPU), asnumpy copies back, and a
  host array (ndim > 0) used in an operation with device arrays, or assigned into one, raises TypeError (cupy does
  too), so a missing transfer is caught here.
* RawModule compiles the CUDA source as C: every `extern "C" __global__` kernel whose first statement is the usual
  `i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;` becomes a per-element function and a
  driver that loops x over 0..total-1 (the kernels of this project write only element x, so the order does not matter).
* linalg.cholesky returns NaNs instead of raising (cupy's default), random numbers come from one numpy generator.
Timings measured with this package mean nothing."""
import re, os, hashlib, subprocess, ctypes, tempfile
import numpy as np


class ndarray(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kw):
        args = [_dev_in(a, ufunc.__name__) for a in inputs]
        if out is not None:
            kw['out'] = tuple(_dev_in(o, ufunc.__name__) for o in out)
        r = getattr(ufunc, method)(*args, **kw)
        return tuple(_wrap(y) for y in r) if isinstance(r, tuple) else _wrap(r)

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray) and not isinstance(value, ndarray) and value.ndim > 0:
            raise TypeError('host array assigned into a device array')
        np.ndarray.__setitem__(self.view(np.ndarray), key, value.view(np.ndarray) if isinstance(value, ndarray) else value)


def _dev_in(a, what):
    if isinstance(a, ndarray):
        return a.view(np.ndarray)
    if isinstance(a, np.ndarray) and a.ndim > 0:
        raise TypeError(f'host array in a device operation ({what})')
    return a


def _wrap(y):
    return y.view(ndarray) if isinstance(y, np.ndarray) and not isinstance(y, ndarray) else y


def _fn(f, name):
    def g(*a, **k):
        for x in list(a) + list(k.values()):                             # check only; device arrays pass as they are
            for z in (x if isinstance(x, (list, tuple)) else [x]):
                if isinstance(z, np.ndarray):
                    _dev_in(z, name)
        r = f(*a, **k)
        return tuple(_wrap(y) for y in r) if isinstance(r, tuple) else _wrap(r)
    return g


def asarray(a, dtype=None):
    if isinstance(a, ndarray):
        return a if dtype is None or a.dtype == dtype else a.astype(dtype)
    return np.array(a, dtype=dtype, copy=True).view(ndarray)


def asnumpy(a):
    return np.array(a.view(np.ndarray) if isinstance(a, ndarray) else a, copy=True)


def __getattr__(name):
    f = getattr(np, name)
    if isinstance(f, type) or not callable(f):
        return f
    return _fn(f, name)


class _Linalg:
    norm = staticmethod(_fn(np.linalg.norm, 'norm'))
    eigh = staticmethod(_fn(np.linalg.eigh, 'eigh'))
    svd = staticmethod(_fn(np.linalg.svd, 'svd'))
    qr = staticmethod(_fn(np.linalg.qr, 'qr'))

    @staticmethod
    def cholesky(a):
        try:
            return _wrap(np.linalg.cholesky(_dev_in(a, 'cholesky')))
        except np.linalg.LinAlgError:
            return _wrap(np.full(a.shape, np.nan))


linalg = _Linalg()


class _Random:
    gen = np.random.default_rng(0)

    def seed(self, s=0):
        self.gen = np.random.default_rng(s)

    def standard_normal(self, size=None):
        return _wrap(np.asarray(self.gen.standard_normal(size)))


random = _Random()


class _Device:
    def __init__(self, *a):
        pass

    def synchronize(self):
        pass


class _Runtime:
    @staticmethod
    def getDeviceProperties(i):
        return {'name': b'numpy stand-in (no GPU)'}


class _Pool:
    def total_bytes(self):
        return 0

    def used_bytes(self):
        return 0

    def free_all_blocks(self):
        pass


_POOL = _Pool()


def get_default_memory_pool():
    return _POOL


class _MemoryHook:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class cuda:
    Device = _Device
    runtime = _Runtime
    MemoryHook = _MemoryHook

    @staticmethod
    def get_current_stream():
        return _Device()


_FIRST = 'i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x; if (x >= total) return;'
_CACHE = os.path.join(tempfile.gettempdir(), 'fakecupy_kernels')


class RawModule:
    def __init__(self, code, options=()):
        src = code.replace('__device__ inline', 'static inline'); kernels = []

        def rep(m):
            kernels.append((m.group(1), ' '.join(m.group(2).split())))
            return f'static void {m.group(1)}__body(i64 x__, {m.group(2)})'
        src = re.sub(r'extern "C" __global__ void (\w+)\(([^)]*)\)', rep, src)
        assert src.count(_FIRST) == len(kernels), 'a kernel does not start with the usual index line'
        src = src.replace(_FIRST, 'i64 x = x__;'); assert 'blockIdx' not in src and 'threadIdx' not in src
        self.sig = {}
        for name, params in kernels:
            ps = [p.strip() for p in params.split(',')]; names = [p.split()[-1].lstrip('*') for p in ps]
            assert 'total' in names, name
            src += f'\nvoid {name}({params}) {{ for (i64 x__ = 0; x__ < total; x__++) {name}__body(x__, {", ".join(names)}); }}\n'
            self.sig[name] = [ctypes.c_void_p if '*' in p else ctypes.c_int64 if p.split()[0] in ('i64', 'long') else ctypes.c_double if p.split()[0] == 'double' else ctypes.c_int for p in ps]
        os.makedirs(_CACHE, exist_ok=True); h = hashlib.sha1(src.encode()).hexdigest()[:16]; so = os.path.join(_CACHE, f'k_{h}.so')
        if not os.path.exists(so):
            c = os.path.join(_CACHE, f'k_{h}.c'); open(c, 'w').write(src)
            subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-x', 'c', c, '-o', so], check=True)
        self.lib = ctypes.CDLL(so)

    def get_function(self, name):
        f = getattr(self.lib, name); f.argtypes = self.sig[name]; f.restype = None

        def call(grid, block, args):
            conv = []
            for a in args:
                if isinstance(a, np.ndarray):
                    assert a.flags.c_contiguous, f'{name}: argument not contiguous'
                    conv.append(a.ctypes.data)
                else:
                    conv.append(int(a) if isinstance(a, (int, np.integer)) else float(a))
            f(*conv)
        return call
