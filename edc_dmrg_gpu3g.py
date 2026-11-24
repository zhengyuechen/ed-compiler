#!/usr/bin/env python3
"""GPU back end of the layer DMRG, grouped matrix products for v3: the matrix products of one product
with the effective Hamiltonian are issued as four grouped cuBLAS calls instead of one or two cupy matmuls per label.

Matrices of different sizes can share one call to
cublasDgemmGroupedBatched (cuBLAS >= 12.5; variable sizes in one call):
    left half    O_i  = LH_i M_i          (beta 0)     then   O_i  += GL_i S^L_i     (beta 1)
    right half   oU_j = U_j RH_j          (beta 0)     then   oU_j += S^R_j GR_j     (beta 1)
The buffers of a Product (M, S^L, O, U, S^R, oU) and the environments do not move during a solve, so the device arrays of
pointers are built once per Product.  Everything else (slab kernels, layouts, Lanczos, moves, environments) is v3
unchanged.  Labels without LH (layer 0) or without coupling terms are left out of the corresponding call, as in v3.

Row-major C (m x n) = A (m x k) B (k x n) is column-major C^T = B^T A^T: cuBLAS gets (m', n', k') = (n, m, k),
A' = B (ld n), B' = A (ld k), C' = C (ld n).  If the library has no grouped call, the class falls back to v3's loops."""
import time, ctypes
import numpy as np
import cupy as cp
import edc_dmrg as ed
import edc_dmrg_gpu as eg
import edc_dmrg_gpu3 as e3

_LIB = None


def _cublas():
    """The libcublas cupy uses, with cublasDgemmGroupedBatched, or None."""
    global _LIB
    if _LIB is None:
        if not hasattr(cp.cuda, 'device'):  # CPU stand-in has no cuBLAS interface
            _LIB = False
            return None
        cp.cuda.device.get_cublas_handle()                              # cupy loads its libcublas first
        names = ['libcublas.so.12', 'libcublas.so.13', 'libcublas.so']
        try:
            names = [l.split()[-1] for l in open('/proc/self/maps') if 'libcublas.so' in l and 'Lt' not in l][:1] + names
        except OSError:
            pass
        _LIB = False
        for n in names:
            try:
                L = ctypes.CDLL(n); f = L.cublasDgemmGroupedBatched
            except (OSError, AttributeError):
                continue
            P = ctypes.c_void_p
            f.argtypes = [P, P, P, P, P, P, P, P, P, P, P, P, P, P, ctypes.c_int, P]; f.restype = ctypes.c_int
            _LIB = L; break
    return _LIB or None


class Grouped:
    """One grouped call: C_p (+)= A_p B_p for every problem p (row-major cupy views, C-contiguous)."""
    def __init__(self, probs, beta):
        probs = [(A, B, C) for A, B, C in probs if A is not None and B is not None and C.size and A.shape[1]]
        self.n = n = len(probs)
        if not n:
            return
        for A, B, C in probs:
            assert A.flags.c_contiguous and B.flags.c_contiguous and C.flags.c_contiguous and A.shape[1] == B.shape[0] and C.shape == (A.shape[0], B.shape[1])
        I = ctypes.c_int * n; Dd = ctypes.c_double * n
        self.ta = I(*([0]*n)); self.tb = I(*([0]*n))                          # CUBLAS_OP_N
        self.m = I(*[C.shape[1] for A, B, C in probs]); self.nn = I(*[C.shape[0] for A, B, C in probs]); self.k = I(*[A.shape[1] for A, B, C in probs])
        self.lda = I(*[B.shape[1] for A, B, C in probs]); self.ldb = I(*[A.shape[1] for A, B, C in probs]); self.ldc = I(*[C.shape[1] for A, B, C in probs])
        self.alpha = Dd(*([1.0]*n)); self.beta = Dd(*([float(beta)]*n)); self.gs = I(*([1]*n))
        self.Ap = cp.asarray(np.array([B.data.ptr for A, B, C in probs], dtype=np.uint64))
        self.Bp = cp.asarray(np.array([A.data.ptr for A, B, C in probs], dtype=np.uint64))
        self.Cp = cp.asarray(np.array([C.data.ptr for A, B, C in probs], dtype=np.uint64))
        self.keep = probs                                                   # the views must outlive the pointers
        self.flops = 2.0*sum(A.shape[0]*A.shape[1]*B.shape[1] for A, B, C in probs)

    def __call__(self):
        if not self.n:
            return
        r = ctypes.addressof
        st = _LIB.cublasDgemmGroupedBatched(ctypes.c_void_p(cp.cuda.device.get_cublas_handle()), r(self.ta), r(self.tb), r(self.m), r(self.nn), r(self.k), r(self.alpha),
                                            ctypes.c_void_p(self.Ap.data.ptr), r(self.lda), ctypes.c_void_p(self.Bp.data.ptr), r(self.ldb), r(self.beta),
                                            ctypes.c_void_p(self.Cp.data.ptr), r(self.ldc), self.n, r(self.gs))
        if st != 0:
            raise RuntimeError(f'cublasDgemmGroupedBatched returned {st}')


class ProductG(eg.Product):
    """eg.Product with the label loops of apply() replaced by grouped calls."""
    def __init__(self, eng, r, prof):
        super().__init__(eng, r, prof); self.grouped = _cublas() is not None
        if self.grouped:
            self.gL1 = Grouped([(LH, M, O) for LH, M, O, GL, SL in self.left if LH is not None], 0.0)
            self.gL2 = Grouped([(GL, SL, O) for LH, M, O, GL, SL in self.left if GL is not None], 1.0)
            self.gR1 = Grouped([(U, RH, oU) for U, RH, oU, SR, GR in self.right if RH is not None], 0.0)
            self.gR2 = Grouped([(SR, GR, oU) for U, RH, oU, SR, GR in self.right if SR is not None], 1.0)

    def apply(self, v):
        if not self.grouped:
            return super().apply(v)
        i32, i64 = np.int32, np.int64; n = self.n; t = time.perf_counter(); self.Mb[...] = v; self.Ob.fill(0.0); self.oU.fill(0.0); t = self._t('copy', t)
        if self.hasL:
            ptr, ein, ev = self.tL
            eg._launch('sl_build', self.SL.size, (self.Mb, self.SL, i64(self.SL.size), i32(self.nL), i32(self.S), i32(self.NTL), self.SLoff, self.C, self.DL,
                                                  self.srcL, self.wstL, self.cst, self.cumc, self.Moff, ptr, ein, ev))
        t = self._t('slab', t)
        self.gL1(); self.gL2()
        t = self._t('gemm', t)
        eg._launch('m2u', n, (self.Mb, self.U, i64(n), i32(self.nR), i32(self.S), self.Uoff, self.DR, self.cumr, self.ul, self.Moff, self.C, self.cst))
        t = self._t('layout', t)
        if self.hasR:
            ptr, ein, ev = self.tR
            eg._launch('sr_build', self.SR.size, (self.U, self.SR, i64(self.SR.size), i32(self.nR), i32(self.S), i32(self.NTR), self.SRoff, self.WRg, self.DR,
                                                  self.srcR, self.wstR, self.rst, self.cumr, self.Uoff, ptr, ein, ev))
        t = self._t('slab', t)
        self.gR1(); self.gR2()
        t = self._t('gemm', t)
        ptr, ein, ev = self.tH
        eg._launch('hr_add', n, (self.U, self.oU, i64(n), i32(self.nR), i32(self.S), self.Uoff, self.DR, self.rst, self.cumr, ptr, ein, ev))
        t = self._t('slab', t)
        eg._launch('u2m_add', n, (self.oU, self.Ob, i64(n), i32(self.nL), i32(self.S), self.Moff, self.C, self.cumc, self.mr, self.Uoff, self.DR, self.rst))
        t = self._t('layout', t)
        out = self.Ob.copy(); self._t('copy', t)
        return out


class DMRGGPU3g(e3.DMRGGPU3):
    """v3 (factor = 'host') with ProductG in the layer solves; the moves use the same Product object as in v3."""
    version = 'gpu-v3g'; product_cls = ProductG
