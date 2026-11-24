"""One change to su2_dmrg.py: every contraction of an operator pair with a tensor block is done as block2
does it (OperatorFunctions::tensor_product_multiply -> GMatrixFunctions::rotate): a chain of two-factor matrix products
through BLAS, instead of numpy's one-pass einsum over all indices.  Nothing else changes (same blocks, same Python loops,
same schedule, same random numbers).  install(su) replaces np.einsum inside the module su2_dmrg."""
import numpy as _np

def _op_mid(M, x):                                                      # 'ys,asb->ayb'
    a, s, b = x.shape
    return (M @ x.transpose(1, 0, 2).reshape(s, a*b)).reshape(M.shape[0], a, b).transpose(1, 0, 2)

def _left(O, x):                                                        # 'xa,asb->xsb'
    a, s, b = x.shape
    return (O @ x.reshape(a, s*b)).reshape(O.shape[0], s, b)

def _right(x, R):                                                       # 'asb,bc->asc'
    a, s, b = x.shape
    return (x.reshape(a*s, b) @ R).reshape(a, s, R.shape[1])

def _tr_left(x, y):                                                     # 'asb,asc->bc'
    return x.reshape(-1, x.shape[2]).T @ y.reshape(-1, y.shape[2])

def _tr_right(x, y):                                                    # 'xsb,ysb->xy'
    return x.reshape(x.shape[0], -1) @ y.reshape(y.shape[0], -1).T

TABLE = {
    'st,atb->asb': lambda H, x: _op_mid(H, x),
    'ys,asb->ayb': lambda M, x: _op_mid(M, x),
    'xa,asb->xsb': lambda O, x: _left(O, x),
    'xb,bsc->xsc': lambda O, x: _left(O, x),
    'asb,bc->asc': lambda x, R: _right(x, R),
    'xsa,ab->xsb': lambda x, R: _right(x, R),
    'xa,ys,asb->xyb': lambda O, M, x: _left(O, _op_mid(M, x)),
    'ys,zb,asb->ayz': lambda M, O, x: _right(_op_mid(M, x), O.T),
    'asb,asc->bc': lambda x, y: _tr_left(x, y),
    'asb,st,atc->bc': lambda A, M, x: _tr_left(A, _op_mid(M, x)),
    'xsb,st,ytb->xy': lambda x, H, y: _tr_right(x, _op_mid(H, y)),
    'xsb,bc,ysc->xy': lambda x, R, y: _tr_right(_right(x, R), y),
    'xsc,st,cd,ytd->xy': lambda B, M, O, x: _tr_right(B, _right(_op_mid(M, x), O.T)),
    'xsc,st,ytc->xy': lambda B, M, x: _tr_right(B, _op_mid(M, x)),
}

class _NP:
    def __init__(self): self.calls = {}
    def __getattr__(self, name): return getattr(_np, name)
    def einsum(self, sub, *ops):
        f = TABLE.get(sub)
        return f(*ops) if f is not None else _np.einsum(sub, *ops)

def install(su):
    su.np = _NP(); return su

if __name__ == '__main__':                                              # every pattern against numpy's einsum
    rng = _np.random.default_rng(0); dim = dict(a=5, b=7, c=4, d=6, s=3, t=2, x=8, y=9, z=3, M=2, m=2); worst = 0.0
    for sub, f in TABLE.items():
        ins, out = sub.split('->'); loc = dict(dim)
        if sub == 'st,atb->asb' or sub == 'xsb,st,ytb->xy': pass
        ops = [rng.standard_normal([loc[ch] for ch in term]) for term in ins.split(',')]
        worst = max(worst, float(abs(f(*ops) - _np.einsum(sub, *ops)).max()))
    print('largest difference to numpy einsum over', len(TABLE), 'patterns:', worst)
