"""Thin factorisation of a tall block for the DMRG moves, with the tall part as matrix products only.

A (m x n, m >= n) = Q C, Q with orthonormal columns: Gram matrix G = A^T A (one product, on the device when A lives
there), columns scaled to unit norm, Cholesky of the small matrix on the host, Q1 = A T1; the same once more on Q1
(the first pass leaves an orthogonality error of about eps * cond(A)^2, the second removes it: "Cholesky QR 2").
C = L2^T L1^T diag(d) is n x n, and its singular value decomposition (host, small) gives the Schmidt values.
Returns None when the block is not tall, not of full rank to working precision, or the check of the second Gram
matrix fails; the caller then uses an ordinary SVD."""
import numpy as np
import scipy.linalg as sl


def tall_qr(A, xp, to_host, to_dev):
    m, n = A.shape
    if m < n or n == 0:
        return None
    G = to_host(A.T @ A); d = np.sqrt(np.diag(G))
    if not np.all(d > 0) or not np.all(np.isfinite(d)):
        return None
    try:
        L1 = np.linalg.cholesky(G/np.outer(d, d)); T1 = sl.solve_triangular(L1, np.diag(1.0/d), lower=True).T     # D^-1 L1^-T
        Q1 = A @ to_dev(T1); G2 = to_host(Q1.T @ Q1)
        if not np.all(np.isfinite(G2)) or np.abs(G2 - np.eye(n)).max() > 0.05:
            return None
        L2 = np.linalg.cholesky(G2); T2 = sl.solve_triangular(L2, np.eye(n), lower=True).T                        # L2^-T
    except np.linalg.LinAlgError:
        return None
    return Q1 @ to_dev(T2), (L2.T @ L1.T)*d[None, :]


if __name__ == '__main__':                                             # self-test on the host
    rng = np.random.default_rng(0); ident = lambda x: x
    for m, n, lo in ((6000, 180, -6), (900, 180, -7), (400, 60, -3), (200, 200, -5), (3000, 120, -9)):
        X = np.linalg.qr(rng.standard_normal((m, n)))[0]*np.logspace(0, lo, n) @ np.linalg.qr(rng.standard_normal((n, n)))[0]
        X = np.concatenate([X, 1e-6*rng.standard_normal((m, n//5))], axis=1) if m > n + n//5 else X
        out = tall_qr(X, np, ident, ident)
        if out is None:
            print(f'{m} x {X.shape[1]}, singular values down to 1e{lo}: declined (ordinary SVD is used)'); continue
        Q, C = out; s = np.linalg.svd(C, compute_uv=False); s0 = np.linalg.svd(X, compute_uv=False)
        print(f'{m} x {X.shape[1]}, singular values down to 1e{lo}: |Q^T Q - 1| = {np.abs(Q.T @ Q - np.eye(Q.shape[1])).max():.1e}, |A - Q C| = {np.abs(X - Q @ C).max():.1e}, '
              f'singular values: largest absolute difference {np.abs(s - s0).max():.1e}')
