import scipy.linalg as _sl
import cupy as _cp

solve_triangular = _cp._fn(_sl.solve_triangular, 'solve_triangular')
