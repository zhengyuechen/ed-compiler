"""su2_check.py with the contractions of su2_dmrg done as chained matrix products (su2_gemm.install): the exact checks
of the (j, k) layer DMRG on 15, 18 and 20 spins.  Usage: su2_gemm_check.py [REPO]"""
import sys, os, runpy
D = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else os.path.join(sys.argv[1], 'bench', 'dmrg_su2'); sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, '..', '..'))
import su2_dmrg, su2_gemm; su2_gemm.install(su2_dmrg); print('kernel: chained BLAS products', flush=True)
runpy.run_path(os.path.join(D, 'su2_check.py'), run_name='__main__')
