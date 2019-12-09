import sr
from cnf_util import *
from pysat.solvers import Solver
from pysat.formula import CNF
import numpy as np

def get_unsat_randkcnf(k,n, alpha=4.4):
  while True:
    result = sample_randkcnf(k,n, alpha)
    with Solver(name="cdl", bootstrap_with=result) as S:
      if S.solve():
        continue
      else:
        break
  return result

def get_unsat_sr(n1, n2,min_cls_len=1,p_binom=0.7, p_geo=0.4):
  n = np.random.choice(range(n1, n2+1))
  return sr.sample_SR(n, min_cls_len, p_binom, p_geo)[0]

def get_unsat_src(x1,x2,min_cls_len=2):
  return sr.gen_src(n1=x1, n2=x2, min_cls_len=min_cls_len)
