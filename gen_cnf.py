import tempfile
import cnfformula
import sys
import os
import numpy as np
import time
import datetime
import ray
from cnf_util import *
from tftd import TFDCR, tfdcr_to_example
from pysat.solvers import Solver
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import mp
from util import *
from pysat.solvers import Solver
from gen_labelled_data import get_unsat_randkcnf
from types import SimpleNamespace

def grid_enc (w,l,i,j,c):
  return i + (j - 1) * w + (c - 1) * (w * l)

def grid_enc_inv (w,l,k):
  """
  Note: i, j, and c all start at 1
  """
  i = ((k-1) % w)+1
  j = (((k - i) % (w*l)))/w + 1
  c = (k - i - ((j - 1) * w))/(w*l) + 1
  return (i,j,c)

def grid_enc_validate ():
  w = 5
  l = 5
  for i in range(1,6):
    for j in range(1,6):
      for k in range(1,10):
        if not grid_enc_inv(5,5,(grid_enc(5,5,i,j,k))) == (i,j,k):
          raise Exception("grid_enc_inv is not a left inverse of grid_enc")
  print("test passed")

def PTN(N):
  return FromCNFFormula(cnfformula.PythagoreanTriples(N))

def at_most_one_clauses (vs):
  clauses = []
  for i in range(0, len(vs)-1):
    for j in range(i+1, len(vs)):
      clauses.append([- vs[i],- vs[j]])
  return clauses

def bad_arithmetic_progressions(K, N):
  return list (filter (lambda p: p[-1] <= N, [[b + n * d for n in range(0,K)] for d in range(1,N+1) for b in range(1,N+1)]))

def arithmetic_progressions(K, N): #arithmetic progressions of length K in [1..N]
  result = []
  for b in range(1,N+2 - K):
    for d in range(1, N):
      bs = [b]
      n = 1
      BREAK_FLAG = False
      while len(bs) < K:
        next_b = b + n * d
        if next_b > N:
          BREAK_FLAG = True
          break
        else:
          bs.append(next_b)
          n += 1
      if BREAK_FLAG:
        continue
      else:
        result.append(bs)
  return result  

def vdW(r,k,N): # SAT iff there exists a coloring of [1..N] such that no arithmetic progression is monochromatic, so UNSAT iff for every coloring of [1..N], there exists a monochromatic arithmetic progression, i.e. if the (r,k)th vdW number is less than or equal to N.
  clauses = []
  for v in range(1, N+1):   # every integer between 1 and N is assigned some color
    clauses.append([grid_enc(N,1,v,1,c) for c in range(1, r+1)])
  # sat-preserving if we omit these clauses, since a multi-coloring can always be converted into a coloring    
  # for v in range(1, N+1): # every integer 1 to N is assigned at most one color
  #   clauses += at_most_one_clauses([grid_enc(N,1,v,1,c) for c in range(1, r+1)])
  for p in arithmetic_progressions(k, N):  # every arithmetic progression of length k in [1..N] is NOT monochromatic
    for c in range(1, r+1):
      clauses.append([-grid_enc(N,1,x,1,c) for x in p])
  return CNF(from_clauses=clauses)

def vdWtest():
  fmla = vdW(3,3,20)
  fmla.to_fp(sys.stdout)
  for N in range(1,50):
    print(N)
    with Solver(name="cdl", bootstrap_with=vdW(3,3,N)) as S:
      print(S.solve())

# schur number of K is greater than or equal to N: if there exists a sum-free K-coloring of [1..n].  
def Schur(K,N):
  result = []
  for v in range(1, N+1):   # every integer between 1 and N is assigned some color
    result.append([grid_enc(N,1,v,1,c) for c in range(1, K+1)])

  for v in range(1, N+1): # every integer 1 to N is assigned at most one color
    result += at_most_one_clauses([grid_enc(N,1,v,1,c) for c in range(1, K+1)])
    
  # for every a b, and c such that a + b = c, a, b and c are not the same color

  for a in range(1, N+1):
    for b in range(1, N-a+1):
      for c in range(1, K+1):
        result.append([-grid_enc(N,1,a,1,c),-grid_enc(N,1,b,1,c),-grid_enc(N,1,a+b,1,c)])

  return CNF(from_clauses=result)


def RamseyLowerBound(s,k,N):
  """
  s: independent set size
  k: clique size
  N: vertices

  Returns:
  A formula asserting that R(s,k) > N
  """
  return FromCNFFormula(cnfformula.RamseyLowerBoundFormula(s,k,N))

def gen_ramsey_fragment(s,k,N,c):
  fmla = RamseyLowerBound(s,k,N)  
  if c == 0:
    return fmla
  else:
    return gen_fmla_fragment(fmla, c)

def parse_gen_ramsey():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("s", type=int)
  parser.add_argument("k", type=int)
  parser.add_argument("N", type=int)
  parser.add_argument("c", type=int)  
  parser.add_argument("n_fmlas", type=int)
  parser.add_argument("--dest", action="store", dest="dest", type=str)
  opts = parser.parse_args()
  return opts

def gen_ramsey(opts=None):
  if opts is None:
    opts = parse_gen_ramsey()

  cnf_dir = opts.dest

  check_make_path(cnf_dir)

  for count in range(opts.n_fmlas):
    name = f"ramsey_{count}.cnf"
    file_path = os.path.join(cnf_dir, name)
    log_path = os.path.join(cnf_dir, "summary.log")
    start = time.time()
    fmla = gen_ramsey_fragment(opts.s, opts.k, opts.N, opts.c)
    elapsed = time.time() - start
    with open(file_path, "w") as f:
        fmla.to_fp(f)
    with open(log_path, "a") as f:
        print(f"generated {name} in {elapsed} seconds", file=f)
        print(f"{fmla.nv} variables and {len(fmla.clauses)} clauses", file=f)
  with open(log_path, "a") as f:
      print("done", file=f)      

# profile hardness of the problems
# if __name__ == "__main__":
#   cnf_files = files_with_extension("cnf_data_test/ramsey/test/", "cnf")
#   for f in cnf_files:
#     fmla = CNF(from_file=f)
#     with Solver(name="cadical", bootstrap_with=fmla) as S:
#       start = time.time()
#       S.solve()
#       elapsed = time.time() - start
#     print(elapsed)

# python gen_ramsey.py 3 250 1000 --alpha=4.4 --dest=cnf_data/randkcnf200/test/
# python gen_ramsey.py 5 60 1000 --alpha=21.4 --dest=cnf_data/rand5cnf60/test/ --workers=8
# python gen_ramsey.py 7 38 1000 --alpha=88.0 --dest=cnf_data/rand7cnf38/test/ --workers=8
@ray.remote
class RandKCNFWorker:
  def __init__(self, index):
    self.index = index

  @ray.method(num_return_vals=1)
  def get_fmla(self, k, n, alpha):
    fmla = get_unsat_randkcnf(k,n,alpha)
    return fmla, self.index

def ParallelRandKCNFGen(k,n,alpha,n_fmlas,num_workers, dest):
  ray.init(num_cpus=num_workers)
  worker_pool = [RandKCNFWorker.remote(i) for i in range(num_workers)] # vroom vroom
  active_jobs = [worker.get_fmla.remote(k,n,alpha) for worker in worker_pool]

  num_results = 0

  check_make_path(dest)

  log_path = os.path.join(dest, "summary.log")

  while num_results < n_fmlas - num_workers:
    ready_ids, active_jobs = ray.wait(active_jobs)
    for x in ready_ids:
      fmla, index = ray.get(x)
      with open(os.path.join(dest,f"rand{k}cnf{n}_{num_results}.cnf"), "w") as f:
        fmla.to_fp(f)
      with open(log_path, "a") as f:
        print(f"{datetime.datetime.now()}: generated rand{k}cnf{n}_{num_results}", file=f)
      num_results += 1
    active_jobs.append(worker_pool[index].get_fmla.remote(k,n,alpha))

  for job in active_jobs:
    fmla, _ = ray.get(job)
    with open(os.path.join(dest,f"rand{k}cnf{n}_{num_results}.cnf"), "w") as f:
        fmla.to_fp(f)
    with open(log_path, "a") as f:
      print(f"{datetime.datetime.now()}: generated rand{k}cnf{n}_{num_results}", file=f)
    num_results += 1


def parallel_gen_randkcnf_cnf():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("k", type=int)
  parser.add_argument("n", type=int)
  parser.add_argument("n_fmlas", type=int)  
  parser.add_argument("--alpha", type=float, dest="alpha", default=4.4)
  parser.add_argument("--dest", action="store", dest="dest", type=str)
  parser.add_argument("--workers", action="store", dest="num_workers", type=int, default=4)
  opts = parser.parse_args()

  ParallelRandKCNFGen(**vars(opts))

  print("done")

def gen_randkcnf_cnf():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("k", type=int)
  parser.add_argument("n", type=int)
  parser.add_argument("n_fmlas", type=int)  
  parser.add_argument("--alpha", type=float, dest="alpha", default=4.4)
  parser.add_argument("--dest", action="store", dest="dest", type=str)
  parser.add_argument("--workers", action="store", dest="num_workers", type=int, default=4)
  opts = parser.parse_args()

  cnf_dir = opts.dest

  check_make_path(cnf_dir)

  for count in range(opts.n_fmlas):
    name = f"randkcnf_{count}.cnf"
    file_path = os.path.join(cnf_dir, name)
    log_path = os.path.join(cnf_dir, "summary.log")
    start = time.time()
    fmla = get_unsat_randkcnf(opts.k, opts.n, opts.alpha)
    elapsed = time.time() - start
    with open(file_path, "w") as f:
        fmla.to_fp(f)
    with open(log_path, "a") as f:
        print(f"generated {name} in {elapsed} seconds", file=f)
        print(f"{fmla.nv} variables and {len(fmla.clauses)} clauses", file=f)
  with open(log_path, "a") as f:
      print("done", file=f)

def parse_gen_schur():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("k", type=int)
  parser.add_argument("n", type=int)
  parser.add_argument("c", type=int)
  parser.add_argument("n_fmlas", type=int)
  parser.add_argument("--dest", action="store", dest="dest", type=str)
  opts = parser.parse_args()
  return opts

# usage: python gen_ramsey.py 4 45 5 1000 --dest=cnf_data/schur/test/
def gen_schur(opts=None):

  if opts is None:
    opts = parse_gen_schur()

  cnf_dir = opts.dest

  check_make_path(cnf_dir)

  big_fmla = Schur(opts.k, opts.n)

  for count in range(opts.n_fmlas):
    name = f"schur_{count}.cnf"
    file_path = os.path.join(cnf_dir, name)
    log_path = os.path.join(cnf_dir, "summary.log")
    start = time.time()
    fmla = gen_fmla_fragment(big_fmla, opts.c)
    elapsed = time.time() - start
    with open(file_path, "w") as f:
        fmla.to_fp(f)
    with open(log_path, "a") as f:
        print(f"generated {name} in {elapsed} seconds", file=f)
        print(f"{fmla.nv} variables and {len(fmla.clauses)} clauses", file=f)
  with open(log_path, "a") as f:
      print("done", file=f)

def parse_gen_vdW():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("r", type=int)
  parser.add_argument("k", type=int)
  parser.add_argument("N", type=int)
  parser.add_argument("c", type=int)
  parser.add_argument("n_fmlas", type=int)
  parser.add_argument("--dest", action="store", dest="dest", type=str)
  opts = parser.parse_args()
  return opts

def gen_vdW(opts=None): # usage: python gen_ramsey.py 2 5 179 3 1000 --dest=cnf_data/vdw/test/
  if opts is None:
    opts = parse_gen_vdw()

  cnf_dir = opts.dest

  check_make_path(cnf_dir)

  big_fmla = vdW(opts.r, opts.k, opts.N)

  for count in range(opts.n_fmlas):
    name = f"vdW_{count}.cnf"
    file_path = os.path.join(cnf_dir, name)
    log_path = os.path.join(cnf_dir, "summary.log")
    start = time.time()
    fmla = gen_fmla_fragment(big_fmla, opts.c)
    elapsed = time.time() - start
    with open(file_path, "w") as f:
        fmla.to_fp(f)
    with open(log_path, "a") as f:
        print(f"generated {name} in {elapsed} seconds", file=f)
        print(f"{fmla.nv} variables and {len(fmla.clauses)} clauses", file=f)
  with open(log_path, "a") as f:
      print("done", file=f)

def test_gen_cnfs():
  ramsey_opts = SimpleNamespace(
    s = 4,
    k = 4,
    N = 18,
    c = 35,
    n_fmlas = 10,
    dest = "cnf_data/ramsey/test/"
  )

  schur_opts = SimpleNamespace(
    k = 4,
    n = 45,
    c = 5,
    n_fmlas = 10,
    dest = "cnf_data/schur/test"
  )
  
  vdw_opts = SimpleNamespace(
    r = 2,
    k = 5,
    N = 179,
    c = 3,
    n_fmlas = 10,
    dest="cnf_data/vdw/test/"
  )

  gen_ramsey(ramsey_opts)

  gen_schur(schur_opts)

  gen_vdW(vdw_opts)
  
  
def gen_all_cnfs():
  ramsey_opts = SimpleNamespace(
    s = 4,
    k = 4,
    N = 18,
    c = 35,
    n_fmlas = 1000,
    dest = "cnf_data/ramsey/test/"
  )

  schur_opts = SimpleNamespace(
    k = 4,
    n = 45,
    c = 5,
    n_fmlas = 1000,
    dest = "cnf_data/schur/test"
  )
  
  vdw_opts = SimpleNamespace(
    r = 2,
    k = 5,
    N = 179,
    c = 3,
    n_fmlas = 1000,
    dest="cnf_data/vdw/test/"
  )

  gen_ramsey(ramsey_opts)

  gen_schur(schur_opts)

  gen_vdW(vdw_opts)

if __name__ == "__main__":
  gen_all_cnfs()
