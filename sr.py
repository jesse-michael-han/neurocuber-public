import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
import datetime
import tempfile
import os
import subprocess
import itertools
import random
import sys
import uuid
import shutil
import collections
import uuid
from pathos.multiprocessing import ProcessPool
from pathos.helpers import mp
from tftd import TFDC
from config import *
from cnf_util import cadical, drat_trim, gen_tfdc, parse_core, parse_drat, lemma_occ, del_occ, validate_TFDC, data_writer

TESTS = []

def is_test(f):
  TESTS.append(f)
  return f

def run_tests():
  for t in TESTS:
    t()
  print("tests ok")


def sample_SR_aux(n, min_cls_len=1, p_binom=0.7, p_geo=0.4):
  """
  Args:
  n: positive integer

  Returns:
  A randomly-generated formula and a clause which makes the formula unsat

  This procedure has no guarantees on the number of clauses in the formula.

  Reference implementation in source code of NeuroSAT:
  https://github.com/dselsam/neurosat/blob/master/python/gen_sr_dimacs.py
  """
  result = CNF()
  with Solver(name="cdl") as S:
    while True:
      k = min_cls_len + np.random.binomial(n=1, p=p_binom) + np.random.geometric(p_geo)
      vs = np.random.choice(n,size=min(n,k),replace=False)
      vs = [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]
      S.add_clause(vs)
      if S.solve():
        result.append(vs)
      else:
        break

  return result, vs

def sample_SR(n, min_cls_len=1, p_binom=0.7, p_geo=0.4):
  """
  Args: same as sample_SR_aux

  Returns:
  1. an unsat formula and sat formula, differing only by the sign of a flipped literal in their final clause
  2. clause and literal index of the flip literal
  """
  result_unsat, vs = sample_SR_aux(n,min_cls_len,p_binom,p_geo)
  result_sat = result_unsat.copy()
  result_unsat.append(vs)
  flip_idx = np.random.choice(len(vs))
  result_sat.append(vs[0:flip_idx] + [-vs[flip_idx]] + vs[flip_idx + 1:])
  return result_unsat, result_sat, len(result_sat.clauses)-1, flip_idx
  

@is_test
def sample_SR_test(num_vars=10, num_rounds=100):
  def sample_SR_test_aux(n):
    result_unsat, result_sat, c_idx, l_idx = sample_SR(n)
    with Solver(name="cdl") as S:
      S.append_formula(result_unsat)
      assert S.solve() is False

    with Solver(name="cdl") as S:
      S.append_formula(result_sat)
      assert S.solve() is True

    with Solver(name="cdl") as S:
      result_sat.clauses[c_idx][l_idx] *= -1
      S.append_formula(result_sat)
      assert S.solve() is False

    print("ok")

  for _ in range(num_rounds):
    sample_SR_test_aux(num_rounds)

def sample_SRC_aux(n,u1,c_idx,l_idx,p_geo=0.4,p_binom=0.7):
  """
  Args:
  n: positive integer
  u1: an unsat core
  c_idx: a clause index
  l_idx: a literal index
  
  u1 must become sat if the literal at (c_idx, l_idx) is flipped.

  Note that if result, vs = sample_SR_aux(args...), then result + vs is a valid argument for u1

  Returns: a random formula drawn from n variables containing u1 as an unsat core, and the unsat core
  """
  result = CNF()
  u2 = u1.copy()
  u2.clauses[c_idx][l_idx] = -u2.clauses[c_idx][l_idx]
  with Solver(name="cdl") as S:
    while True:
      S.append_formula(u2)
      k = 1 + np.random.binomial(n=1, p=p_binom) + np.random.geometric(p_geo)
      vs = np.random.choice(n,size=min(n,k),replace=False)
      vs = [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]
      S.add_clause(vs)
      if S.solve():
        result.append(vs)
      else:
        break
    for cls in u1.clauses:
      result.append(cls)
  return result, u1 # TODO(jesse): output a core clause mask

def unsat_core_example():
  fmla = CNF()
  fmla.append([1,2,3])
  fmla.append([-1,2,3])
  fmla.append([2,-3])
  fmla.append([-2,-3])
  fmla.append([-2,3])
  return fmla

@is_test
def u_test():
  c_idx = 4
  l_idx = 0
  fmla = unsat_core_example()
  fmla2 = unsat_core_example()
  fmla2.clauses[c_idx][l_idx] = -(fmla2.clauses[c_idx][l_idx])
  # is this an example of a blocked clause?
  with Solver(name="cdl") as S:
    S.append_formula(fmla)
    result = S.solve()
    assert result is False
    
  with Solver(name="cdl") as S:
    S.append_formula(fmla2)
    result = S.solve()
    assert result is True

  print("ok")

@is_test  
def sample_SRC_aux_test1():
  u1 = unsat_core_example()
  c_idx = 4
  l_idx = 0
  n = 10
  for _ in range(100):
    fmla, u1 = sample_SRC_aux(n,u1,c_idx,l_idx)
    fmla.to_fp(sys.stdout)
    with Solver(name="cdl") as S:
      S.append_formula(fmla)
      result = S.solve()
      assert result is False
    print("ok")

@is_test
def sample_SRC_aux_test2():
  core_sizes = []
  ratios = []
  valid_count = 0
  num_rounds = 200
  for _ in range(num_rounds):
    fmla_unsat, fmla_sat, c_idx, l_idx = sample_SR(10)
    # fmla_unsat.to_fp(sys.stdout)
    fmla_src, u  = sample_SRC_aux(100, fmla_unsat, c_idx, l_idx)
    with Solver(name="cdl") as S:
      S.append_formula(fmla_src)
      result = S.solve()
      assert result is False
    core_size = len(fmla_unsat.clauses)
    formula_core_size_ratio = float(len(fmla_src.clauses)/core_size)
    ratios.append(formula_core_size_ratio)
    core_sizes.append(core_size)
    if core_size <= 100 and 50 <= core_size and formula_core_size_ratio <= 20 and 5 <= formula_core_size_ratio:
      valid_count += 1
    print("ratio of fmla_src to fmla_unsat: ", formula_core_size_ratio)
    print("ok")
  print("mean fmla-core ratio:", np.mean(ratios))
  print("fmla-core ratio variance:", np.var(ratios))
  print("min fmla-core ratio", np.min(ratios))
  print("max fmla-core ratio", np.max(ratios))
  print("mean core size:", np.mean(core_sizes))
  print("core size variance:", np.var(core_sizes))
  print("max core size:", np.max(core_sizes))
  print("min core size:", np.min(core_sizes))
  print("percent valid datapoints:", float(valid_count/num_rounds))

def sample_SRC_aux_test3(core_min=20, core_max=100, ratio_min=5, ratio_max=20, n1=10, n2=100):
  core_sizes = []
  ratios = []
  valid_count = 0
  num_rounds = 200
  with tempfile.TemporaryDirectory() as tmpdir:
    cnfdir = tmpdir + "/"
    cdl = cadical(cnf_dir =cnfdir)
    drat = drat_trim(cnf_dir = cnfdir)
    for _ in range(num_rounds):
      name = str(uuid.uuid4())
      fmla_unsat, fmla_sat, c_idx, l_idx = sample_SR(n1,2,p_binom=0.3)
      with open(os.path.join(cnfdir, name + ".cnf"), "w") as f:
        fmla_unsat.to_fp(f)
      cdl.set_rootname(name)
      cdl.run()   # compute a proof of unsat for this formula
      cdl.process_output()
      assert cdl.result is False
      cdl.write_proof()

      drat.set_rootname(name)
      drat.run()   # extract an unsat core by calling DRAT-trim

      tsr = parse_drat(drat.opt_path, fmla_unsat.nv)
      var_lemma_counts = lemma_occ(tsr)
      var_del_counts = del_occ(tsr)

      masks = parse_core(drat.core_path, fmla_unsat.nv, len(fmla_unsat.clauses))
      core_clause_mask = masks["core_clause_mask"]
      u = CNF()  # this unsat core should still satisfy the property that it becomes sat if the sign of a single literal is flipped
      for i in range(len(core_clause_mask)):
        if core_clause_mask[i] == 1:
          u.append(fmla_unsat.clauses[i])
      l_idx = u.clauses[-1].index(fmla_unsat.clauses[c_idx][l_idx]) # get new l_idx since DRAT sometimes permutes literals inside clauses

      fmla_src, u = sample_SRC_aux(n2, u, len(u.clauses)-1, l_idx)   # use this unsat core as the seed to a call to sample_SR_aux, and obtain fmla_src

      core_size = len(u.clauses)

      ratio = float(len(fmla_src.clauses)/core_size)

      core_sizes.append(core_size)
      ratios.append(ratio)

      if core_size <= core_max and core_min <= core_size and ratio <= ratio_max and ratio_min <= ratio:
        valid_count += 1
  print("max/min/mean core size:", np.max(core_sizes), np.min(core_sizes), np.mean(core_sizes))
  print("max/min/mean ratios:", np.max(ratios), np.min(ratios), np.mean(ratios))      
  print("percent valid datapoints: ", str(float(valid_count/num_rounds)*100) + "%")

def gen_src_aux(core_min=20, core_max=100, ratio_min=5, ratio_max=20, n1=10, n2=100, min_cls_len=2, cnfdir = None):
  """
  Repeatedly samples formulas from SRC until it finds one which meets the specifications, then returns that formula.

  These parameters need to be tuned before running at scale to ensure that each call to this function rarely needs more than one iteration.

  Args:
  core_min: minimum size of the unsat core of the formula
  core_max: maximum size of the unsat core of the formula
  ratio_min: minimum ratio of formula-size to unsat-core-size
  ratio_max: maximum ratio of formula-size to unsat-core-size
  n1: `n` parameter passed to sample_SR when sampling the unsat core
  n2: `n` parameter passed to sample_SRC when sampling the larger formula containing the unsat core 

  Returns:
  A formula from SRC obeying the constraints given by `core_min`, `core_max`, `ratio_min`, and `ratio_max`.
  """
  # with tempfile.TemporaryDirectory() as cnfdir:
  cdl = cadical(cnf_dir = cnfdir)
  drat = drat_trim(cnf_dir=cnfdir)
  while True:
    name = str(uuid.uuid4())
    fmla_unsat, fmla_sat, c_idx, l_idx = sample_SR(n1,min_cls_len,p_binom=0.3)
    with open(os.path.join(cnfdir, name + ".cnf"), "w") as f:
      fmla_unsat.to_fp(f)
    cdl.set_rootname(name)
    cdl.run()   # compute a proof of unsat for this formula
    cdl.process_output()
    assert cdl.result is False
    cdl.write_proof()

    drat.set_rootname(name)
    drat.run()   # extract an unsat core by calling DRAT-trim

    tsr = parse_drat(drat.opt_path, fmla_unsat.nv)
    var_lemma_counts = lemma_occ(tsr)
    var_del_counts = del_occ(tsr)

    masks = parse_core(drat.core_path, fmla_unsat.nv, len(fmla_unsat.clauses))
    core_clause_mask = masks["core_clause_mask"]
    u = CNF()  # this unsat core should still satisfy the property that it becomes sat if the sign of a single literal is flipped
    for i in range(len(core_clause_mask)):
      if core_clause_mask[i] == 1:
        u.append(fmla_unsat.clauses[i])
    l_idx = u.clauses[-1].index(fmla_unsat.clauses[c_idx][l_idx]) # get new l_idx since DRAT sometimes permutes literals inside clauses

    fmla_src, u = sample_SRC_aux(n2, u, len(u.clauses)-1, l_idx)   # use this unsat core as the seed to a call to sample_SR_aux, and obtain fmla_src

    core_size = len(u.clauses)

    ratio = float(len(fmla_src.clauses)/core_size)

      # if fmla_src satisfies the constraints, return the TFDC
    if (ratio_min <= ratio and ratio <= ratio_max and core_min <= core_size and core_size <= core_max):
      break
    else:
      continue
  return fmla_src

def gen_src(core_min=20, core_max=100, ratio_min=5, ratio_max=20, n1=10, n2=100, min_cls_len=2, cnfdir = None):
  if cnfdir is None:
    with tempfile.TemporaryDirectory() as tmpdir:
      return gen_src_aux(core_min, core_max, ratio_min, ratio_max, n1, n2, min_cls_len, tmpdir)
  else:
    return gen_src_aux(core_min, core_max, ratio_min, ratio_max, n1, n2, min_cls_len, cnfdir)
  
# def gen_src_tfdc(core_min=20, core_max=50, ratio_min=5, ratio_max=20, n1=10, n2=100, min_cls_len=2, cnfdir=None):
#   """
#   Repeatedly samples formulas from SRC until it finds one which meets the specifications, then returns that formula's serialization as a `TFDC`.

#   These parameters need to be tuned before running at scale to ensure that each call to this function rarely needs more than one iteration.

#   Args:
#   core_min: minimum size of the unsat core of the formula
#   core_max: maximum size of the unsat core of the formula
#   ratio_min: minimum ratio of formula-size to unsat-core-size
#   ratio_max: maximum ratio of formula-size to unsat-core-size
#   n1: `n` parameter passed to sample_SR when sampling the unsat core
#   n2: `n` parameter passed to sample_SRC when sampling the larger formula containing the unsat core 

#   Returns:
#   A `TFDC` serializing a formula obeying the constraints given by `core_min`, `core_max`, `ratio_min`, and `ratio_max`.
#   """
#   if cnfdir is None:
#     raise Exception("no CNF directory specified")
#   cdl = cadical(cnf_dir = cnfdir)
#   drat = drat_trim(cnf_dir=cnfdir)
#   while True:
#     name = str(uuid.uuid4())
#     fmla_unsat, fmla_sat, c_idx, l_idx = sample_SR(n1,min_cls_len,p_binom=0.3)
#     with open(os.path.join(cnfdir, name + ".cnf"), "w") as f:
#       fmla_unsat.to_fp(f)
#     cdl.set_rootname(name)
#     cdl.run()   # compute a proof of unsat for this formula
#     cdl.process_output()
#     assert cdl.result is False
#     cdl.write_proof()

#     drat.set_rootname(name)
#     drat.run()   # extract an unsat core by calling DRAT-trim

#     tsr = parse_drat(drat.opt_path, fmla_unsat.nv)
#     var_lemma_counts = lemma_occ(tsr)
#     var_del_counts = del_occ(tsr)
    
#     masks = parse_core(drat.core_path, fmla_unsat.nv, len(fmla_unsat.clauses))
#     core_clause_mask = masks["core_clause_mask"]
#     u = CNF()  # this unsat core should still satisfy the property that it becomes sat if the sign of a single literal is flipped
#     for i in range(len(core_clause_mask)):
#       if core_clause_mask[i] == 1:
#         u.append(fmla_unsat.clauses[i])
#     l_idx = u.clauses[-1].index(fmla_unsat.clauses[c_idx][l_idx]) # get new l_idx since DRAT sometimes permutes literals inside clauses

#     fmla_src, u = sample_SRC_aux(n2, u, len(u.clauses)-1, l_idx)   # use this unsat core as the seed to a call to sample_SR_aux, and obtain fmla_src

#     core_size = len(u.clauses)

#     ratio = float(len(fmla_src.clauses)/core_size)

#       # if fmla_src satisfies the constraints, return the TFDC
#     if (ratio_min <= ratio and ratio <= ratio_max and core_min <= core_size and core_size <= core_max):
#       break
#     else:
#       continue

#   # don't recompute an unsat proof, just re-use knowledge of the core
#   new_core_clause_mask = np.zeros(shape=[len(fmla_src.clauses)], dtype="int32")
#   for i in range(1, len(u.clauses)+1):
#     new_core_clause_mask[-i] = 1

#   def mask_pad(mask, n_var):
#     if n_var <= len(mask):
#       return mask
#     else:
#       return np.pad(mask, (0, n_var-len(mask)), "constant", constant_values=(0,0))

#   # compute result
#   tfdc = TFDC(
#     n_vars = fmla_src.nv,
#     n_clauses = len(fmla_src.clauses),
#     CL_idxs = clgraph(fmla_src)
#     core_var_mask = mask_pad(masks["core_var_mask"], fmla_src.nv),
#     core_clause_mask = new_core_clause_mask,
#     var_lemma_counts = mask_pad(var_lemma_counts, fmla_src.nv),
#     var_del_counts = mask_pad(var_del_counts, fmla_src.nv)
#   )
  
#   return tfdc

@is_test
def test_gen_src_tfdc():
  with tempfile.TemporaryDirectory() as tmp:
    cnfdir = tmp + "/"
    tfdcs = [gen_src_tfdc(cnfdir = cnfdir) for _ in range(20)]
    for tfdc in tfdcs:
      validate_TFDC(tfdc)
