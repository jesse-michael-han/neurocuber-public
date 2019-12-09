import os
from cnf_util import *
from pysat.formula import CNF
from pysat.solvers import Solver
from neurocuber import init_neurocuber
import tensorflow as tf
import numpy as np
import time
import random
from gen_fmlas import get_unsat_randkcnf
import ray
from query_model import NeuroResQuery
import re
from tftd import example_to_tfdc, TFDC
from Z3 import *
from util import *


class VarSelector:
  def __init__(self):
    pass

  def call(self, fmla):
    raise NotImplementedError

  def get_top_logit(self, fmla):
    raise NotImplementedError

  def __call__(self, fmla, **kwargs): # returns sorted list of variables, *not* indices.
    return self.call(fmla, **kwargs)

class RandomVarSelector(VarSelector):
  def __init__(self):
    super(RandomVarSelector, self).__init__()

  def call(self, fmla):
    result = list(range(1, fmla.nv+1))
    random.shuffle(result)
    return result

  def get_top_logit(self, fmla):
    return self.call(fmla)[0]

class NeuralDRATVarSelector(VarSelector):
  def __init__(self, cfg_path, **kwargs):
    super(NeuralDRATVarSelector, self).__init__()
    self.model = NeuroResQuery(cfg_path, **kwargs)

  def get_var_ps(self, fmla):
    return tf.nn.softmax(self.model(fmla)[0])

  def call(self, fmla):
    ps = self.get_var_ps(fmla)
    return np.argsort(ps, kind="mergesort")[::-1] + 1

  def get_top_logit(self, fmla):
    return self.call(fmla)[0]

class NeuralCoreVarSelector(VarSelector):
  def __init__(self, cfg_path, **kwargs):
    super(NeuralCoreVarSelector, self).__init__()
    self.model = NeuroResQuery(cfg_path, **kwargs)

  def get_var_ps(self, fmla):
    return tf.nn.softmax(self.model(fmla)[1])

  def call(self, fmla):
    ps = self.get_var_ps(fmla)
    return np.argsort(ps, kind="mergesort")[::-1] + 1

  def get_top_logit(self, fmla):
    return self.call(fmla)[0]

class Z3VS(VarSelector):
  def __init__(self, **kwargs):
    super(Z3VS, self).__init__()
    self.model = Z3VarSelector()
    self.failure_count = 0

  def call(self, fmla):
    result = self.model(fmla)
    result_aux = list(range(1, fmla.nv +1))
    random.shuffle(result_aux)
    if result is not None:
      return [result] + result_aux
    else:
      self.failure_count += 1
      return result_aux

  def get_top_logit(self, fmla):
    return self.call(fmla)[0]

class ConstantVarSelector(VarSelector):
  def __init__(self, **kwargs):
    super(ConstantVarSelector, self).__init__()

  def call(self, fmla):
    return list(range(1, fmla.nv+1))

  def get_top_logit(self, fmla):
    return self.call(fmla)[0]

def test_Z3VS():
  z3 = Z3VS()
  for _ in range(10):
    fmla = get_unsat_randkcnf(3,100)
    v = z3(fmla)
    print(v[0])

  print(z3.failure_count)

def test_ConstantVarSelector():
  vs = ConstantVarSelector()
  for _ in range(10):
    fmla = get_unsat_randkcnf(3,20)
    cubevars = vs(fmla)
    print(cubevars[0:5])

def mk_cube(fmla, vs, binary_sequence):
  """
  Args:
  fmla: CNF object
  vs: list of variables
  binary_sequence: a list or tuple of 0s and 1s of same length as vs

  Returns: a subformula given by simplifying fmla according to zip(vs, binary_sequence)
  """
  assumptions = [v if eps else -v for v, eps in zip(vs, binary_sequence)]
  cube = simplify_CNF(fmla, assumptions)
  return cube

class CubedFormula(CNF):
  def __init__(self, fmla, vs):
    super(CubedFormula, self).__init__(from_clauses=fmla.clauses)
    self.cube_vars = vs

  # returns a generator enumerating the cubes
  def __iter__(self):
    return (mk_cube(self, self.cube_vars, binary_sequence) for binary_sequence in itertools.product([True,False], repeat=len(self.cube_vars)))

def offline_uniform_cubing(fmla, N_V, VS):
  """
  Args:
  fmla: CNF object
  N_K: number of variables. will produce 2**N_K cubes.
  VS: a VarsSelector object.
  """
  # K = int(np.floor(np.log(N_V)*(1/np.log(2.)) + 1e-5))
  K = N_V
  try:
    assert K <= fmla.nv
  except AssertionError:
    raise Exception("K is too large, exceeds n_vars")

  ranked_vars = VS(fmla)

  cube_vars = []
  # pop the top ranked var, as soon as it appears in the formula, place it in cube_vars
  count = 0
  while len(cube_vars) < K:
    FOUND_FLAG = False
    top_var = ranked_vars[count]
    for cls in fmla.clauses:
      if FOUND_FLAG:
        break
      for lit in cls:
        if top_var == abs(lit):
          cube_vars.append(top_var)
          FOUND_FLAG = True
          break
    count += 1

  return CubedFormula(fmla, cube_vars)

def test_offline_uniform_cubing():
  clauses =[ [-5,-6,7],
             # [5,6,-7],
             [10,11,-12] ]
  fmla = CNF(from_clauses=clauses)

  vs = RandomVarSelector()

  cubed_fmla = offline_uniform_cubing(fmla, 3, vs)

  for v in cubed_fmla.cube_vars:
    assert v in [5,6,7,10,11,12]

  print("ok")

@ray.remote
def conqueror(fmla, solver):
  with Solver(name=solver, bootstrap_with=fmla) as S:
    start = time.time()
    S.solve()
    elapsed = time.time() - start
    return elapsed

class ParallelCubeEvaluator:
  """
  Given a CubedFormula, records the cube sizes and CDCL runtimes
  """
  def __init__(self, cfmla, solver="cadical"):
    self.cfmla = cfmla
    self.solver = solver
    self.runtimes = []
    self.cube_sizes = []

  def conquer(self):
    jobs = []
    for cube in self.cfmla:
      self.cube_sizes.append(len(cube.clauses))
      S = conqueror.remote(cube, self.solver)
      jobs.append(S)
    self.runtimes = ray.get(jobs)
    return np.mean(np.array(self.cube_sizes, dtype="float32")), np.mean(np.array(self.runtimes, dtype="float32"))

class SequentialCubeEvaluator:
  """
  Given a CubedFormula, records the cube sizes and *single threaded* averaged CDCL runtime

  Solving is not incremental. The solver is reinstantiated for every cube.
  """
  def __init__(self, cfmla, solver="cadical"):
    self.cfmla = cfmla
    self.solver = solver
    self.runtimes = []
    self.cube_sizes = []

  def conquer(self):
    for cube in self.cfmla:
      with Solver(name=self.solver, bootstrap_with=cube) as S:
        start = time.time()
        S.solve()
        elapsed = time.time() - start
        self.runtimes.append(elapsed)
        self.cube_sizes.append(len(cube.clauses))
    return np.mean(np.array(self.cube_sizes, dtype="float32")), np.mean(np.array(self.runtimes, dtype="float32"))  

class AccumulateMean:
  def __init__(self):
    self.count = 0
    self.value = 0

  def add(self, new_value):
    self.value = (1./(self.count+1)) * ((self.value * (self.count)) + new_value)
    self.count += 1

  def get(self):
    return self.value

def cubing_eval_from_CNF(cfg_path, drat_head, n_vars, cpus, data_dir, experiment_name, limit=-1, test=False):
  ray.init(num_cpus=cpus, object_store_memory=3e9)
  if not test:
    if data_dir is None:
      raise Exception("--data-dir must be specified")
    path_to_folder = data_dir
    cnf_files = files_with_extension(path_to_folder, "cnf")

  if cfg_path == "random":
    VS = RandomVarSelector()
    name = "random"
  elif cfg_path == "z3":
    if not n_vars == 1:
      raise Exception("cubing with more than one variable not supported with Z3")
    VS = Z3VS()
    name = "z3"
  else:
    if drat_head:
      VS = NeuralDRATVarSelector(cfg_path)
      name = VS.model.model.name + "_drat"
    else:
      VS = NeuralCoreVarSelector(cfg_path)
      name = VS.model.model.name

  if test:
    log_dir = os.path.join(f"test_cubing_eval/{experiment_name}/",name + "/")
  else:
    log_dir = os.path.join(f"cubing_eval/{experiment_name}/",name + "/")

  check_make_path(log_dir)
  writer = tf.summary.create_file_writer(log_dir)

  MeanRuntime = AccumulateMean()
  MeanCubeSize = AccumulateMean()
  step_count = 0
  if test:
    for _ in range(10):
      fmla = get_unsat_randkcnf(3,100)
      cube_size, runtime = ParallelCubeEvaluator(offline_uniform_cubing(fmla, n_vars, VS)).conquer()
      MeanRuntime.add(runtime)
      MeanCubeSize.add(cube_size)
      avg_runtime = MeanRuntime.get()
      avg_cube_size = MeanCubeSize.get()
      with writer.as_default():
        tf.summary.scalar("avg runtime", runtime, step=(step_count+1))
        tf.summary.scalar("avg avg runtime", avg_runtime, step=(step_count+1))
        tf.summary.scalar("avg cube size", cube_size, step=(step_count+1))
        tf.summary.scalar("avg avg cube size", avg_cube_size, step=(step_count+1))
      step_count += 1
  else:
    for cnf_file in cnf_files:
      fmla = CNF(from_file=cnf_file)
      cube_size, runtime = ParallelCubeEvaluator(offline_uniform_cubing(fmla, n_vars, VS)).conquer()
      MeanRuntime.add(runtime)
      MeanCubeSize.add(cube_size)
      avg_runtime = MeanRuntime.get()
      avg_cube_size = MeanCubeSize.get()
      with writer.as_default():
        tf.summary.scalar("avg runtime", runtime, step=(step_count+1))
        tf.summary.scalar("avg avg runtime", avg_runtime, step=(step_count+1))
        tf.summary.scalar("avg cube size", cube_size, step=(step_count+1))
        tf.summary.scalar("avg avg cube size", avg_cube_size, step=(step_count+1))
      step_count += 1
      if limit > 0:
        if step_count > limit:
          break


# python cubing_eval.py EXPERIMENT_NAME CFG_PATH DATA_DIR N_VARS DRAT_HEAD_FLAG CPUS LIMIT TEST_FLAG
# example usage:
# python cubing_eval.py test2 z3 --n-vars=1 --test --cpus=1
# python cubing_eval.py test2 random --n-vars=1 --cpus=2 --test
# python cubing_eval.py test2 res_models/res_grid_2_sr.json --n-vars=1 --drat-head --cpus=1 --test
# python cubing_eval.py foo res_models/res_grid_2_sr.json --n-vars=3 --cpus=4
# python cubing_eval.py top1schur random --n-vars=1 cpus=2
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment_name", action="store")
  parser.add_argument("cfg_path", action="store")
  parser.add_argument("--data-dir", action="store", type=str, dest="data_dir", default=None)
  parser.add_argument("--drat-head", action="store_true", dest="drat_head")
  parser.add_argument("--test", action="store_true", dest="test")
  parser.add_argument("--n-vars", action="store", type=int, dest="n_vars")
  parser.add_argument("--cpus", action="store", type=int, dest="cpus")
  parser.add_argument("--limit", action="store", type=int, default=-1)
  opts = parser.parse_args()

  cubing_eval_from_CNF(**vars(opts))
