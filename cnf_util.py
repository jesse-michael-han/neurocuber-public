import numpy as np
import tempfile
import os
import subprocess
import itertools
import random
import sys
import uuid
import shutil
import collections
from io import StringIO
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
import tensorflow as tf
from tftd import TFDC, tfdc_to_example, TFDCR, tfdcr_to_example
from config import *

def CNF_to_string(cnf):
  with StringIO() as result_buffer:
    cnf.to_fp(result_buffer)
    result = result_buffer.getvalue()
  return result

# create a pysat CNF object from a cnfformula CNF object
class FromCNFFormula(CNF):
  def __init__(self, fmla):
    super(FromCNFFormula, self).__init__()
    for cls in fmla._clauses:
      self.append(list(cls))

def randomly_flip(xs):
  return [random.choice([1,-1]) * x for x in xs]

def RandKCNF(k,n,m):
  clauses = [randomly_flip(random.sample(range(1, int(n) + 1), k)) for _ in range(int(m))]
  return CNF(from_clauses=clauses)

def sample_randkcnf(k,n,alpha=4.4,eps=0.05):
  a = np.random.uniform(low=alpha-eps, high=alpha+eps, size=None)
  m = np.ceil(n*a)
  return RandKCNF(k,n,m)

def parse_drat(filename=None, n_vars=None, from_string=None):
  """
  Args:
  filename: path to an optimized DRAT proof outputted by DRAT-trim
  n_vars: number of variables in the original CNF
  
  Returns:
  A tensor of shape [n_vars, 2, 2], i.e.
  (n_vars × (lemma, deletion) × (pos occurrences, neg occurrences))
  """
  assert ((filename is not None) or (from_string is not None)) and type(n_vars) is int
  result = np.zeros(shape=[n_vars,2,2], dtype="int32")
  if filename is not None:
    f = open(filename, "r")
  else:
    f = StringIO(from_string)
  for line in f:
    del_index = 0
    l = line.split()
    for x in l:
      if x is "d":
        del_index = 1
        continue
      if x is "0":
        continue
      x = int(x)
      pos_index = 0
      if np.sign(x) < 0:
        pos_index = 1
      # print(pos_index)
      # print(f"inserting at {abs(x) - 1}, {del_index}, {pos_index}")
      result[abs(x)-1, del_index, pos_index] += 1
  f.close()
  return result

def lemma_occ(tsr):
  n_vars = tsr.shape[0]
  result = np.zeros(shape=[n_vars])
  for idx in range(n_vars):
    result[idx] = np.sum(tsr[idx, 0, :])
  return result

def del_occ(tsr):
  n_vars = tsr.shape[0]
  result = np.zeros(shape=[n_vars])
  for idx in range(n_vars):
    result[idx] = np.sum(tsr[idx, 1, :])
  return result

def parse_core(filename=None, n_vars=None, n_clauses=None, from_string=None):
  """
  Args:
  filename: path to core lemmas outputted by modified drat-trim
  n_vars: number of variables

  Returns:
  A dict of the form
    { "core_clause_mask" : _,
      "core_var_mask" : _ }
  where the values are int32 numpy arrays.
  """
  assert ((filename is None) or (from_string is None)) and type(n_vars) is int and type(n_clauses) is int
  core_clause_mask = np.zeros(shape=[n_clauses], dtype="int32")
  core_var_mask = np.zeros(shape=[n_vars], dtype="int32")
  if filename is not None:
    f = open(filename, "r")
  else:
    f = StringIO(from_string)
  f.readline()
  for line in f:
    reading_lits = True
    l = line.split()
    for x in l:
      if reading_lits:
        if x is "0":
          reading_lits = False
          continue
      x = abs(int(x))
      if reading_lits:
        assert x is not 0
        core_var_mask[x-1] = 1
      else:
        core_clause_mask[x] = 1
  f.close()
  return {
    "core_var_mask" : core_var_mask,
    "core_clause_mask" : core_clause_mask
  }

def test_parsing():
  example_opt = "d 1 2 3 \n 1 4 5 \n d -4 3 \n "
  print("example optimized proof:")
  print(example_opt)

  example_core = " p cnf 9 2 \n 1 3 2 0 0 \n -1 -3 -9 0 5    "
  print("example core:")
  print(example_core)
  
  drat_result = parse_drat(None, 10, from_string=example_opt)
  core_result = parse_core(None, 10, 10, from_string=example_core)

  print("parsed lemma occurrences:")
  print(lemma_occ(drat_result))

  print("parsed deletion occurrences:")
  print(del_occ(drat_result))

  assert lemma_occ(drat_result)[2] == 0 # 3 only occurs in deletion clauses
  assert lemma_occ(drat_result)[0] == 1
  assert lemma_occ(drat_result)[3] == 1
  assert del_occ(drat_result)[3] == 1

  print("core_var_mask:")
  print(core_result["core_var_mask"])

  for i in [0,1,2,8]:
    assert (core_result["core_var_mask"][i] == 1)

  print("core_clause_mask:")
  print(core_result["core_clause_mask"])

  for i in [0,5]:
    assert (core_result["core_clause_mask"][i] == 1)
  
  
  # assert lemma_occ(drat_result)[0] == 0 # 1 and -1 occurs only in deletion clauses
  # assert 

class drat_trim:
  def __init__(self, rootname=None, drat_trim_path=os.path.join(TOOLS_DIR, "drat-trim"), cnf_dir=None, no_binary=False):
    self.drat_trim_path = drat_trim_path
    self.cnf_dir = cnf_dir
    self.cnf_dir = cnf_dir
    if rootname is not None:
      self.set_rootname(rootname)

  def set_rootname(self,rootname):
    self.rootname = rootname
    self.cnf_path = os.path.join(self.cnf_dir, rootname + ".cnf")
    self.drat_path = os.path.join(self.cnf_dir, rootname + ".drat")
    self.core_path = os.path.join(self.cnf_dir, rootname + ".core")
    self.opt_path = os.path.join(self.cnf_dir, rootname + ".opt")

  def run(self):
    drat_trim_command = [self.drat_trim_path, self.cnf_path, self.drat_path, "-c", self.core_path, "-l", self.opt_path]
    result = subprocess.run(drat_trim_command, capture_output=True)
    print(result.stdout)
    # TODO(jesse): parse output and add assertion that verification succeeeded
    
class cadical:
  def __init__(self,rootname=None,cadical_path=CADICAL_PATH,cnf_dir=None, no_binary=False):
    self.cadical_path = cadical_path
    self.cnf_dir = cnf_dir
    if rootname is not None:
      self.set_rootname(rootname)
    self.output=None
    self.no_binary=no_binary
    self.proof=None
    self.result=None

  def run(self):
    if self.no_binary:
      cadical_command = [self.cadical_path, "-q", "-n", "--no-binary", self.cnf_path, "-"]
    else:
      cadical_command = [self.cadical_path, "-q", "-n", self.cnf_path, "-"]
    out = subprocess.run(cadical_command, capture_output=True)
    self.output = out.stdout
    
  def process_output(self):
    # check if result is sat or unsat
    if self.no_binary:
      if self.output[0] == 's':
        self.result = True
      else:
        self.result = False
        self.proof=self.output[0:-16]
    else:
      if self.output[-14] == 115:
        self.result = True
      else:
        self.result = False
        self.proof=self.output[0:-16]

  def write_proof(self):
    try:
      assert self.proof is not None
      if self.no_binary:
        mode = "w"
      else:
        mode = "wb"
      with open(self.drat_path, mode) as f:
        f.write(self.proof)
    except AssertionError as e:
      print("Warning: formula is sat, proof is None")

  def set_rootname(self,rootname):
    self.rootname = rootname
    self.cnf_path = os.path.join(self.cnf_dir, rootname + ".cnf")
    self.drat_path = os.path.join(self.cnf_dir, rootname + ".drat")
    self.output=None
    self.proof=None
    self.result=None

DataWriterOpts = collections.namedtuple("DataWriterOpts", ["data_dir", "n_tfrs_per_file"])

def mk_data_writer_opts(data_dir, n_tfrs_per_file):
  return DataWriterOpts(data_dir=data_dir, n_tfrs_per_file=n_tfrs_per_file)

class data_writer:
  def __init__(self, opts):
      self.data_dir = opts.data_dir
      self.n_tfrs_per_file = opts.n_tfrs_per_file
      self.tmpdir = tempfile.TemporaryDirectory()
      self._next_file()

  def _next_file(self):
      self.n_writes = 0
      self.n_files = 0
      self.outfile = "file%d_%s.tfr" % (self.n_files, str(uuid.uuid4()))
      self.outfile_path = os.path.join(self.tmpdir.name, self.outfile)
      self.n_files += 1
      # tfropts = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)
      self.writer   = tf.io.TFRecordWriter(os.path.join(self.tmpdir.name, self.outfile), options="GZIP")

  def _move_file(self):
      self.writer.flush()
      self.writer.close()
      shutil.move(self.outfile_path, os.path.join(self.data_dir, self.outfile)) # TODO(jesse): test this setup with a normal writer

  def finalize(self):
      # util.log(kind='info', author='tfwriter', msg="finalize: %s" % str(self.n_writes))
      if self.n_writes > 0:
          # util.log(kind='info', author='tfwriter', msg="moving last file #%d (%s)" % (self.n_files, self.outfile))
          self._move_file()

  def write_example(self, example):
    self.writer.write(example.SerializeToString())
    self.n_writes += 1
    if self.n_writes == self.n_tfrs_per_file:
      # util.log(kind='info', author='tfwriter', msg="file #%d ready (%s)" % (self.n_files, self.outfile))
      self._move_file()
      self._next_file()    

  def write_tfdc(self, tfdc):
    self.write_example(tfdc_to_example(tfdc))

  def write_tfdcr(self, tfdcr):
    self.write_example(tfdcr_to_example(tfdcr))

def clgraph(fmla):
  """
  Args:
  fmla: CNF object
  Returns:
  an array of edges in the clause-literal graph
  """
  result = set()
  for cls_idx in range(len(fmla.clauses)):
    for lit in fmla.clauses[cls_idx]:
      if np.sign(lit) >= 0:
        lit_enc = lit - 1
      else:
        lit_enc = fmla.nv + abs(lit) - 1
      result.add((cls_idx, lit_enc))
  return np.array([list(x) for x in result], dtype="int32")

def gen_tfdc(fmla, cnfdir,log=None):
  fmla = fmla
  n_vars = fmla.nv
  n_clauses = len(fmla.clauses)
  name = str(uuid.uuid4())
  cdl = cadical(cnf_dir=cnfdir)
  drat = drat_trim(cnf_dir=cnfdir)
  with open(cnfdir + name + ".cnf", "w") as f:
    fmla.to_fp(f)
  cdl.set_rootname(name)
  if log is not None:
    log.write(cdl.cnf_path + "\n")
  cdl.run()
  if log is not None:
    log.write(str(cdl.output) + "\n")
  cdl.process_output()
  if cdl.result:
    return None

  cdl.write_proof()

  drat.set_rootname(name)
  drat.run()
  if log is not None:
    log.write (drat.drat_trim_path + "\n")
    log.write (drat.opt_path + "\n")

  tsr = parse_drat(drat.opt_path, n_vars)

  var_lemma_counts = lemma_occ(tsr)
  var_del_counts = del_occ(tsr)

  masks = parse_core(drat.core_path, n_vars, n_clauses)

  tfdc = TFDC(
    n_vars = n_vars,
    n_clauses = n_clauses,
    CL_idxs = clgraph(fmla),
    core_var_mask = masks["core_var_mask"],
    core_clause_mask = masks["core_clause_mask"],
    var_lemma_counts = var_lemma_counts,
    var_del_counts = var_del_counts
  )  
  return tfdc
  
def gen_randkcnf_data(opts):
  i = 0
  n_datapoints = opts.n_datapoints
  wtr = data_writer(opts)
  logfile = log_dir + "log.txt"
  with tempfile.TemporaryDirectory() as tmpdir:
    with open(logfile, "a") as log:
      cnfdir = tmpdir + "/"
      while i < n_datapoints:
        if log is not None:
          log.write("step number " + str(i) + "\n")
        fmla = sample_randkcnf(opts.k, opts.n, opts.alpha)
        tfdc = gen_tfdc(fmla,cnfdir, log)
        if tfdc is None:
          continue
        else:
          wtr.write_tfdc(tfdc)
          i += 1

  wtr.finalize()


def validate_TFDC(tfdc):
  def decode_l_idx(k):
    n_vars = tf.cast(tfdc.n_vars,dtype="int32")
    if k+1 > n_vars:
      return -(k+1 - n_vars)
    else:
      return k + 1
  
  fmla = CNF()
  for _ in range(tfdc.n_clauses):
    fmla.append([])
  for edge in tfdc.CL_idxs:
    fmla.clauses[edge[0]].append(int(decode_l_idx(edge[1])))

  core = CNF()
  for i in range(len(tfdc.core_clause_mask)):
    if tfdc.core_clause_mask[i] == 1:
      core.append(fmla.clauses[i])

  with Solver(name="cdl") as S:
    S.append_formula(core)
    assert S.solve() is False
    
  with Solver(name="cdl") as S:
    S.append_formula(fmla)
    assert S.solve() is False

  # all variables in the core_var_mask are in the core
  for i in range(len(tfdc.core_var_mask)):
    if tfdc.core_var_mask[i] == 1:
      var = i+1
      flag = False
      for cls in core.clauses:
        for lit in cls:
          if abs(lit) == var:
            flag = True
      assert flag

  # all variables in the core are in the core_var_mask
  for cls in core.clauses:
    for lit in cls:
      l_idx = abs(lit) - 1
      assert tfdc.core_var_mask[l_idx] == 1

  print("ok")

def CNF_of_TFDCR(tfdcr):
  def decode_l_idx(k):
      n_vars = tf.cast(tfdcr.n_vars,dtype="int32")
      if k+1 > n_vars:
        return -(k+1 - n_vars)
      else:
        return k + 1  
  clauses = []
  for _ in range(tfdcr.n_clauses):
    clauses.append([])
  for edge in tfdcr.CL_idxs:
    clauses[edge[0]].append(int(decode_l_idx(edge[1])))
  fmla = CNF(from_clauses=clauses)
  return fmla  

def CNF_of_TFDC(tfdc):
  def decode_l_idx(k):
      n_vars = tf.cast(tfdc.n_vars,dtype="int32")
      if k+1 > n_vars:
        return -(k+1 - n_vars)
      else:
        return k + 1  
  clauses = []
  for _ in range(tfdc.n_clauses):
    clauses.append([])
  for edge in tfdc.CL_idxs:
    clauses[edge[0]].append(int(decode_l_idx(edge[1])))
  fmla = CNF(from_clauses=clauses)
  return fmla

def validate_deserialized_TFDC(tfdc):
  def decode_l_idx(k):
    n_vars = tf.cast(tfdc.n_vars,dtype="int32")
    if k+1 > n_vars:
      return -(k+1 - n_vars)
    else:
      return k + 1
  
  fmla = CNF_of_TFDC(tfdc)

  core = CNF()
  for i in range(len(tfdc.core_clause_mask)):
    if (tfdc.core_clause_mask[i]) == True:
      core.append(fmla.clauses[i])

  core.to_fp(sys.stdout)
  print("")
  print("%%%%%%%% FORMULA %%%%%%%%")
  print("")
  fmla.to_fp(sys.stdout)
      
  with Solver(name="cdl") as S:
    S.append_formula(core)
    assert S.solve() is False
    
  with Solver(name="cdl") as S:
    S.append_formula(fmla)
    assert S.solve() is False

  # all variables in the core_var_mask are in the core
  for i in range(len(tfdc.core_var_mask)):
    if tfdc.core_var_mask[i] == True:
      var = i+1
      flag = False
      for cls in core.clauses:
        for lit in cls:
          if abs(lit) == var:
            flag = True
      assert flag

  # all variables in the core are in the core_var_mask
  for cls in core.clauses:
    for lit in cls:
      l_idx = abs(lit) - 1
      assert tfdc.core_var_mask[l_idx] == True

  print("ok")

def validate_get_unsat_core_pysat(fmla, result, bad_asms):
  core = CNF(from_clauses=[fmla.clauses[i] for i in result])
  for asm in bad_asms:
    core.append([asm])
  with Solver(name="cdl") as S:
    S.append_formula(core)
    assert S.solve() is False
    print("ok")

def get_unsat_core_pysat(fmla, alpha=None):
    n_vars = fmla.nv
    vpool = IDPool(start_from=n_vars+1)
    r = lambda i: vpool.id(i)
    new_fmla = fmla.copy()
    num_clauses = len(new_fmla.clauses)
    for count in list(range(0, num_clauses)):
        new_fmla.clauses[count].append(r(count)) # add r_i to the ith clause
    s = Solver(name="cdl")
    s.append_formula(new_fmla)
    asms = [-r(i) for i in list(range(0, num_clauses))]
    if alpha is not None:
      asms = asms + alpha
    if not s.solve(assumptions=asms):
        core_aux = s.get_core()
    else: # TODO(jesse): better error handling
        raise Exception ("formula is sat")
    # return list(filter(lambda x: x is not None, [vpool.obj(abs(r)) for r in core_aux]))
    result = []
    bad_asms = []
    for lit in core_aux:
      if abs(lit) > n_vars:
        result.append(vpool.obj(abs(lit)))
      else:
        bad_asms.append(lit)
    return result, bad_asms

def test_get_unsat_core():
  with open("cnf/unsat_fmla.cnf", "r") as f:
    fmla = CNF(from_fp=f)
  result, bad_asms = get_unsat_core_pysat(fmla, None)
  validate_get_unsat_core_pysat(fmla, result, bad_asms)

def run_tests():
  test_parsing()
  print("all tests passed")

from copy import deepcopy
# given a formula, return a new formula simplified by the assumptions
def simplify_CNF(fmla, assumptions):
  result = list(filter(lambda cls:all(map(lambda lit: lit not in cls, assumptions)),
                       deepcopy(fmla.clauses)))
  for cls in result:
    for lit in assumptions:
      try:
        cls.remove(-lit)
      except ValueError:
        continue
  return CNF(from_clauses=result)

def test_simplify_CNF():
  clauses = [[1,2],[4,5],[-1,2,3]]
  fmla = CNF(from_clauses=clauses)
  simplify_CNF(fmla, [1,4]).to_fp(sys.stdout)
  fmla.to_fp(sys.stdout)


def res_graph_idxs(fmla):
  """
  Args:
  fmla: CNF object

  Returns:
  A list of edges for the resolution graph
  # TODO(jesse): add weights 1/1+(size of resolvent) and exclude edges for tautologies
  """
  result = set()
  for cls1_idx in range(1,len(fmla.clauses)):
    for cls2_idx in range(0, cls1_idx):
      for lit in fmla.clauses[cls1_idx]:
        if (-lit) in fmla.clauses[cls2_idx]:
          result.add((cls1_idx, cls2_idx))
          result.add((cls2_idx, cls1_idx))
  return list(result)

def test_res_graph_idxs():
  fmla = CNF(from_clauses=[[1,2,3], [-1,2,3], [1,-2,3], [-3]])
  idxs = res_graph_idxs(fmla)
  edges = [[0,1],[1,0],[0,2], [2,0], [1,2],[2,1], [0,3], [3,0],[1,3], [3,1], [2,3],[3,2]]
  for edge in edges:
    assert (edge[0], edge[1]) in idxs
  print(idxs)
  assert len(edges) == len(idxs)
  print("ok")


def gen_tfdcr(fmla, cnfdir,log=None):
  fmla = fmla
  n_vars = fmla.nv
  n_clauses = len(fmla.clauses)
  name = str(uuid.uuid4())
  cdl = cadical(cnf_dir=cnfdir)
  drat = drat_trim(cnf_dir=cnfdir)
  with open(cnfdir + name + ".cnf", "w") as f:
    fmla.to_fp(f)
  cdl.set_rootname(name)
  if log is not None:
    log.write(cdl.cnf_path + "\n")
  cdl.run()
  if log is not None:
    log.write(str(cdl.output) + "\n")
  cdl.process_output()
  if cdl.result:
    return None

  cdl.write_proof()

  drat.set_rootname(name)
  drat.run()
  if log is not None:
    log.write (drat.drat_trim_path + "\n")
    log.write (drat.opt_path + "\n")

  tsr = parse_drat(drat.opt_path, n_vars)

  var_lemma_counts = lemma_occ(tsr)
  var_del_counts = del_occ(tsr)

  masks = parse_core(drat.core_path, n_vars, n_clauses)

  rgraph_idxs = np.array(res_graph_idxs(fmla), dtype="int32")

  CL_idxs = clgraph(fmla)

  tfdcr = TFDCR(
    n_vars = n_vars,
    n_clauses = n_clauses,
    CL_idxs = CL_idxs,
    core_var_mask = masks["core_var_mask"],
    core_clause_mask = masks["core_clause_mask"],
    var_lemma_counts = var_lemma_counts,
    var_del_counts = var_del_counts,
    res_idxs=rgraph_idxs
  )
  return tfdcr

def G_cl_of_idxs(n_clauses, n_vars, CL_idxs):
  G = tf.SparseTensor(indices=tf.cast(CL_idxs, tf.int64),
                                         values=tf.ones(tf.shape(CL_idxs)[0]),
                                         dense_shape=[tf.cast(n_clauses, tf.int64), tf.cast(2*n_vars, tf.int64)])
  return G

def gen_fmla_fragment(fmla, c):
  if c == 0:
    return fmla
  else:
    cube_vars = np.random.choice(fmla.nv, size=c, replace=False)
    assumptions = [v if np.random.uniform() > 0.5 else -v for v in cube_vars]
    return simplify_CNF(fmla, assumptions)

# use multithreaded data generation --- see gen_data.py
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data-dir", action="store", dest="data_dir", type=str, default = os.path.join(PROJECT_DIR, "data/"))
#     parser.add_argument("--batch-size", action="store", dest="n_tfrs_per_file", type=int, default = 5000)
#     parser.add_argument("--datapoints", action="store", dest="n_datapoints", type=int, default = 50000)
#     parser.add_argument("--test", action="store", dest="is_test", type=bool, default=False)
#     parser.add_argument("--vars-per-clause", action="store", dest="k", type=int, default=3)
#     parser.add_argument("--num-vars", action="store", dest="n", type=int, default=40)
#     parser.add_argument("--alpha", action="store", dest="alpha", type=float, default=4.4)
#     opts = parser.parse_args()

#     print("Parsed options:")

#     for key in vars(opts):
#       print(f"  {key} := {vars(opts)[key]}")

#     print("Continue?")

#     input()

#     if not opts.is_test:
#       gen_randkcnf_data(opts)
#     else:
#       run_tests()
