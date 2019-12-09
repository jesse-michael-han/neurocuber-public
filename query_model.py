from train_util import *
from neurocuber import *
from cnf_util import clgraph

# TODO(jesse): wrap in gRPC server
class NeuroResQuery:
  def __init__(self, cfg_path, restore=True, restore_from=None):
    self.cfg = ModelCfg_from_file(cfg_path)
    self.model = init_neurocuber(self.cfg, restore, restore_from)

  def get_logits(self, fmla):
    n_clauses = len(fmla.clauses)
    n_vars = fmla.nv
    CL_idxs = tf.cast(clgraph(fmla), tf.int64)
    G_cls = [G_cl_of_idxs(n_clauses, n_vars, CL_idxs)]
    
    DRAT_logits, V_core_logits, C_core_logits = self.model(G_cls, tf.cast([n_clauses], tf.int64), tf.cast([n_vars], tf.int64))
    return DRAT_logits[0], V_core_logits[0], C_core_logits[0]

  def get_core_clause_ps(self, fmla):
    logits = self.get_logits(fmla)[2]
    ps = tf.nn.softmax(logits)
    return ps

  def get_core_var_ps(self, fmla):
    logits = self.get_logits(fmla)[1]
    ps = tf.nn.softmax(logits, tau=1)
    return ps

  def get_drat_var_ps(self, fmla):
    logits = self.get_logits(fmla)[0]
    ps = tf.nn.softmax(logits, tau=1)
    return ps

  def __call__(self, fmla):
    return self.get_logits(fmla)

class NeuroCuberQuery:
  def __init__(self, cfg_path, restore=True, restore_from=None):
    self.cfg = ModelCfg_from_file(cfg_path)
    self.model = init_neurocuber(self.cfg, restore, restore_from)

  def get_logits(self, fmla):
    n_clauses = len(fmla.clauses)
    n_vars = fmla.nv
    CL_idxs = tf.cast(clgraph(fmla), tf.int64)
    G_cls = [G_cl_of_idxs(n_clauses, n_vars, CL_idxs)]
    
    DRAT_logits, V_core_logits, C_core_logits = self.model(G_cls, tf.cast([n_clauses], tf.int64), tf.cast([n_vars], tf.int64))
    return DRAT_logits[0], V_core_logits[0], C_core_logits[0]

  def get_core_clause_ps(self, fmla):
    logits = self.get_logits(fmla)[2]
    ps = tf.nn.softmax(logits)
    return ps

  def get_core_var_ps(self, fmla):
    logits = self.get_logits(fmla)[1]
    ps = tf.nn.softmax(logits, tau=1)
    return ps

  def get_drat_var_ps(self, fmla):
    logits = self.get_logits(fmla)[0]
    ps = tf.nn.softmax(logits, tau=1)
    return ps

  def __call__(self, fmla):
    return self.get_logits(fmla)  
