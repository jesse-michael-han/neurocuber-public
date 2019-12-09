import numpy as np
from tensorflow.python.framework import tensor_shape
import tfutil
# from neurosat import *
from train_util import *
from pysat.formula import CNF
from cnf_util import res_graph_idxs, G_cl_of_idxs, clgraph
from config import *
import os
import time
import tensorflow as tf
import numpy as np

default_cfg_path = os.path.join(PROJECT_DIR, "res_models/", "test.json")
default_cfg = ModelCfg_from_file(default_cfg_path)

class DenseWeightNormLayer(tf.keras.layers.Layer):
  """
  A dense layer with optional weight reparametrization.
  """
  def __init__(self, d_out, activation="relu6", weight_reparam=True, dtype="float32"):
    super(DenseWeightNormLayer, self).__init__(dtype=dtype)
    self.weight_reparam = weight_reparam
    self.d_out = d_out
    if activation == "relu": activation = tf.nn.relu
    elif activation == "relu6": activation =  tf.nn.relu6
    elif activation == "tanh": activation = tf.nn.tanh
    elif activation == "sig": activation = tf.nn.sigmoid
    elif activation == "elu": activation =  tf.nn.elu
    elif activation == "softmax": activation = tf.nn.softmax
    elif activation == None: activation = None
    else:
        raise Exception(f"Unsupported transfer function {activation}")
    self.transfer_fn = activation
    
  def build(self, input_shape):
    self.last_dim = tensor_shape.dimension_value(input_shape[-1])
    if self.weight_reparam:
      self.kernel_w = self.add_weight("w", shape = [self.last_dim, self.d_out], initializer = tf.keras.initializers.he_normal(seed=None))
      self.kernel_g = self.add_weight("g", shape = [1, self.d_out], initializer = tf.ones_initializer())
    else:
      self.kernel = self.add_weight("w", shape = [self.last_dim, self.d_out],initializer = tf.initializers.GlorotNormal())
    self.bias = self.add_weight("b", shape = [self.d_out], initializer = tf.zeros_initializer())

  def call(self, input_):
    if self.weight_reparam:
      kernel = tf.nn.l2_normalize(self.kernel_w, axis = 0) * tf.tile(self.kernel_g,[self.last_dim,1])
      if self.transfer_fn is not None:
        return self.transfer_fn(tf.matmul(input_, kernel) + self.bias)
      else:
        return tf.matmul(input_, kernel) + self.bias
    else:
      if self.transfer_fn is not None:
        return self.transfer_fn(tf.matmul(input_, self.kernel) + self.bias)
      else:
        return tf.matmul(input_, self.kernel) + self.bias

class NeuroSAT_MLP(tf.keras.layers.Layer):
  def __init__(self, hidden_layers=3, hidden_dim=80,output_dim=80, activation="relu6", weight_reparam=True, dtype="float32", **kwargs):
    super(NeuroSAT_MLP, self).__init__(dtype=dtype, **kwargs)
    self.hidden_layers=hidden_layers
    self.hidden_dim=hidden_dim
    self.output_dim=output_dim

    self.layer_list = []
    for i in list(range(hidden_layers)):
      self.layer_list.append(DenseWeightNormLayer(hidden_dim))
    self.layer_list.append(DenseWeightNormLayer(output_dim, activation=None))
    
  def call(self, inputs):
    z = inputs
    for l in self.layer_list:
      z = l(z)
    return z


def blockify3(Gs, n_row_list, n_col_list, DEBUG=False):
  """
  shift columns only
  """
  i = tf.constant(0, dtype=tf.int64)
  batch_size = len(Gs)
  n_col_total = tf.reduce_sum(n_col_list)
  
  def lambda_body(n):
    n_cells = Gs[n].indices.shape[0]
    shifts = tf.tile(np.array([[0 # tf.reduce_sum(n_row_list[0:n])
                                , tf.reduce_sum(n_col_list[0:n])]], dtype="int64"), [n_cells,1])    
    Gs[n] = tf.SparseTensor(indices=tf.add(Gs[n].indices, shifts), values=Gs[n].values, dense_shape=(Gs[n].dense_shape[0], n_col_total))
    n = tf.add(n, 1)
    return [n]
    
  tf.while_loop(lambda i: i < batch_size, lambda_body, [i])

  # print(tf.sparse.to_dense(Gs[0], validate_indices=False))

  result = tf.sparse.concat(axis=0, sp_inputs=Gs)

  return tf.sparse.reorder(result)

class NeuroCuber(tf.keras.Model):
  # This implements the following behavior:
  # - default argument of subsequent arguments to cfg is `None`
  # - if an argument is None, fall back to the value of the corresponding key of `cfg`, otherwise override `cfg`
  
  def __init__(self, cfg = default_cfg, dtype="float32", # TODO(jesse): define a default ModelCfg
               **kwargs):

    self.RES_FLAG = False
    self.RESL_FLAG = False
    self.CUBE_FLAG = False

    if "mode" in kwargs:
      self.mode = kwargs["mode"]
    else:
      self.mode = cfg.mode
    
    if self.mode == "res":
      raise Exception("res not supported")
      self.RES_FLAG = True
    elif self.mode == "resl":
      raise Exception("resl not supported")
      self.RESL_FLAG = True
    elif self.mode == "cube":
      self.CUBE_FLAG = True
    else:
      raise Exception("unsupported mode")

    if "name" in kwargs:
      name = kwargs["name"]
    else:
      name = cfg.model_id

    super(NeuroCuber, self).__init__(name=name, dtype=dtype)

    ### begin setting hyperparameters

    if "d" in kwargs:
      self.d = kwargs["d"]
    else:
      self.d = cfg.d

    if "C_res_depth" in kwargs:
      self.C_res_depth = kwargs["C_res_depth"]
    else:
      self.C_res_depth = cfg.C_res_depth

    if "C_update_depth" in kwargs:
      self.C_update_depth = kwargs["C_update_depth"]
    else:
      self.C_update_depth = cfg.C_update_depth

    if "L_update_depth" in kwargs:
      self.L_update_depth = kwargs["L_update_depth"]
    else:
      self.L_update_depth = cfg.L_update_depth

    if "V_proof_depth" in kwargs:
      self.V_proof_depth = kwargs["V_proof_depth"]
    else:
      self.V_proof_depth = cfg.V_proof_depth

    if "V_core_depth" in kwargs:
      self.V_core_depth = kwargs["V_core_depth"]
    else:
      self.V_core_depth = cfg.V_core_depth

    if "C_core_depth" in kwargs:
      self.C_core_depth = kwargs["C_core_depth"]
    else:
      self.C_core_depth = cfg.C_core_depth

    if "n_rounds" in kwargs:
      self.n_rounds = kwargs["n_rounds"]
    else:
      self.n_rounds = cfg.n_rounds

    if "weight_reparam" in kwargs:
      self.weight_reparam = kwargs["weight_reparam"]
    else:
      self.weight_reparam = cfg.weight_reparam

    if "res_layers" in kwargs:
      self.res_layers = kwargs["res_layers"]
    else:
      self.res_layers = cfg.res_layers

    if "norm_axis" in kwargs:
      self.norm_axis = kwargs["norm_axis"]
    else:
      self.norm_axis = cfg.norm_axis

    if "norm_eps" in kwargs:
      self.norm_eps = kwargs["norm_eps"]
    else:
      self.norm_eps = cfg.norm_eps

    if "CL_scale" in kwargs:
      self.CL_scale = kwargs["CL_scale"]
    else:
      self.CL_scale = cfg.CL_scale

    if "LC_scale" in kwargs:
      self.LC_scale = kwargs["LC_scale"]
    else:
      self.LC_scale = cfg.LC_scale

    if "activation" in kwargs:
      self.activation = kwargs["activation"]
    else:
      self.activation = cfg.activation

    self.L_init_scale = 1.0/np.sqrt(self.d)
    self.C_init_scale = 1.0/np.sqrt(self.d)

    ### end hyperparameters

    self.C_update = NeuroSAT_MLP( hidden_layers = self.C_update_depth,
                                  hidden_dim = 2 * self.d,
                                  output_dim = self.d,
                                  activation = self.activation,
                                  weight_reparam = self.weight_reparam,
                                  dtype = "float32",
                                  name = "C_update"
    )

    self.L_update = NeuroSAT_MLP( hidden_layers = self.L_update_depth,
                                  hidden_dim = 3 * self.d,
                                  output_dim = self.d,
                                  activation = self.activation,
                                  weight_reparam = self.weight_reparam,
                                  dtype = "float32",
                                  name = "L_update"
    )

    self.V_proof = NeuroSAT_MLP( hidden_layers = self.V_proof_depth,
                                 hidden_dim = 2 * self.d,
                                 output_dim = 1,
                                 activation = self.activation,
                                 weight_reparam = self.weight_reparam,
                                 dtype = "float32",
                                 name = "V_proof"
    )

    self.V_core = NeuroSAT_MLP( hidden_layers = self.V_core_depth,
                                 hidden_dim = 2 * self.d,
                                 output_dim = 1,
                                 activation = self.activation,
                                 weight_reparam = self.weight_reparam,
                                 dtype = "float32",
                                 name = "V_core"
    )

    self.C_core = NeuroSAT_MLP( hidden_layers = self.C_core_depth,
                                 hidden_dim = self.d,
                                 output_dim = 1,
                                 activation = self.activation,
                                 weight_reparam = self.weight_reparam,
                                 dtype = "float32",
                                 name = "C_core"
    )

  def call(self, G_cls, n_clauses_list, n_vars_list):
    """
    Accepts:
    A list of sparse clause-literal adjacency matrices, a list of n_clause dimensions, and a list of n_var dimensions
    """
    
    n_clauses = tf.reduce_sum(n_clauses_list, axis=-1)
    n_vars = tf.reduce_sum(n_vars_list, axis=-1)
    n_lits = tf.multiply(2, n_vars)
    batch_size = n_clauses_list.shape[0]
    G_cl = blockify3(G_cls, n_clauses_list, 2*n_vars_list)
    L  = tf.ones(shape=[2 * n_vars, self.d], dtype=tf.float32) * self.L_init_scale
    C  = tf.ones(shape=[n_clauses, self.d], dtype=tf.float32) * self.C_init_scale

    def flip(lits):
      Ls = tf.split(tf.cast(lits, tf.float32), tf.multiply(2, n_vars_list))

      i = tf.constant(0, dtype=tf.int64)

      def lambda_body(i):
        Ls[i] = tf.concat([Ls[i][tf.cast(Ls[i].shape[0]/2, dtype=tf.int64):,:], Ls[i][0:tf.cast(Ls[i].shape[0]/2, dtype=tf.int64),:]], axis=0)
        i = tf.add(i,1)
        return [i]

      tf.while_loop(lambda i: i < batch_size, lambda_body, [i])

      result = tf.concat(Ls, axis=0)
      return tf.reshape(result, (n_lits, self.d))


    for t in range(self.n_rounds):
      if self.res_layers:
        C_0, L_0 = C, L
      # C = tf.debugging.check_numerics(C, message="C before G_cl update")
      
      C = self.C_update(tf.concat((C, tf.sparse.sparse_dense_matmul(G_cl, L) * self.LC_scale), axis=1))
      # C = tf.debugging.check_numerics(C, message="C after G_cl update")
      C = tfutil.normalize(C, axis=self.norm_axis, eps=self.norm_eps)
      # C = tf.debugging.check_numerics(C, message="C after norm")

      C = tfutil.normalize(C, axis=self.norm_axis, eps=self.norm_eps)
      # C = tf.debugging.check_numerics(C, message="C after norm")

      if self.res_layers:
        C = C + C_0

      # L = tf.debugging.check_numerics(L, message="L before update")
      L = self.L_update(tf.concat((L, tf.sparse.sparse_dense_matmul(tf.sparse.transpose(G_cl), C, adjoint_a = False) * self.CL_scale, flip(L)), axis=1))
      # L = tf.debugging.check_numerics(L, message="L after update")
      L = tfutil.normalize(L, axis=self.norm_axis, eps=self.norm_eps)
      # L = tf.debugging.check_numerics(L, message="L after norm")
      if self.res_layers: # TODO(jesse): test res_layers
        L = L + L_0

    # flop the literals
    def flop(lits):
      # split the tensors, flip them, then re-stack them
      Ls = tf.split(tf.cast(lits, tf.float32), tf.multiply(2, n_vars_list))
      i = tf.constant(0, dtype=tf.int64)
      def lambda_body(i):
        Ls[i] = tf.concat([Ls[i][0:tf.cast(Ls[i].shape[0]/2, dtype=tf.int64):,:], Ls[i][tf.cast(Ls[i].shape[0]/2, dtype=tf.int64):,:]], axis=1)
        i = tf.add(i,1)
        return [i]

      tf.while_loop(lambda i: i < batch_size, lambda_body, [i])

      
      return tf.reshape((tf.concat(Ls, axis=0)), (n_vars, 2*self.d))
    
    V = flop(L)

    V_proof_logits = tf.squeeze(self.V_proof(V))

    V_core_logits = tf.squeeze(self.V_core(V))

    C_core_logits = tf.squeeze(self.C_core(C))

    # print(V_proof_logits)

    return tf.split(V_proof_logits, n_vars_list), tf.split(V_core_logits, n_vars_list), tf.split(C_core_logits, n_clauses_list)  

test_fmla = CNF(from_clauses=[[1,2,3],[-1,2,3],[1,-2,3],[-3]])

G_cl_test = G_cl_of_idxs(len(test_fmla.clauses), test_fmla.nv, tf.cast(clgraph(test_fmla), tf.int64))

def initialize_neurocuber(neurocuber): # TODO(jesse): fix
  # print(G_cl_test)
  # print(G_cl_test.dense_shape)
  return neurocuber([G_cl_test], np.array([len(test_fmla.clauses)]), np.array([test_fmla.nv]))

def mk_neurocuber_loss(loss_fn1, loss_fn2=tfutil.mask_kldiv(), loss_fn3=tfutil.mask_kldiv(), pv_loss_scale = 1.0# , cv_loss_scale = 1.0, cc_loss_scale = 1.0, l2_loss_scale = 1e-6 # TODO(jesse): implement this
):
  """
  deprecated
  """
  return [loss_fn1, loss_fn2, loss_fn3]

class mk_neurocuber_loss2(tf.keras.losses.Loss):
  """
  TODO(jesse): document this hack
  """
  def __init__(self, loss_fn1, loss_fn2=tfutil.mask_kldiv(), loss_fn3=tfutil.mask_kldiv(), pv_loss_scale = 1.0, cv_loss_scale = 1.0, cc_loss_scale = 1.0, l2_loss_scale = 1e-6):
    self.loss_fn1 = loss_fn1
    self.loss_fn2 = loss_fn2
    self.loss_fn3 = loss_fn3
    self.pv_loss_scale = pv_loss_scale
    self.cv_loss_scale = cv_loss_scale
    self.cc_loss_scale = cc_loss_scale
    self.l2_loss_scale = l2_loss_scale
    super(mk_neurocuber_loss2, self).__init__(name="losszilla")

  def call(self, y_true, y_pred):
    return self.pv_loss_scale * self.loss_fn1(np.array([y_true[0][0]]), np.array([y_pred[0][0]]))+ self.cv_loss_scale * self.loss_fn2(np.array([y_true[0][1]]), np.array([y_pred[0][1]])) + self.cc_loss_scale * self.loss_fn3(np.array([y_true[0][2]]), np.array([y_pred[0][2]]))

def mk_CL_idxs(tfdc):
  return tf.concat((np.array([[tfdc.n_clauses, tfdc.n_vars]]), tfdc.CL_idxs), axis=0)

# test datapoint:
# formula
# p cnf 3 4
# 1 2 3
# -2 3
# 2
# -3

# CL_idxs
# [[4,3], [0,0],[0,1],[0,2],[1,4],[1,2],[2,1],[3,5]]
test_CL_idxs = np.array([[4,3],[0,0],[0,1],[0,2],[1,4],[1,2],[2,1],[3,5]])

# resolution graph indices:


# nonsense DRAT lemma count:
# [0,2,1]

test_proof_count = np.array([[0,2,1]])

# var_mask:
# [0,1,1]

test_var_mask = np.array([[0,1,1]])

# core_mask:
# [0,1,1,1]

test_core_mask = np.array([[0,1,1,1]])

def init_neurocuber(cfg, restore=False, restore_from=None, **kwargs):
    neurocuber = NeuroCuber(cfg=cfg, **kwargs)
    initialize_neurocuber(neurocuber)
    if restore:
      if restore_from is None:
        checkpoint = tf.train.Checkpoint(model=neurocuber)
        latest = tf.train.latest_checkpoint(cfg.ckpt_dir)
        print("restoring from", latest)
        restore_from = latest
      # neurocuber.load_weights(restore_from).expect_partial()
      checkpoint.restore(latest).expect_partial()
    return neurocuber

# def init_core_model(cfg):
#     neurocuber = NeuroCore(cfg=cfg)
#     x = initialize_neurosat(neurocuber)
#     print(x)
#     return neurocuber

# input: a list of sparse adjacency matrices which do not need to be resized
# returns: a sparse matrix with the list as diagonal blocks
# @tf.function
# from each G in Gs, get the list of indices
# shift the indices so that both row and column start from n * max_n_row, n * max_n_col for n in range(len_Gs)
# return a new sparse tensor with shape n * max_n_row, n * max_n_col

# TODO(jesse): is this efficient?

@tf.function(experimental_relax_shapes=True)
def reconstitute_CL_idxs(CL_idxs, n_vars, max_n_clauses, max_n_vars):
  """
  Args:
  CL_idxs: CL_idxs from a TFDCR
  n_vars: number of variables for CL_idxs
  max_n_clauses: maximum number of clauses in a batch
  max_n_vars: maximum number of variables in a batch

  Returns:
  A sparse clause-literal adjacency matrix, appropriately padded to dimensions max_n_clauses x 2*(max_n_var)
  """
  # indices = tf.cast(CL_idxs, dtype="int64")
  # start = time.time()
  # loop variables: CL_idxs
  # operation: CL_idx + shift_indices = 
  # new_indices = tf.while_loop(lambda i: True, b = lambda i: )

  # print((lambda pr: [pr[0], (lambda l_idx: tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars))))(pr[1])])([1,15]))

  new_indices = tf.cast(tf.map_fn(lambda pr:tf.stack([pr[0], tf.cond(pr[1] < n_vars, lambda: pr[1], lambda: tf.add(pr[1], tf.subtract(max_n_vars, n_vars)))]), CL_idxs), dtype=tf.int64)
  
  # new_indices = tf.cast(tf.vectorized_map(lambda pr: tf.stack([pr[0], (lambda l_idx: tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars))))(pr[1])]), CL_idxs), dtype=tf.int64)

  # raise Exception  



  
  # new_indices = tf.cast([[c_idx, tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars)))] for c_idx, l_idx in CL_idxs], dtype="int64")
  # print("reconstitution time: ", time.time() - start)
  return tf.sparse.reorder(tf.SparseTensor(
    indices=new_indices,
    values=tf.ones(tf.shape(new_indices)[0]),
    dense_shape=tf.cast((max_n_clauses, 2*max_n_vars), tf.int64)
  )) # TODO(jesse): test

# @tf.function(experimental_relax_shapes=True)
@tf.function
def transform_CL_idxs(CL_idxs, max_n_clauses, max_n_vars):
  """
  combines reconstitute + shift_pad

  Args:
  CL_idxs: CL_idxs from a TFDCR with a constant value at the end of each row
  n_vars: number of variables for CL_idxs
  max_n_clauses: maximum number of clauses in a batch
  max_n_vars: maximum number of variables in a batch

  Returns:
  Indices for a sparse adjacency matrix, which should be padded to max_n_clauses x 2*max_n_vars
  """
  # indices = tf.cast(CL_idxs, dtype="int64")
  # start = time.time()
  # loop variables: CL_idxs
  # operation: CL_idx + shift_indices = 
  # new_indices = tf.while_loop(lambda i: True, b = lambda i: )

  # print((lambda pr: [pr[0], (lambda l_idx: tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars))))(pr[1])])([1,15]))

  n = CL_idxs[0,2]
  # print(n)

  n_cells = CL_idxs.shape[0]

  n_vars = CL_idxs[0,3]

  new_indices = tf.cast(tf.map_fn(lambda pr:tf.stack([pr[0], tf.cond(pr[1] < n_vars, lambda: pr[1], lambda: tf.add(pr[1], tf.subtract(max_n_vars, n_vars)))]), CL_idxs), dtype=tf.int64)

  # print(tf.stack([tf.multiply(n,max_n_clauses), 2*tf.multiply(n, max_n_vars)]))

  # shifts = tf.cast(tf.tile(tf.cast([[n*max_n_clauses, 2*n* max_n_vars]], dtype=tf.int64), [n_cells,1]),dtype=tf.int64)
  shifts = tf.tile(tf.cast([[n*max_n_clauses, 2*n*max_n_vars]], dtype="int64"), [n_cells, 1])
  # print("HEWWO")

  final_indices = tf.add(new_indices, shifts)

  # print(final_indices)

  return final_indices
  # return new_indices

@tf.function
def mk_batch_G_cl2(new_indices, max_n_clauses, max_n_vars, batch_size):
  final_indices = tf.reshape(new_indices, shape=(new_indices.shape[0]*new_indices.shape[1], 2))
  return tf.sparse.reorder(tf.SparseTensor(
    indices=final_indices,
    values=tf.ones(tf.shape(final_indices)[0]),
    dense_shape=(tf.cast((max_n_clauses * batch_size, 2*max_n_vars*batch_size), tf.int64))
  ))
  
  # new_indices = tf.cast(tf.vectorized_map(lambda pr: tf.stack([pr[0], (lambda l_idx: tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars))))(pr[1])]), CL_idxs), dtype=tf.int64)

  # raise Exception  



  
  # new_indices = tf.cast([[c_idx, tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars)))] for c_idx, l_idx in CL_idxs], dtype="int64")
  # print("reconstitution time: ", time.time() - start)
  # return tf.sparse.reorder(tf.SparseTensor(
  #   indices=new_indices,
  #   values=tf.ones(tf.shape(new_indices)[0]),
  #   dense_shape=tf.cast((max_n_clauses, 2*max_n_vars), tf.int64)
  # )) # TODO(jesse): test
  # return new_indices

# def reconstitute_shift_pad(G_idxs, max_n_row, max_n_col, batch_size,n, DEBUG=False):
#   """
#   shift columns only and pad to max width
#   """
#   if DEBUG:
#     start = time.time()
#   G = tf.sparse.reset_shape(G, [max_n_row, batch_size * max_n_col])
#   if DEBUG:
#     print("sparse reset shape time: ", time.time() - start)
#   n_cells = G.indices.shape[0]
#   if DEBUG:
#     start = time.time()
#   shifts = tf.tile(np.array([[0, n*max_n_col]], dtype="int64"), [n_cells,1])
#   if DEBUG:
#     print("shifts creation time: ", time.time() - start)

#   if DEBUG:
#     start = time.time()
#   G = tf.SparseTensor(indices=tf.add(G.indices, shifts), values=G.values, dense_shape=G.dense_shape)
#   if DEBUG:
#     print("final G creation time: ", time.time() - start)
#   return G

# def reconstitute_CL_idxs(CL_idxs, n_vars, max_n_clauses, max_n_vars):
#   """
#   Args:
#   CL_idxs: CL_idxs from a TFDCR
#   n_vars: number of variables for CL_idxs
#   max_n_clauses: maximum number of clauses in a batch
#   max_n_vars: maximum number of variables in a batch

#   Returns:
#   A sparse clause-literal adjacency matrix, appropriately padded to dimensions max_n_clauses x 2*(max_n_var)
#   """
#   # indices = tf.cast(CL_idxs, dtype="int64")
#   start = time.time()
  
#   new_indices = tf.cast([[c_idx, tf.cond(l_idx < n_vars, lambda: l_idx, lambda: tf.add(l_idx, tf.subtract(max_n_vars, n_vars)))] for c_idx, l_idx in CL_idxs], dtype="int64")
#   print("reconstitution time: ", time.time() - start)
#   return tf.sparse.reorder(tf.SparseTensor(
#     indices=new_indices,
#     values=tf.ones(tf.shape(new_indices)[0]),
#     dense_shape=tf.cast([max_n_clauses, 2*max_n_vars], tf.int64)
#   )) # TODO(jesse): test

@tf.function(experimental_relax_shapes=True)
def reconstitute_res_idxs(res_idxs,max_n_clauses):
  """
  Args:
  res_idxs: res_idxs from a TFDCR
  max_n_clauses: maximum number of clauses in a batch

  Returns:
  A sparse resolution graph adjacency matrix, reshaped to max_n_clauses x max_n_clauses
  """
  # indices = tf.cast(res_idxs, dtype="int64")
  new_indices = tf.cast(res_idxs, dtype="int64")
  return tf.sparse.reorder(tf.SparseTensor(
    indices=new_indices,
    values=tf.ones(tf.shape(new_indices)[0]),
    dense_shape=tf.cast([max_n_clauses, max_n_clauses], tf.int64)
  )) # TODO(jesse): test

def blockify(Gs,max_n_row, max_n_col, batch_size):
  result = tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=[], dense_shape=np.array([batch_size*max_n_row, batch_size*max_n_col]))
  for G, n in zip(Gs, range(batch_size)): # maybe pass batch dim instead of calculating len?
    G = tf.sparse.reset_shape(G, [batch_size * max_n_row, batch_size * max_n_col]) # uniformize shape
    n_cells = G.indices.shape[0]
    shifts = tf.tile(np.array([[n*max_n_row, n*max_n_col]], dtype="int64"), [n_cells,1])
    G = tf.SparseTensor(indices=tf.add(G.indices, shifts), values=G.values, dense_shape=G.dense_shape)
    result = tf.sparse.add(result, G)
  return tf.sparse.reorder(result)

  
  # if DEBUG:
  #   start = time.time()
  # # G = tf.sparse.reset_shape(G, [max_n_row, batch_size * max_n_col])
  # if DEBUG:
  #   print("sparse reset shape time: ", time.time() - start)
  # n_cells = G.indices.shape[0]
  # if DEBUG:
  #   start = time.time()
    
  # shifts = tf.tile(np.array([[0, n*max_n_col]], dtype="int64"), [n_cells,1])
  # if DEBUG:
  #   print("shifts creation time: ", time.time() - start)

  # if DEBUG:
  #   start = time.time()
  # G = tf.SparseTensor(indices=tf.add(G.indices, shifts), values=G.values, dense_shape=G.dense_shape)
  # if DEBUG:
  #   print("final G creation time: ", time.time() - start)
  # return G

# def shift_pads(Gs, max_n_row, max_n_col, batch_size, DEBUG=False):
#   result = []
#   for sample_count in range(batch_size):
#     result.append(shift_pad(Gs[sample_count], max_n_row, max_n_col, batch_size, sample_count, DEBUG))
#   return result
  # return [shift_pad(x[0], max_n_row, max_n_col, batch_size, x[1], DEBUG) for x in zip(Gs, range(batch_size))]

# accepts a single sparse tensor whose slices are the Gs

def blockify2(Gs,max_n_row, max_n_col, batch_size, DEBUG=False):
  if DEBUG:
    start = time.time()
    # Gs = shift_pads(Gs, max_n_row, max_n_col, batch_size, DEBUG)
  i = tf.constant(0, dtype=tf.int64)
  
  def lambda_body(i):
    # i = tf.add(i,1)
    Gs[i] = shift_pad(Gs[i], max_n_row, max_n_col, batch_size, i, DEBUG)
    i = tf.add(i, 1)
    return [i]
    
  tf.while_loop(lambda i: i < batch_size, lambda_body, [i])
  # Gs = [shift_pad(x[0], max_n_row, max_n_col, batch_size, x[1], DEBUG) for x in zip(Gs, range(batch_size))]
  if DEBUG:
    print("total Gs shift_pad time: ", time.time() - start)
  if DEBUG:
    start = time.time()
  result = tf.sparse.concat(axis=0, sp_inputs=Gs)
  if DEBUG:
    print("total concat time", time.time() - start)
  if DEBUG:
    start = time.time()
  final_result = tf.sparse.reorder(result)
  if DEBUG:
    print("total sparse_reorder time", time.time() - start)
  return final_result

def pad_CL_idxs(CL_idxss, max_n_cells):
  result = tf.stack(list(map(lambda CL_idxs: tf.concat((CL_idxs,tf.tile([CL_idxs[-1]], [max_n_cells - CL_idxs.shape[0], 1])),axis=0), CL_idxss)))
  return result

@tf.function
def pad_CL_idxs_index_vars(CL_idxss, n_varss,max_n_cells, batch_size):

  n_varss = tf.reshape(n_varss, shape=(batch_size, 1, 1))
  n_varss = tf.broadcast_to(n_varss, shape=(batch_size, max_n_cells, 1))
  x = tf.reshape(tf.range(batch_size), shape=(batch_size, 1, 1) )
  x = tf.cast(tf.broadcast_to(x, shape=(batch_size, max_n_cells, 1)), dtype=tf.int64)
  return tf.cast(tf.concat((CL_idxss, x, n_varss), axis=-1), dtype=tf.int64)

@tf.function
def map_transform_CL_idxs(CL_idxs, max_n_clauses, max_n_vars):
  return tf.map_fn(lambda x: transform_CL_idxs(x,max_n_clauses, max_n_vars), CL_idxs)

if __name__ == "__main__": # for testing
  G_cl = G_cl_test
  n_vars = test_fmla.nv
  n_clauses = len(test_fmla.clauses)

  n_vars_list = tf.cast(np.array([n_vars, n_vars, n_vars]), dtype=tf.int64)

  n_clauses_list = tf.cast(np.array([n_clauses, n_clauses, n_clauses]), dtype=tf.int64)  

  G_cls = [G_cl, G_cl, G_cl]

  neurocuber = NeuroCuber(d=20, mode="cube")

  logits = neurocuber(G_cls, n_clauses_list, n_vars_list)
  print(logits)
