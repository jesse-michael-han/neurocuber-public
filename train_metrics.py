from cnf_util import *
from tfutil import *
from pysat.solvers import Solver
import tensorflow as tf
import numpy as np
from scipy.stats import rankdata

# by convention, these metrics only operate on individual samples
# extend to batches by vectorized_map or map_fn and reduce_mean the result manually

# CORE PREDICTION METRICS

# each metric in this section accepts y_true (bitmask from dataset)
# and y_pred (a softmaxed vector of core prediction probabilities)

# description of metrics:
# 1. infimum of probabilities in the labelled unsat core
# 2. size of infimum-core (i.e. number of times you have to greedily select unsat core predictions to get the labelled core) # actually, this isn't so useful a metric without knowing the size of the label
# 3. ratio of (size of infimum core) to (size of labelled core)
# 4. percentage of false positives (i.e. (1 - labelled core / infimum core))
# 5. (expensive, only use for separate evaluation pass: greedily select unsat core from predictions and see how many iterations it takes to produce an actual unsat core.)
# y_true, y_pred
# 6. mask kldiv (loss function)

# VARIABLE SELECTION METRICS

# description of metrics:

# 1. KLdiv against DRAT count

# 2. KLdiv against core var mask

# 3. infimum of probabilities among variables with any label (see above, use tf.cast(_, dtype=tf.bool) to convert into a bitmask)

# 4. size of infimum-core needed to get all labelled variables (see above)

# 5. ratio of size of infimum-core to labelled core (see above)

# 6. percentage of false positives (i.e. (1 - labelled core/infimum core)) (see above)

# 7 (expensive, and I might not implement): test number of steps needed by a basic DPLL solver to prove unsat with these branching heuristics

# 8. (expensive): test by playout evaluation on the prover-adversary game

# 9 (expensive): test quality of variable selection in uniform cubing with fixed cube budget

# 10 (slightly less expensive, higher variance): test quality of variable selection in random cubing with fixed cube budget

# 11 (maybe for later): revive importance sampling from other neurocuber

# metric 1
def InfimumMetric(y_true, y_pred):
  
  def inf_prob( y_true, y_pred):
    """
    Args:
    y_true: a 1D tensor of type bool
    y_pred: a float32 1D tensor forming a probability distribution
    """
    labelled_core_preds = tf.boolean_mask(y_pred, y_true)
    inf = tf.reduce_min(labelled_core_preds, axis=-1)
    return inf

  # metric 2
  # note(jesse): this seems to return an int32 tensor
  def inf_core( y_pred, inf):
    inf_core_mask = tf.map_fn(lambda x: tf.cond(x >= inf, lambda: True, lambda: False), y_pred)
    return inf_core_mask
    

  # metric 3
  def inf_core_to_label_ratio( y_true, inf_core_mask):
    return tf.reduce_sum(tf.cast(inf_core_mask, tf.float32), axis=-1) / tf.reduce_sum(tf.cast(y_true, tf.float32), axis=-1)

  # metric 4
  def inf_core_false_pos( ratio):
    return (1. - (1./ratio))

    # ensure labels are a bitmask
  y_true = tf.cast(y_true, tf.bool)
  inf = inf_prob(y_true, y_pred)
  inf_core_mask = inf_core(y_pred, inf)
  inf_core_label_ratio = inf_core_to_label_ratio(y_true, inf_core_mask)
  false_pos_ratio = inf_core_false_pos(inf_core_label_ratio)
  return inf, inf_core_label_ratio, false_pos_ratio

  
def rankmin(x):
    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]

# metric 5
def GreedyCoreMetric(fmla, y_true, y_pred):
  """
  Args:
  y_true: a bitmask
  y_pred: probability distribution
  """
  UNSAT_FLAG = False
  sorted_indices = np.argsort(y_pred, kind="mergesort")
  print(sorted_indices)
  with Solver(name="cdl") as S:
    count = 0
    indices = []
    for i in range(len(sorted_indices)):
      next_index = sorted_indices[-(i+1)]
      indices.append(next_index)
      next_clause = fmla.clauses[next_index]
      S.add_clause(next_clause)
      count += 1
      if not S.solve():
        UNSAT_FLAG = True
        break
  assert UNSAT_FLAG
  gen_core_length = len(indices)
  labelled_core_length = int(tf.reduce_sum(tf.cast(y_true, tf.int32), axis=-1))
  return indices, gen_core_length, labelled_core_length, float(gen_core_length/labelled_core_length)

def KLDivMetric(y_true, y_pred, tau=1):
  """
  thin wrapper around tfutil.softmax_kldiv2. we continue to follow the convention that y_true and y_pred are single samples

  y_true and y_pred are both logits
  """
  y_true = tf.stack([tf.cast(y_true, tf.float32)])
  y_pred = tf.stack([tf.cast(y_pred, tf.float32)])
  return tfutil.softmax_kldiv2(tau=tau)(y_true, y_pred)

def MaskKLDivMetric(y_true, y_pred):
  """
  thin wrapper around tfuitl.mask_kldiv. we continue to follow the convention that y_true and y_pred are single samples

  y_true is a bitmask, and y_pred is logits
  """
  y_true = tf.stack([tf.cast(y_true, tf.float32)])
  y_pred = tf.stack([tf.cast(y_pred, tf.float32)])
  return tfutil.mask_kldiv()(y_true, y_pred)
  
