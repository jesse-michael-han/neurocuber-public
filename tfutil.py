import tensorflow as tf
import numpy as np

def normalize(x, axis, eps):
    mean, variance = tf.nn.moments(x, axes=[axis], keepdims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=eps)

# stable softmax
# def softmax(tsr, tau=1, axis=None):
#   if axis is None:
#     axis =- -1
#   # tsr = tf.cast(tsr, dtype="float32")
#   return tf.exp(tau * (tsr - tf.reduce_max(tsr))) / tf.reduce_sum(tf.exp(tau * (tsr - tf.reduce_max(tsr))), axis)

def softmax(tsr, tau=1, axis=None):
    return tf.nn.softmax(tau*tsr, axis=axis)
    

def cross_entropy (p,q, axis=None):
  if axis is None:
    axis = -1
  p = tf.cast(p, dtype="float32")
  q = tf.cast(q, dtype="float32")
  return -(tf.reduce_sum(p * tf.math.log(q)))

class softmax_kldiv(tf.keras.losses.Loss):
  def __init__(self, tau, swap=False):
    self.swap = swap
    self.tau = tau
    super(softmax_kldiv, self).__init__(
      name="softmax_kldiv"
    )
  def call(self, y_true, y_pred):
    if self.swap:
      a = y_pred
      y_pred = y_true
      y_true = a
    return tf.keras.losses.kullback_leibler_divergence(softmax(y_true, self.tau) + 1e-8,
                                                       softmax(y_pred, self.tau))

def stable_kldiv(labels, y_pred, tau):
    """
    `labels` must be a valid probability distribution, while `y_pred` should be logits
    """
    return tf.nn.softmax_cross_entropy_with_logits(logits=tau*y_pred, labels=labels) + tf.reduce_sum(labels * tf.math.log(labels + 1e-8), axis=1) # reduce_op with unspecified axis is the root of all evil

class softmax_kldiv2(tf.keras.losses.Loss):
  def __init__(self, tau, swap=False):
    self.tau = tau
    self.swap = swap
    super(softmax_kldiv2, self).__init__(
      name="softmax_kldiv2")
    
  def call(self, y_true, y_pred):
    if self.swap:
        a = y_pred
        y_pred = y_true
        y_true = a
    labels = tf.nn.softmax(self.tau * y_true)
    return stable_kldiv(labels=labels,y_pred=y_pred, tau=self.tau)


class softmax_kldiv3(tf.keras.losses.Loss):
  def __init__(self, tau, swap=False):
    self.tau = tau
    self.swap = swap
    super(softmax_kldiv3, self).__init__(
      name="softmax_kldiv3")
    
  def call(self, y_true, y_pred):
    if self.swap:
        a = y_pred
        y_pred = y_true
        y_true = a
    labels = softmax(y_true, self.tau)
    return cross_entropy(softmax(y_pred), labels) + tf.reduce_sum(labels * tf.math.log(labels + 1e-8))

class cross_entropy_loss(tf.keras.losses.Loss):
  def __init__(self, tau, swap=False):
    self.tau = tau
    self.swap = swap
    super(cross_entropy_loss, self).__init__(
      name="cross_entropy_loss")
    
  def call(self, y_true, y_pred):
    if self.swap:
        a = y_pred
        y_pred = y_true
        y_true = a
    labels = softmax(y_true, self.tau) + 1e-8
    return cross_entropy(softmax(y_pred, self.tau) + 1e-8, labels)

class mask_kldiv(tf.keras.losses.Loss):
    """
    Assumes that y_true is a bitmask, and converts this into a uniform distribution with the same support as the mask before taking KL-divergence.
    """
    # TODO(jesse): test this
    def __init__(self):
        super(mask_kldiv, self).__init__(name="mask_kldiv")

    def call(self, y_true, y_pred):
        mask = y_true
        logits = y_pred
        denom = tf.broadcast_to(tf.transpose([tf.reduce_sum(mask, axis=1)]), mask.shape)
        labels = mask/(denom + 1e-8)
        # print(labels)
        # print(logits)
        return stable_kldiv(labels,logits,tau=1)

# TODO(jesse): implement Renyi entropy loss function and play with parameter α

def sparsify(arr):
    idx  = np.where(arr != 0.0)
    return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)

if __name__=="__main__":
  k = softmax_kldiv(tau=1.5)
  loss = k(np.array([[10],[3],[0],[0],[0]]),np.array([[0],[0],[0],[0],[5]]))
  print(loss.numpy())
