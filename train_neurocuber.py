from cnf_util import *
from gen_fmlas import *
from tftd import example_to_tfdc, tfdc_to_example
from train_util import *
import tempfile
from neurocuber import *
import re
import time
import json
import datetime
from types import SimpleNamespace
from train_metrics import *
from util import *

CL_PREP_TIMES = []
CL2_PREP_TIMES = []

def prepare_dataset(path_to_folder, batch_size, **kwargs):
  """
  Args:
  path_to_folder: path to folder containing TFRs containing TFDCs

  Returns:
  A generator which returns batches from the aggregated TFRecordDataset.

  Each batch comprises a list of clause-literal matrices, a list of n_clauses, and a list a n_vars.
  """
  dataset_files = [os.path.join(path_to_folder, x) for x in os.listdir(path_to_folder) if re.search(".tfr", x)]
  dataset = tf.data.TFRecordDataset(filenames=dataset_files, compression_type="GZIP")
  dataset = dataset.repeat()
  dataset = dataset.map(example_to_tfdc)
  dataset = dataset.prefetch(2*batch_size)

  def get_next_sample_gen():
    for tfdc in dataset:
      yield tfdc

  dataset_gen = get_next_sample_gen()

  def get_next_batch():

    def return_if_larger(x,y):
      if y > x:
        return y
      else:
        return x

    while True:
      max_n_clauses = 0
      max_n_vars = 0

      CL_idxss = []
      n_varss = []
      n_clausess = []
      DRAT_target = []
      core_var_target = []
      core_clause_target = []
      for sample_count in range(batch_size):
        sample = next(dataset_gen)
        max_n_clauses = return_if_larger(max_n_clauses, sample.n_clauses)
        max_n_vars = return_if_larger(max_n_vars, sample.n_vars)
        n_varss.append(sample.n_vars)
        n_clausess.append(sample.n_clauses)
        CL_idxss.append(tf.cast(sample.CL_idxs, tf.int64))
        DRAT_target.append(sample.var_lemma_counts)
        core_var_target.append(sample.core_var_mask)
        core_clause_target.append(sample.core_clause_mask)

      G_cls = list(map(lambda x: G_cl_of_idxs(x[2], x[1], x[0]), zip(CL_idxss, n_varss, n_clausess)))

      DRAT_target = tf.cast(tf.stack(list(map(lambda x : tf.pad(x[0], [[0, max_n_vars - x[0].shape[0]]], "CONSTANT"), DRAT_target))), tf.float32)

      core_var_target = tf.stack(list(map(lambda x : tf.pad(x, [[0, max_n_vars - x.shape[0]]], "CONSTANT", constant_values=False), core_var_target)))

      core_clause_target = tf.stack(list(map(lambda x : tf.pad(x, [[0, max_n_clauses - x.shape[0]]], "CONSTANT", constant_values=False), core_clause_target)))

      yield ((G_cls, tf.cast(n_clausess, tf.int64), tf.cast(n_varss, tf.int64), max_n_clauses, max_n_vars), (DRAT_target, core_var_target, core_clause_target))

  return get_next_batch

# @tf.function(experimental_relax_shapes=True)
def neurocuber_pad_logits(logits, max_n_clauses, max_n_vars):
  logits0 = tf.stack([tf.pad(x,[[0,max_n_vars - x.shape[0]]]) for x in logits[0]])
  logits1 = tf.stack([tf.pad(x,[[0,max_n_vars - x.shape[0]]]) for x in logits[1]])
  logits2 = tf.stack([tf.pad(x,[[0,max_n_clauses - x.shape[0]]]) for x in logits[2]])
  return [logits0, logits1, logits2] # TODO(jesse): don't rewrap in list, just pass along to loss

# @tf.function
def neurocuber_loss(logits, targets):
  loss1 = tfutil.softmax_kldiv2(tau=3)(targets[0],logits[0])
  loss2 = tfutil.mask_kldiv()(tf.cast(targets[1], tf.float32), logits[1])
  loss3 = tfutil.mask_kldiv()(tf.cast(targets[2], tf.float32), logits[2])
  return loss1, loss2, loss3

def neurocuber_loss2(logits, targets):
  return while_loop_softmax_kldiv(logits[0], targets[0]), while_loop_mask_kldiv(logits[1], targets[1]), while_loop_mask_kldiv(logits[2], targets[2])


def while_loop_softmax_kldiv(logits, targets):
  batch_size = len(logits)
  i = tf.constant(0, dtype=tf.int64)

  losses = [0 for _ in range(batch_size)]

  def lambda_body(i):
    losses[i] = tfutil.softmax_kldiv2(tau=3)(tf.stack([targets[i]]),tf.stack([logits[i]]))
    i = tf.add(i,1)
    return [i]

  tf.while_loop(lambda i: i < batch_size, lambda_body, [i])

  return tf.reduce_mean(tf.cast(losses, tf.float32))

def while_loop_mask_kldiv(logits, targets):
  batch_size = len(logits)
  i = tf.constant(0, dtype=tf.int64)

  losses = [0 for _ in range(batch_size)]

  # print(logits)
  def lambda_body(i):
    losses[i] = tfutil.mask_kldiv()(tf.stack([tf.cast(targets[i], tf.float32)]), tf.stack([logits[i]]))
    i = tf.add(i,1)
    return [i]

  tf.while_loop(lambda i: i < batch_size, lambda_body, [i])

  return tf.reduce_mean(tf.cast(losses, tf.float32))


def lr_scheduler(learning_rate, DECAY_LR, sample_count, schedule):
  if not DECAY_LR:
    return learning_rate
  else:
    count = 0
    for x in schedule:
      if sample_count < x:
        break
      count += 1
    return learning_rate * np.power(10.0, -count)
        
    

def train_loop(dataset, model, num_samples, batch_size, log_dir, ckpt_dir, ckpt_prefix, ckpt_freq, learning_rate=1e-3, DECAY_LR=True, DEBUG=False):
  """
  Args:
  dataset: a generator produced by prepare_dataset
  model: an instance of neurocuber
  num_samples: total number of samples to train on. will loop through the dataset.
  batch_size: samples per batch
  log_dir: tensorboard summary destination
  ckpt_dir: weight destination
  ckpt_freq: frequency in samples for saving checkpoints

  Returns:
  nothing
  """
  if ckpt_prefix is None:
    checkpoint_prefix = os.path.join(ckpt_dir, "ckpt")
  else:
    checkpoint_prefix = ckpt_prefix

  writer = tf.summary.create_file_writer(log_dir)

  sample_count = 0
  step_count = 0
  checkpoint_sample_count = 0

  lr_tensor = tf.Variable(learning_rate, name="lr")
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_tensor)

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  
  while sample_count < num_samples:
    lr_tensor.assign(lr_scheduler(learning_rate, DECAY_LR, sample_count, [25000,150000,225000])) # TODO(jesse): don't hardcode the schedule, make it configurable
    x, (DRAT_target, core_var_target, core_clause_target) = next(dataset)
    with tf.GradientTape() as tape:
      if DEBUG:
        start = time.time()
      logits = model(x[0], x[1], x[2])
      logits = neurocuber_pad_logits(logits, x[3], x[4])
      if DEBUG:
        print("model inference time: ", time.time() - start)
      loss1, loss2, loss3 = neurocuber_loss(logits, (DRAT_target, core_var_target, core_clause_target))
      loss_value = loss1 + loss2 + loss3 # TODO(jesse): make weighted sum of losses configurable
      
    print(f"current loss: {loss_value.numpy()}")

    grads = tape.gradient(loss_value, model.trainable_variables)
    if DEBUG:
      start = time.time()
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if DEBUG:
      print("apply_gradient time:", time.time() - start)

    step_count += 1
    sample_count += batch_size
    checkpoint_sample_count += batch_size
    print("checkpoint sample count: ", checkpoint_sample_count)
    print(f"sample count: {sample_count}")

    with writer.as_default():
      # # for now, compute these metrics on a random sample from each batch
      metric_idx = np.random.choice(batch_size)
      y_pred0 = logits[0][metric_idx]
      y_pred1 = logits[1][metric_idx]
      y_pred2 = logits[2][metric_idx]

      y_true0 = DRAT_target[metric_idx]
      y_true1 = core_var_target[metric_idx]
      y_true2 = core_clause_target[metric_idx]

      # compute and store core_clause prediction metrics
      inf, inf_core_label_ratio, false_pos_ratio = InfimumMetric(y_true2, tf.nn.softmax(y_pred2))

      tf.summary.scalar("inf core probability", inf, step=(step_count+1))
      tf.summary.scalar("inf-core to label-core ratio", inf_core_label_ratio, step=(step_count+1))
      tf.summary.scalar("core false pos ratio", false_pos_ratio, step=(step_count+1))

      # compute and store DRAT count head metrics
      drat_inf, drat_inf_core_label_ratio, drat_false_pos_ratio = InfimumMetric(y_true0, tf.nn.softmax(y_pred0))
      tf.summary.scalar("inf drat mask probability", drat_inf, step=(step_count+1))
      tf.summary.scalar("inf drat to drat mask ratio", drat_inf_core_label_ratio, step=(step_count+1))
      tf.summary.scalar("drat mask false pos ratio", drat_false_pos_ratio, step=(step_count+1))

      # compute and store core_var prediction metrics
      core_var_inf, core_var_label_ratio, core_var_false_pos_ratio = InfimumMetric(y_true1, tf.nn.softmax(y_pred1))
      tf.summary.scalar("core var inf probability", core_var_inf, step=(step_count+1))
      tf.summary.scalar("core var to label ratio", core_var_label_ratio, step=(step_count+1))
      tf.summary.scalar("core var false pos ratio", core_var_false_pos_ratio, step=(step_count+1))

      # store individual and total training losses
      tf.summary.scalar("total training loss", loss_value, step=(step_count+1))
      tf.summary.scalar("DRAT_var training loss", loss1, step=(step_count+1))
      tf.summary.scalar("core_var training loss", loss2, step=(step_count+1))
      tf.summary.scalar("core_clause training loss", loss3, step=(step_count+1))
      tf.summary.scalar("learning rate", lr_tensor, step=(step_count+1))

    if checkpoint_sample_count >= ckpt_freq: # save checkpoint
      print(f"Saving checkpoint #{checkpoint.save_counter.numpy()}")
      checkpoint.save(file_prefix=checkpoint_prefix)
      checkpoint_sample_count = checkpoint_sample_count - ckpt_freq

class NeuroCuberTrainer:
  def __init__(self, path_to_traincfg):
    with open(path_to_traincfg, "r") as f:
      train_cfg_dict = json.load(f)
    model_cfg = ModelCfg_from_file(train_cfg_dict.pop("model_cfg_path"))
    train_cfg_dict["model_cfg"] = model_cfg
    self.traincfg = SimpleNamespace(**train_cfg_dict)
    self.process_traincfg()

  def process_traincfg(self):
    self.cfg = self.traincfg.model_cfg
    self.ckpt_dir = self.cfg.ckpt_dir
    self.log_dir = os.path.join(self.ckpt_dir, "log/")
    self.model = NeuroCuber(self.cfg)
    self.batch_size = self.traincfg.batch_size
    self.dataset = prepare_dataset(self.traincfg.data_dir, self.batch_size)()
    self.ckpt_prefix = os.path.join(self.ckpt_dir, self.cfg.model_id)
    self.num_samples = self.traincfg.num_samples
    self.ckpt_freq = self.traincfg.ckpt_freq
    self.learning_rate = self.traincfg.learning_rate

  def train(self):
    train_loop(self.dataset,
               self.model,
               self.num_samples,
               self.batch_size,
               self.log_dir,
               self.ckpt_dir,
               self.ckpt_prefix,
               self.ckpt_freq,
               self.learning_rate)
    print("done")
    
def debug_batching():
  path_to_folder = "/home/pv/org/projects/neurocuber-public/train_data_test/sr/train/"

  batch_size = 32

  dataset = prepare_dataset(path_to_folder, batch_size)()

  neurocuber = NeuroCuber(d=80, name="bob")
  N_SAMPLES = 200
  START = time.time()
  train_loop(dataset, neurocuber, N_SAMPLES, batch_size, "scratch/", "scratch/", None, 1000, learning_rate=1e-3, DECAY_LR=True, DEBUG=False)
  ELAPSED = time.time() - START

  print("TIME PER SAMPLE: ", float(ELAPSED/N_SAMPLES))
  print("TIME PER BATCH: ", float((ELAPSED/N_SAMPLES) * batch_size))

def test_train():
  trainer = NeuroCuberTrainer("res_models/train_test.json")
  initialize_neurocuber(trainer.model)
  check_make_path(trainer.ckpt_dir)
  with open(os.path.join(trainer.ckpt_dir, "train_log.txt"), "w") as f:
    f.write(f"{datetime.datetime.now()}: starting training loop\n")
    trainer.model.summary(print_fn = lambda x: f.write(x + "\n"))
  trainer.train()
  
# if __name__ == "__main__":
#   debug_batching()

# if __name__ == "__main__":
#   test_train()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-cfg", action="store", dest="train_cfg", type=str)
  parser.add_argument("--interactive", action="store_true", dest="interactive")
  opts = parser.parse_args()

  if opts.train_cfg is None:
    raise Exception("path to train_cfg required")

  if opts.interactive:
      print("Parsed options:")

      for key in vars(opts):
        print(f"  {key} := {vars(opts)[key]}")

      if opts.prompt:
        print("Continue?")
        intput()

  trainer = NeuroCuberTrainer(opts.train_cfg)
  initialize_neurocuber(trainer.model)
  if not os.path.exists(trainer.ckpt_dir):
    os.makedirs(trainer.ckpt_dir)
  check_make_path(trainer.ckpt_dir)
  with open(os.path.join(trainer.ckpt_dir, "train_log.txt"), "w") as f:
    f.write(f"{datetime.datetime.now()}: starting training loop\n")
    trainer.model.summary(print_fn = lambda x: f.write(x + "\n"))
  trainer.train()
