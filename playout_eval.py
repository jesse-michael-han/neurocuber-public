from cnf_util import *
from Game import ProverAdversaryGame
import numpy as np
from query_model import *
from gen_fmlas import get_unsat_randkcnf, get_unsat_sr, get_unsat_src
from Z3 import *
from types import SimpleNamespace
import ray
import numpy as np
from util import *

# from playout_eval import Match

def softmax_sample(action_logit_list, tau=3): # TODO(jesse): test
  logits = [x[1] for x in action_logit_list]
  actions = [x[0] for x in action_logit_list]
  ps = tf.nn.softmax(np.array(logits, dtype="float32"))
  selected_action = random.choices(actions, ps)[0]
  return selected_action

class NeuroDRATProver():
  def __init__(self, model_cfg):
    self.oracle = NeuroCuberQuery(model_cfg)

  def get_action(self, image, legal_actions):
    drat_logits = self.oracle.get_logits(image)[0]
    action_logit_list = [(i, drat_logits[i]) for i in legal_actions]
    return softmax_sample(action_logit_list)

class NeuroCoreVarProver():
  def __init__(self, model_cfg):
    self.oracle = NeuroCuberQuery(model_cfg)

  def get_action(self, image, legal_actions):
    drat_logits = self.oracle.get_logits(image)[1]
    action_logit_list = [(i, drat_logits[i]) for i in legal_actions]
    return softmax_sample(action_logit_list)

class RandomProver():
  def __init__(self):
    pass

  def get_action(self, image, legal_actions):
    choice = int(np.random.choice(legal_actions))
    return choice

class RandomAdversary():
  def __init__(self):
    pass

  def __call__(self, image, legal_actions):
    choice = bool(np.random.choice(legal_actions))
    return choice

class Match:
  def __init__(self, game, prover, adversary):
    self.game = game
    self.prover = prover
    self.adversary = adversary
    self.start_fmla = self.game.fmla
    self.move_history = []
    self.terminal_value = None
    self.unit_prop_counts = []

  def statistics(self):
    return np.array([np.mean(self.unit_prop_counts), self.terminal_value])

  def play_round(self):
    assert not self.game.DONE_FLAG
    image = self.game.make_image(0, True)
    legal_actions = self.game.legal_actions()
    if self.game.PROVER_TURN:
      action = self.prover.get_action(image, legal_actions) # call remote prover
    else:
      action = self.adversary(image, legal_actions)
    if not self.game.PROVER_TURN:
      old_trail_length = len(self.game.trail) + 1
    self.game.apply(action)
    if self.game.PROVER_TURN:
      self.unit_prop_counts.append(len(self.game.trail) - old_trail_length)

  def play_game(self):
    while not self.game.DONE_FLAG:
      self.play_round()
    # self.move_history = [(x[1], x[2]) for x in self.game.history]
    self.terminal_value = self.game.terminal_value
    return self.statistics()

@ray.remote
class NeuroProverService:
  def __init__(self, model_cfg):
    self.oracle = NeuroCuberQuery(model_cfg)

  @ray.method(num_return_vals=1)
  def get_action(self, image, legal_actions):
    drat_logits = self.oracle.get_logits(image)[0]
    action_logit_list = [(i, drat_logits[i]) for i in legal_actions]
    return softmax_sample(action_logit_list)

class RemoteMatch: # initialized with actor handle of a NeuroProverService
  def __init__(self, game, prover, adversary):
    self.game = game
    self.prover = prover
    self.adversary = adversary
    self.start_fmla = self.game.fmla
    self.move_history = []
    self.terminal_value = None
    self.unit_prop_counts = []

  def statistics(self):
    return [np.mean(self.unit_prop_counts), self.terminal_value]

  def play_round(self):
    assert not self.game.DONE_FLAG
    image = self.game.make_image(0, True)
    legal_actions = self.game.legal_actions()
    if self.game.PROVER_TURN:
      action = ray.get(self.prover.get_action.remote(image, legal_actions)) # call remote prover
    else:
      action = self.adversary(image, legal_actions) # assume adversary is local
    if not self.game.PROVER_TURN:
      old_trail_length = len(self.game.trail) + 1
    self.game.apply(action)
    if self.game.PROVER_TURN:
      self.unit_prop_counts.append(len(self.game.trail) - old_trail_length)

  def play_game(self):
    while not self.game.DONE_FLAG:
      self.play_round()
    self.terminal_value = self.game.terminal_value
    return self.statistics()

@ray.remote
def play_match_remote(fmla, prover, adversary): # accepts reference to a prover service instead of instantiating own prover
  match = RemoteMatch(ProverAdversaryGame(fmla),prover,adversary)
  return match.play_game()

@ray.remote
def play_match(fmla, model_cfg, adversary): # TODO(jesse): make more efficient
  prover = NeuroDRATProver(model_cfg)
  match = Match(ProverAdversaryGame(fmla),prover,adversary)
  return match.play_game()

def play_match_single(fmla, model_cfg, adversary): # TODO(jesse): make more efficient
  prover = NeuroDRATProver(model_cfg)
  match = Match(ProverAdversaryGame(fmla),prover,adversary)
  return match.play_game()

def ParallelFormulaPlayout(fmla, model_cfg, adversary, num_matches):
  jobs = []
  for _ in range(num_matches):
    jobs.append(play_match.remote(fmla, model_cfg, adversary))
  return jobs

def SequentialFormulaPlayout(fmla, model_cfg, adversary, num_matches):
  results = []
  for _ in range(num_matches):
    results.append(play_match_single(fmla, model_cfg, adversary))
  return results

def test_running_time():
  ray.init()
  fmla = get_unsat_randkcnf(3,70)
  NUM_MATCHES = 100

  start = time.time()
  jobs = ParallelFormulaPlayout(fmla, "res_models/res_grid_0_randkcnf.json", RandomAdversary(), NUM_MATCHES)
  stats = np.mean(ray.get(jobs), axis=0)
  elapsed1 = time.time() - start
  print(stats)
  ray.shutdown()

  start = time.time()
  results = SequentialFormulaPlayout(fmla, "res_models/res_grid_0_randkcnf.json", RandomAdversary(), NUM_MATCHES)
  stats = np.mean(results, axis=0)
  print(stats)


  print("elapsed time", elapsed1)
  # print("elapsed time", elapsed2)
  print("elapsed time", time.time() - start)

@ray.remote
class FormulaWorker:
  def __init__(self, prover_type, model_cfg, adversary, index):
    if prover_type == "z3":
      self.prover = Z3Prover()
    elif prover_type == "random":
      self.prover = RandomProver()
    elif prover_type == "drat":
      assert model_cfg is not None
      self.prover = NeuroDRATProver(model_cfg)
    elif prover_type == "core":
      assert model_cfg is not None
      self.prover = NeuroCoreVarProver(model_cfg)
    else:
      raise Exception("unsupported prover type")
    self.adversary = RandomAdversary()
    self.index = index

  @ray.method(num_return_vals=1)
  def play_match(self, fmla):
    match = Match(ProverAdversaryGame(fmla), self.prover, self.adversary)
    return match.play_game(), self.index

def ParallelPlayout(fmla, num_workers, num_matches, prover_type, model_cfg, adversary=RandomAdversary()):
  assert num_matches > num_workers
  fmla = ray.put(fmla)

  worker_pool = [FormulaWorker.remote(prover_type, model_cfg, adversary, i) for i in range(num_workers)]


  active_jobs = [worker.play_match.remote(fmla) for worker in worker_pool]
  results = []

  while len(results) < num_matches - num_workers:
    ready_ids, active_jobs = ray.wait(active_jobs)
    for x in ready_ids:
      result, index = ray.get(x)
      results.append(result)
    active_jobs.append(worker_pool[index].play_match.remote(fmla))

  for job in active_jobs:
    result, _ = ray.get(job)
    results.append(result)

  return np.mean(results, axis=0)


def playout_eval(experiment_name, prover_type, data_dir, model_cfg, n_cpus, n_matches, test): # play 50 matches per formula
  ray.init(num_cpus=n_cpus)
  cnf_files = files_with_extension(data_dir, "cnf")
  step_count = 0

  if model_cfg is not None:
    name = prover_type + "_" + os.path.basename(os.path.splitext(model_cfg)[0])
  else:
    name = prover_type

  log_dir = os.path.join("playout_eval/", experiment_name, name +"/")
  writer = tf.summary.create_file_writer(log_dir)
  for f in cnf_files:
    fmla = CNF(from_file=f)
    avg_unit_props, avg_terminal_value = ParallelPlayout(fmla, n_cpus, n_matches, prover_type, model_cfg)
    with writer.as_default():
      tf.summary.scalar("avg unit props", avg_unit_props, step=(step_count+1))
      tf.summary.scalar("avg terminal value", avg_terminal_value, step=(step_count+1))
    print("finished step", step_count)
    step_count += 1



def test_playout_eval(experiment_name): # play 100 matches per formula
  ray.init(num_cpus=4)

  for prover_type in ["random", "drat", "core"]:
    for model_cfg in ["res_models/res_grid_2_sr.json", "res_models/res_grid_2_src.json"]:
      step_count = 0

      if model_cfg is not None:
        name = prover_type + "_" + os.path.basename(os.path.splitext(model_cfg)[0])
      else:
        name = prover_type

      log_dir = os.path.join("test_playout_eval/", experiment_name, name +"/")
      writer = tf.summary.create_file_writer(log_dir)
      for _ in range(3):
        fmla = get_unsat_randkcnf(3,40)
        avg_unit_props, avg_terminal_value = ParallelPlayout(fmla, 4, 10, prover_type, model_cfg)
        with writer.as_default():
          tf.summary.scalar("avg unit props", avg_unit_props, step=(step_count+1))
          tf.summary.scalar("avg terminal value", avg_terminal_value, step=(step_count+1))
        step_count += 1
      print(f"{name} ok")

# example usage:
# # python playout_eval.py EXPERIMENT_NAME drat --model=res_models/res_grid_2_sr.json --cpus=16 --matches=100 --data="cnf_data/ramsey/test/"
# # python playout_eval.py EXPERIMENT_NAME core --model=res_models/res_grid_2_src.json --cpus=16 --matches=100 --data="cnf_data/ramsey/test/"
# # python playout_eval.py EXPERIMENT_NAME z3 --cpus=16 --matches=50 --data="cnf_data/ramsey/test/"
# # python playout_eval.py EXPERIMENT_NAME random --cpus=16 --matches=100 --data="cnf_data/ramsey/test/"
# # python playout_eval.py test1 random --cpus=4 --matches=50 --data="cnf_data/ramsey/test/" --test
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment_name")
  parser.add_argument("prover_type")
  parser.add_argument("--data", action="store", dest="data_dir")
  parser.add_argument("--model", action="store", dest="model_cfg", default=None)
  parser.add_argument("--cpus", action="store", dest="n_cpus", type=int)
  parser.add_argument("--matches", action="store", dest="n_matches", type=int)
  parser.add_argument("--test", action="store_true", dest="test")

  opts = parser.parse_args()

  if opts.test:
    test_playout_eval(opts.experiment_name)
  else:
    playout_eval(**vars(opts))

  # for key in vars(opts):
  #   print(vars(opts)[key])

  # test_playout_eval(**vars(opts))

  # ramsey_playout_eval(**vars(opts))

# if __name__ == "__main__":
#   ray.init()
#   start = time.time()
#   result0, result1 = ParallelPlayout(get_unsat_randkcnf(3,100), 4, 100, "drat", "res_models/res_grid_2_randkcnf.json")
#   # result0, result1 = ParallelPlayout(gen_ramsey_fragment(4,4,18,35), 4, 100, "drat", "res_models/res_grid_0_randkcnf.json")
#   print("elapsed", time.time() - start)
#   print(result0, result1)
