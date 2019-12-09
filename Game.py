from cnf_util import *
import numpy as np
from typing import List

# utilities for single-threaded MCTS

class Node(object):

  def __init__(self, prior: float, parent=None):
    self.visit_count = 0
    self.to_play = None
    self.prior = prior
    self.value_sum = 0
    self.parent = parent
    self.children = {}

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class Network:
  def inference(self, image):
    raise NotImplemented

  def get_weights(self):
    pass

  def __call__(self, image):
    return self.inference(image)
  
class ProverAdversaryGame:
  def __init__(self, fmla, history=None, var_dict=None, current_var=None, PROVER_TURN=True, trail = None, solver_name="glucose3"):
    self.history = history or [] # store triples (formula, variable, polarity)
    self.current_var = current_var
    if trail is None:
      self.trail = []
    else:
      self.trail = trail
    self.child_visits = []
    self.fmla = fmla
    self.n_max_var = fmla.nv
    self.DONE_FLAG = False
    self.PROVER_TURN = PROVER_TURN
    self.var_dict = None
    self.initialize_var_dict()
    self.solver = Solver(name=solver_name, bootstrap_with=self.fmla)
    self.terminal_value = None
    
  def initialize_var_dict(self):
    assert self.var_dict is None
    self.var_dict = {}
    for cls in self.fmla.clauses:
      for lit in cls:
        self.var_dict[abs(lit)] = True

  def num_actions(self):
    if self.PROVER_TURN:
      return self.n_max_var
    else:
      return 2
        
  def terminal(self):
    return self.DONE_FLAG

  def terminal_value(self):
    assert self.terminal_value is not None
    assert self.DONE_FLAG
    return self.terminal_value

  def legal_actions(self, to_play=None):
    # returns indices for logit
    if to_play is None:
      to_play = self.to_play()
    result = []
    if to_play == 0:
      for var in range(1, self.fmla.nv+1):
        try:
          if self.var_dict[var]:
            result.append(var-1)
          else:
            continue
        except KeyError:
          continue
    else:
      result = [True, False]
    return result

  def clone(self):
    return ProverAdversaryGame(self.fmla.copy(), deepcopy(self.history), deepcopy(self.var_dict), self.current_var, self.PROVER_TURN, deepcopy(self.trail))

  def clone_backtrack(self, n):
    assert n < 0
    return ProverAdversaryGame(self.history[n][0])

  def apply(self, action):
    if self.PROVER_TURN:
      assert type(action) is not bool and self.current_var is None
      self.current_var = action+1 # index shifting
      self.PROVER_TURN = False
    else:
      assert type(action) is bool
      assert self.current_var is not None
      lit = self.current_var if action else -self.current_var
      self.trail.append(lit) # manually add unit clauses to the trail
      for cls in self.fmla.clauses:
        if len(cls) == 1:
          self.trail.append(cls[0])
      self.history.append((self.fmla, self.current_var, action))
      result, prop_lits = self.solver.propagate(assumptions=self.trail)
      self.trail = prop_lits # TODO(jesse): consider destroying the solver each round amd refeeding the formula
      for lit in self.trail:
        self.var_dict[abs(lit)] = False # mask out assigned variables
      self.fmla = simplify_CNF(self.fmla, prop_lits)
      if not result:
        self.DONE_FLAG = True
        self.solver.delete() # destroy the solver once the game is done
      else:
        pass
      self.current_var = None
      self.PROVER_TURN = True
      self.terminal_value = len(self.history)/float(len(self.var_dict))

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])

  def make_image(self, state_index, return_fmla=False):
    if return_fmla:
      return self.fmla
    else:
      if state_index == -1:
        fmla = self.fmla
      else:
        fmla = self.history[state_index][0]
      n_clauses = len(fmla.clauses)
      n_vars = fmla.nv
      CL_idxs = clgraph(fmla)
      G_cls = [G_cl_of_idxs(n_clauses, n_vars, CL_idxs)]
      result = G_cls, tf.cast([n_clauses], tf.int64), tf.cast([n_vars], tf.int64)
      return result
  # TODO(jesse): condition on the trail?

  def make_target(self, state_index: int):
    return (self.terminal_value(),
            self.child_visits[state_index])

  def to_play(self):
    if self.PROVER_TURN:
      return 0
    else:
      return 1

class AlphaCuberConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.max_moves = 512 # note, this will be the number of variables in the formula; less if the formula is large
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }

def dummy_config():
  result = AlphaCuberConfig()
  result.num_actors = 5
  result.num_sampling_moves = 30
  result.max_moves = 50
  result.num_simulations = 50
  return result
    
class ReplayBuffer(object):

  def __init__(self, config):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    move_sum = float(sum(len(g.history) for g in self.buffer))
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class SharedStorage(object): # TODO(jesse): implement this stub
  def __init__(self):
    self._networks = {}

  def latest_network(self):
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      return None # make_uniform_network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network):
    self._networks[step] = network  

def run_selfplay(config, storage,
                 replay_buffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config, network):
  game = ProverAdversaryGame()
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game

# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaCuberConfig, game: ProverAdversaryGame, network): # network should be an Actor
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(config, root)
  print("initialization of MCTS done")

  for _ in range(config.num_simulations):
    print("STARTING NEW SIMULATION")
    node = root
    scratch_game = game.clone()
    search_path = [node]
    network.reset_results()

    while node.expanded():
      print("node children", node.children.values())
      print("search path: ", search_path)
      print("node is expanded, selecting a child")
      print("turn is now:", scratch_game.to_play())
      print("selecting a child")
      action, node = select_child(config, node)
      print("selected a child")
      scratch_game.apply(action)
      print("applied action")
      search_path.append(node)
      print("new search path: ", search_path)

    print("node is not expanded, expanding")
    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
    print("SIMULATION COMPLETE")
  return select_action(config, game, root), root

def test_run_mcts(resume_flag):
  for _ in range(10):
    if resume_flag:
      print("resuming")
      with open("cnf/mcts_test_fmla.cnf", "r") as f:
        fmla = CNF(from_fp=f)
    else:
      fmla = get_unsat_randkcnf(3,10)
      with open("cnf/mcts_test_fmla.cnf", "w") as f:
        fmla.to_fp(f)
    print("got formula")
    print(fmla.nv)
    game = ProverAdversaryGame(fmla)
    print("initialized game")
    config = dummy_config()
    print("dummy configuration initialized")
    actor = DummyActor()
    print("dummy network initialized")
    run_mcts(config, game, actor)
  
def softmax_sample(visit_counts_actions):
  visit_counts = np.array([x[0] for x in visit_counts_actions], dtype="float")
  softmaxed_visit_counts = tf.nn.softmax(visit_counts)
  sample_index = np.random.choice(len(visit_counts_actions), p=softmaxed_visit_counts)
  return visit_counts_actions[sample_index]
  
def select_action(config: AlphaCuberConfig, game: ProverAdversaryGame, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.items()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action

# Select the child with the highest UCB score.
def select_child(config: AlphaCuberConfig, node: Node):
  #TODO(jesse): test
  action, child = max(node.children.items(), key=lambda pr: ucb_score(config, node, pr[1]))
  print (action, child)
  return action, child



# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaCuberConfig, parent: Node, child: Node):
  print("starting ucb score calculation")
  pb_c = np.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
  
  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


class Actor:
  def __init__(self):
    self.result_dict = {}
    self.results = None

  def get_prover_logits(self, image):
    raise NotImplementedError

  def get_adversary_logits(self, image):
    raise NotImplementedError

  def get_value(self, image, game):
    raise NotImplementedError

  def post_evaluate_hook(self, node):
    pass

  def evaluate(self, node, game):
    print("evaluating")
    node.to_play = game.to_play()
    current_image = game.make_image(-1)
    current_var = game.current_var
    # print(current_image)
    value = self.get_value(current_image, game, node)
    if node.to_play == 0:
      policy_logits = self.get_prover_logits(current_image, game)
      game.fmla.to_fp(sys.stdout)
      print("LEGAL ACTIONS", game.legal_actions(node.to_play))
      print("POLICY LOGITS BEFORE POLICY DICT", policy_logits)
      policy = {a :np.exp(policy_logits[a]) for a in game.legal_actions(node.to_play)} # off by one error here      
    else:
      print("getting adversary logits")
      policy_logits = self.get_adversary_logits(current_image, current_var, game, node)
      print("got policy logits")
      print("LEGAL ACTIONS", game.legal_actions(node.to_play))            
      policy = {a :np.exp(policy_logits[0 if a is True else 1]) for a in game.legal_actions(node.to_play)}

    print("POLICY LOGITS", "TURN", game.to_play(),":", policy_logits)
    print("POLICY VECTOR", policy)
    policy_sum = sum(policy.values())
    print("populating children")
    for action, p in policy.items():
      print("USING THIS POLICY", p)
      node.children[action] = Node(p / policy_sum, parent = node)
    if game.PROVER_TURN:
      self.result_dict[node] = self.results
    print("populated children")
    self.post_evaluate_hook(node)
    return value

def evaluate(node :Node, game:ProverAdversaryGame, actor: Actor):
  return actor.evaluate(node, game)

# TODO(jesse): make softmax temperature configurable?
class NeuroActor(Actor):
  def __init__(self, network):
    # a network is a callable object which accepts a game state and returns a 3-tuple of prover logits, adversary logits, and value estimation (in this case, normalized episode step)
    super(NeuroActor, self).__init__()
    self.network = network
    self.current_var = None
    self.RESULTS_COMPUTED = False
    self.results = None

  def maybe_compute_results(self, image, game, node=None):
    if node is not None and not game.PROVER_TURN:
      self.results = self.result_dict[node.parent]
      self.RESULTS_COMPUTED = True
    if not self.RESULTS_COMPUTED:
      self.results = self.network(image)
      self.RESULTS_COMPUTED = True

  def reset_results(self):
    self.results = None
    self.RESULTS_COMPUTED = False
    self.current_var = None

  def get_prover_logits(self, image, game):
    self.maybe_compute_results(image, game)
    return self.results[0]

  def get_adversary_logits(self, image, current_var, game, node):
    self.maybe_compute_results(image, game, node)
    return tf.transpose(self.results[1])[current_var - 1]

  def get_value(self, image, game, node):
    self.maybe_compute_results(image, game, node)
    print("IMAGE BEFORE GETTING VALUE", image)
    return self.get_value_aux(self.results[2], game)

  def get_value_aux(self, value_logits, game):
    """
    Args:
    value_logits: a 2 x num_lits matrix of normalized esteps, representing afterstate value estimation for each of Adversary's moves; afterstate value estimation for Prover is a weighted average of the value estimates for the next two branches (weighted by Adversary's logits)
    game: a ProverAdversaryGame object

    Returns:
    a scalar representing the afterstate value estimate for this node
    """
    if not game.PROVER_TURN: # TODO(jesse): compute and store tf.matmul(tf.transpose(value_logits), self.results[1]) after querying the network
      current_var = game.current_var
      print("CURRENT VAR",current_var)
      print("VALUE LOGITS BEFORE SLICE", value_logits)
      print("CURRENT_VAR BEFORE SLICE", current_var)
      lit_logits = value_logits[:,current_var-1]
      adversary_logits = self.results[1]
      prior_ps = tf.nn.softmax(adversary_logits[:, current_var-1])
      print("VALUES HERE")
      print(np.dot(lit_logits, prior_ps))
      return np.dot(lit_logits, prior_ps)
    else:
      self.current_var = game.current_var
      # print(tf.transpose(value_logits))
      x = tf.reduce_sum(tf.multiply(tf.transpose(value_logits), tf.nn.softmax(tf.transpose(self.results[1]))), axis=-1)
      # print(x)
      z = tf.reduce_sum(tf.multiply(x, tf.nn.softmax(self.results[0])))
      return z

  def post_evaluate_hook(self, node):
    if node.to_play == 0:
      pass
    else:
      self.reset_results()
      
class DummyNetwork(Network):
  def __init__(self):
    super(DummyNetwork, self).__init__()
    pass

  def inference(self, arg):
    n_clauses = arg[0][0]
    n_vars = arg[0][1]
    print("NUMBER OF VARIABLES BEFORE INFERENCE", n_vars)
    prover_logits = np.random.uniform(size=(n_vars))
    adversary_logits = np.random.uniform(size=(2,n_vars))
    value_logits = tf.nn.sigmoid(np.random.uniform(size=(2,n_vars)))
    return [prover_logits, adversary_logits, value_logits]

class DummyActor(NeuroActor):
  def __init__(self):
    super(DummyActor, self).__init__(DummyNetwork())
  
# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == 1 else (1 - value)
    node.visit_count += 1

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config, node):
  actions = node.children.keys()
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    print("PRIOR PRIOR",node.children[a].prior)
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
    print("NEW PRIOR", node.children[a].prior)

### testing ###

# interactive mode for testing        
def interactive_test():
  while True:
    seed_fmla = sample_randkcnf(3,5)
    with Solver(name="cdl", bootstrap_with=seed_fmla) as S:
      if S.solve():
        continue
      else:
        break
  game = ProverAdversaryGame(seed_fmla)
  while not game.DONE_FLAG:
    print("current formula:")
    game.fmla.to_fp(sys.stdout)
    print("")
    if game.PROVER_TURN:
      print("pick a variable:")
      arg = input()
      if arg == "backtrack":
        game = game.clone_backtrack(-1)
        print("backtracking by 1")
        continue
      else:
        current_var = int(arg)
      game.apply(current_var)
    else:
      print("pick a value")
      value = int(input())
      game.apply(bool(value))
    if game.DONE_FLAG:
      print("done")

  
