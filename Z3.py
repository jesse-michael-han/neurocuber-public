from sat_util import *
from pysat.formula import CNF
from gen_fmlas import get_unsat_randkcnf

def opts(max_conflicts):
    return Z3Options(max_conflicts=max_conflicts, sat_restart_max=0)

def int_to_Lit(k):
  assert not k == 0
  if k > 0:
    v = Var(k-1)
    return Lit(v,True)
  else:
    v = Var(abs(k)-1)
    return Lit(v, False)

def Lit_to_int(l):
  if l.sign():
    return l.var().idx() + 1
  else:
    return -(l.var().idx() + 1)

def clauses_to_Clauses(clauses):
  return [list(map(int_to_Lit, cls)) for cls in clauses]

def Clauses_to_clauses(Clauses):
  return [list(map(Lit_to_int, cls)) for cls in Clauses]
  
def SATProblemFromCNF(fmla):
  return SATProblem(fmla.nv, clauses_to_Clauses(fmla.clauses))

def SATProblemToCNF(problem):
  return CNF(from_clauses=Clauses_to_clauses(problem.clauses()))

class Z3Cuber:
  def __init__(self, name, lookahead_rewards=["march_cu"]):
      self.name = name
      self.lookahead_rewards = lookahead_rewards

  def cube(self, s# , assumptions
  ):
      lr = self.lookahead_rewards[0]
      status, lits = s.cube(# assumptions=assumptions,
                            lookahead_reward=lr, lookahead_delta_fraction=1.0)
      if status != Z3Status.unknown: return None
      else: return lits[0].var()

class Z3VarSelector:
  def __init__(self, name="bob"):
    self.name = name

  def __call__(self, fmla):
    problem = SATProblemFromCNF(fmla)
    S = Z3Solver(problem, opts(max_conflicts=0))
    status, lits = S.cube(lookahead_reward="march_cu", lookahead_delta_fraction=1.0)
    if not status == Z3Status.unknown:
      return None
    else:
      return lits[0].var().idx()

if __name__ == "__main__":
    test_1_dimacs = CNF(from_clauses=[[1,2,3,4],[-1,-2,-3],[5,6],[1,3,5],[-4,-2,-1],[-5],[4,-2,5]])
    test_1_problem = SATProblemFromCNF(test_1_dimacs)
    s = Z3Solver(test_1_problem, opts(max_conflicts=100))
    assert s.check() ==  Z3Status.sat
