import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import time

PWD = os.path.dirname(os.path.realpath(__file__))

VALID_TRAIN_DATASETS = ["sr", "src"]

VALID_EVAL_DATASETS = ["ramsey", "schur", "vdw"]

VALID_METRICS = ["top1", "top3", "random playout"]

PLOT_DPI = 300

def CAvgSeries(series):
  cavg = 0
  count = 0
  result = []
  for v in series:
    cavg = (1.0/(count+1) * ((count * cavg) + v))
    result.append(cavg)
    count += 1
  return pd.Series(result)

def DataFrameMap(map_fn, df): # TODO(jesse): test
  new_series_dict = dict()
  for clabel in df.columns:
    new_series_dict[clabel] = CAvgSeries(df[clabel])

  return pd.DataFrame(new_series_dict)

def mk_neurocuber_dataframe(path_to_folder):
  """
  Args:
  path to a directory containing CSV files exported from tensorboard.

  Returns:
  a pd.Dataframe aggregating all values in all CSV files according to hardcoded naming conventions
  """

  train_models = VALID_TRAIN_DATASETS
  heads = ["core", "drat"]

  neuros = [train_model + "_" + head for train_model in train_models for head in heads]

  rootnames = neuros + ["random", "z3"]

  for rootname in rootnames:
    try:
      assert rootname + ".csv" in os.listdir(path_to_folder)
    except AssertionError:
      try:
        assert rootname == "z3"
      except AssertionError:
        raise Exception(f"missing CSV file for {rootname}")

  series_dict = dict()

  for rootname in rootnames:
    try:
      series_dict[rootname] = pd.read_csv(os.path.join(path_to_folder, rootname + ".csv"))["Value"]
    except FileNotFoundError:
      continue
      # series_dict[rootname] = pd.Series(None)

  return pd.DataFrame(series_dict)

class NeuroCuberData:
  def __init__(self, eval_dataset):
    self.eval_dataset = eval_dataset
    try:
      assert eval_dataset in VALID_EVAL_DATASETS
    except AssertionError:
      raise Exception(f"unsupported eval_dataset {eval_dataset}")

    self.make_dataframes()

  def make_dataframes(self):
    raise NotImplementedError

class PlayoutData(NeuroCuberData):
  def __init__(self, eval_dataset):
    super(PlayoutData, self).__init__(eval_dataset)

  def make_dataframes(self):
    self.data_dir = os.path.join(PWD, "playout", self.eval_dataset + "/")
    self.tvDataFrame = mk_neurocuber_dataframe(os.path.join(self.data_dir, "tv"))
    self.apDataFrame = mk_neurocuber_dataframe(os.path.join(self.data_dir, "ap"))
    self.avg_tvDataFrame = DataFrameMap(CAvgSeries, self.tvDataFrame)
    self.avg_apDataFrame = DataFrameMap(CAvgSeries, self.apDataFrame)

  def mk_tv_plot(self):
    self.avg_tvDataFrame.plot(kind="line")
    plt.legend(prop={"size":8}, loc="best", ncol=3)
    plt.xlabel("# formulas")
    plt.ylabel("average terminal value")
    plt.title(f"random playout evaluation on {self.eval_dataset}")
    plt.savefig(f"plots/randomplayout_tv_{self.eval_dataset}.png", dpi=PLOT_DPI)
    plt.close()

  def mk_ap_plot(self):
    self.avg_apDataFrame.plot(kind="line")
    plt.legend(prop={"size":8}, loc="best", ncol=3)
    plt.xlabel("# formulas")
    plt.ylabel("average unit props/round")
    plt.title(f"random playout evaluation on {self.eval_dataset}")
    plt.savefig(f"plots/randomplayout_ap_{self.eval_dataset}.png", dpi=PLOT_DPI)
    plt.close()

  def mk_both_plot(self):
    fig, axes = plt.subplots(1,2, squeeze=False, figsize=(7, 2.5), constrained_layout=True)
    l1 = self.avg_tvDataFrame.plot(ax=axes[0,0],kind="line",legend=False)
    l2 = self.avg_apDataFrame.plot(ax=axes[0,1],kind="line", legend=False)
    l1.set_xlabel("# formulas")
    l2.set_xlabel("# formulas")
    l1.set_ylabel("avg terminal value")
    l2.set_ylabel("avg unit props")
    l1.legend(prop={"size":7.5}, loc="upper center", ncol=3, handlelength=0.7, labelspacing=0.25, borderpad=0.4)
    # l1.set_title("average terminal value")
    # l2.set_title("average unit props/round")
    # l1.set_aspect(15000)
      # ax.set_aspect("equal")
    # fig.legend(prop={"size":8}, loc="lower right", ncol=3)
    # plt.xlabel("# formulas", ax=axes[0,0])
    # plt.ylabel("average terminal value", ax=axes[0,0])
    # plt.ylabel("average unit props/round", ax=axes[0,1])
    # plt.suptitle(f"random playout evaluation on {self.eval_dataset}", ax=axes[0,0])
    # plt.suptitle(f"random playout evaluation on {self.eval_dataset}", ax=axes[0,1])
    plt.savefig(f"plots/randomplayout_both_{self.eval_dataset}.png", dpi=PLOT_DPI)
    plt.close()

class Top1Data(NeuroCuberData):
  def __init__(self, eval_dataset):
    super(Top1Data, self).__init__(eval_dataset)

  def make_dataframes(self):
    self.data_dir = os.path.join(PWD, "top1", self.eval_dataset + "/")

    self.timingDataFrame = mk_neurocuber_dataframe(os.path.join(self.data_dir)).head(250)
    self.avg_timingDataFrame = DataFrameMap(CAvgSeries, self.timingDataFrame)

  def mk_plot(self): # cumulative average plot
    self.avg_timingDataFrame.plot(kind="line")
    plt.legend(prop={"size":7}, ncol=3)
    plt.xlabel("# formulas")
    plt.ylabel("seconds (wall clock)")
    plt.title(f"top-1 timing evaluation on {self.eval_dataset}")
    plt.savefig(f"plots/top1{self.eval_dataset}.png", dpi=PLOT_DPI)
    plt.close()

class Top3Data(NeuroCuberData):
  def __init__(self, eval_dataset):
    super(Top3Data, self).__init__(eval_dataset)

  def make_dataframes(self):
    self.data_dir = os.path.join(PWD, "top3", self.eval_dataset + "/")

    self.timingDataFrame = mk_neurocuber_dataframe(os.path.join(self.data_dir)).head(250)
    self.avg_timingDataFrame = DataFrameMap(CAvgSeries, self.timingDataFrame)

  def mk_plot(self): # cumulative average plot
    self.avg_timingDataFrame.plot(kind="line")
    plt.legend(prop={"size":7}, ncol=3)
    plt.xlabel("# formulas")
    plt.ylabel("seconds (wall clock)")
    plt.title(f"top-3 timing evaluation on {self.eval_dataset}")
    plt.savefig(f"plots/top3{self.eval_dataset}.png", dpi=PLOT_DPI)
    plt.close()

def percent_change(a,b):
  """
  Returns the percentage change from a to b.
  """
  return float((b - a)/a)

def DRAT_vs_core_aux(train_dataset, eval_dataset):
  try:
    assert train_dataset in VALID_TRAIN_DATASETS
  except AssertionError:
    raise Exception(f"invalid train_dataset {train_dataset}")
  try:
    assert eval_dataset in VALID_EVAL_DATASETS
  except AssertionError:
    raise Exception(f"invalid eval_dataset {eval_dataset}")
  # try:
  #   assert metric in VALID_METRICS
  # except AssertionError:
  #   raise Exception(f"invalid metric {metric}")

  result_dict = {} # currently reload the dataset each run. this objectively sucks, but it works

  top1data = Top1Data(eval_dataset)
  top3data = Top3Data(eval_dataset)
  playoutdata = PlayoutData(eval_dataset)

  drat_name = train_dataset + "_" + "drat"

  core_name = train_dataset + "_" + "core"

  top1data.avg_timingDataFrame.tail(1)[core_name]


  # note(jesse, December 03 2019, 01:24 PM): we flip the sign so that higher percentage means improvement
  def percent_change_postprocess(x):
    return round(100 * (-1.0) * x, 2)

  result_dict["top1 timing"] = percent_change_postprocess(percent_change(top1data.avg_timingDataFrame.tail(1)[core_name], top1data.avg_timingDataFrame.tail(1)[drat_name]))
  result_dict["top3 timing"] = percent_change_postprocess(percent_change(top3data.avg_timingDataFrame.tail(1)[core_name], top3data.avg_timingDataFrame.tail(1)[drat_name]))
  result_dict["random playout"] = percent_change_postprocess(percent_change(playoutdata.avg_tvDataFrame.tail(1)[core_name], playoutdata.avg_tvDataFrame.tail(1)[drat_name]))

  return pd.Series(result_dict)

def DRAT_vs_core_DataFrame(train_dataset): # TODO(jesse): test this
  series_dict = dict()
  for eval_dataset in VALID_EVAL_DATASETS:
    series_dict[eval_dataset] = DRAT_vs_core_aux(train_dataset, eval_dataset)

  return pd.DataFrame(series_dict)

def playout_DataFrame(eval_dataset):
  path_to_folder=f"playout/{eval_dataset}/"
  tv_df = DataFrameMap(CAvgSeries, mk_neurocuber_dataframe(os.path.join(path_to_folder, "tv/")))
  ap_df = DataFrameMap(CAvgSeries, mk_neurocuber_dataframe(os.path.join(path_to_folder, "ap/")))
  tv_row = tv_df.tail(1).rename(index={tv_df.index[-1]:"avg terminal value"})
  ap_row = ap_df.tail(1).rename(index={ap_df.index[-1]:"avg unit props"})
  return pd.concat([tv_row, ap_row]).round(3)

def test_pdf():
  print(playout_DataFrame("ramsey"))

def timing_DataFrame(kind):
  assert kind == "top1" or kind == "top3"
  dfs = []
  for eval_dataset in VALID_EVAL_DATASETS:
    last_avg_row = DataFrameMap(CAvgSeries, mk_neurocuber_dataframe(os.path.join(kind + "/", eval_dataset + "/"))).tail(1)
    last_avg_row = last_avg_row.rename(index={last_avg_row.index[-1]:eval_dataset})
    dfs.append(last_avg_row)
  return pd.concat(dfs).round(3)

def test_tdf():
  print(timing_DataFrame("top1"))

def generate_other_tables():
  p_df = playout_DataFrame("ramsey")
  p_df.to_csv(f"tables/ramsey_playout_table.csv")
  t_df = timing_DataFrame("top1")
  t_df.to_csv(f"tables/timing_table.csv")
  print("other tables generated")

def generate_other_plots():
  p_df = playout_DataFrame("ramsey")
  self.avg_timingDataFrame.plot(kind="line")
  plt.legend(prop={"size":7}, ncol=3)
  plt.xlabel("# formulas")
  plt.ylabel("seconds (wall clock)")
  plt.title(f"top-3 timing evaluation on {self.eval_dataset}")
  plt.savefig(f"plots/top3{self.eval_dataset}.png", dpi=PLOT_DPI)
  plt.close()

def generate_ramsey():
  pdata = PlayoutData("ramsey")
  pdata.mk_both_plot()
  print("generated dual plot for ramsey")

def generate_all_plots():
  for eval_dataset in VALID_EVAL_DATASETS:
    t1data = Top1Data(eval_dataset)
    t3data = Top3Data(eval_dataset)
    pdata = PlayoutData(eval_dataset)

    t1data.mk_plot()
    t3data.mk_plot()
    pdata.mk_tv_plot()
    pdata.mk_ap_plot()
    pdata.mk_both_plot()
  print("generated all plots")

def generate_all_tables():
  for train_dataset in VALID_TRAIN_DATASETS:
    DRAT_core_data = DRAT_vs_core_DataFrame(train_dataset)
    DRAT_core_data.to_csv(f"tables/{train_dataset}_table.csv")

  print("generated all tables")

def generate_all():
  start = time.time()
  generate_all_plots()
  generate_all_tables()
  generate_other_tables()
  print("processed all data in", time.time() - start, "seconds")
  print("done")

def exactly_one(bools):
  FOUND_FLAG = False
  for b in bools:
    if b:
      if FOUND_FLAG:
        return False
      FOUND_FLAG = True
  return FOUND_FLAG

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--all", action="store_true", dest="all")
  parser.add_argument("--tables", action="store_true", dest="tables")
  parser.add_argument("--plots", action="store_true", dest="plots")
  parser.add_argument("--ramsey", action="store_true", dest="ramsey" ) 
  opts = parser.parse_args()

  assert exactly_one([opts.all, opts.tables, opts.plots, opts.ramsey])

  for folder in ["plots/", "tables"]:
    if not os.path.exists(os.path.join(PWD, folder)):
      os.makedirs(os.path.join(PWD, folder))

  if opts.all:
    generate_all()
  if opts.tables:
    generate_all_tables()
    generate_other_tables()
  if opts.plots:
    generate_all_plots()
  if opts.ramsey:
    generate_ramsey()
