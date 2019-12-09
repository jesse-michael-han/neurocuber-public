import tempfile
import datetime
import time
import os
import sys
import tensorflow as tf
import numpy as np
import ray
from cnf_util import *
from tftd import TFDC, tfdc_to_example
import sr
from util import *
from gen_fmlas import *

@ray.remote
def worker(writer, get_fmla, tmpdir):
  while True:
    try:
      writer.write_example.remote(tfdc_to_example(gen_tfdc(get_fmla(), tmpdir)))      
      return 0
    except FileNotFoundError:
      continue

@ray.remote
class Worker:
  def __init__(self, writer, get_fmla, tmpdir):
    self.datapoint = None
    self.get_fmla = get_fmla
    self.tmpdir = tmpdir
    self.writer = writer

  def _gen_datapoint(self):
    while True:
      try:
        self.datapoint = gen_tfdc(self.get_fmla(), self.tmpdir)
        break
      except FileNotFoundError: # TODO(jesse): fix this upstream
        continue

  def serialize_datapoint(self):
    return tfdc_to_example(self.datapoint)

  def main(self):
    self._gen_datapoint()
    self.writer.write_example.remote(self.serialize_datapoint())
    return 0

@ray.remote
class Writer:
  def __init__(self, data_dir, n_tfrs_per_file):
    self.start_time = time.time()
    self.opts = mk_data_writer_opts(data_dir, n_tfrs_per_file)
    self.wtr = data_writer(self.opts)
    self.write_count = 0
    datestring = datetime.date.strftime(datetime.datetime.now(),"%Y-%m-%d-%H-%M")
    self.log_path = os.path.join(data_dir,"logs/", "datagen_" + datestring + ".log")
    if not os.path.exists(os.path.dirname(self.log_path)):
      os.makedirs(os.path.dirname(self.log_path))
    with open(self.log_path, "a") as f:
      f.write(f"{datetime.datetime.now()}: writer initialized\n")

  def write_log(self, arg=None):
    with open(self.log_path, "a") as f:
      if arg is None:
        f.write(f"{datetime.datetime.now()}: current count {self.write_count}" + "\n")
        f.write(f"elapsed time: {datetime.timedelta(seconds=(time.time() - self.start_time))}" + "\n")
      else:
        f.write(arg + "\n")

  def write_example(self, e):
    print("got something, writing example")
    self.wtr.write_example(e)
    self.write_count += 1
    return 0

  def finalize(self):
    self.write_log("finalizing")
    self.write_log()
    self.wtr.finalize()

  def count(self):
    return self.write_count

def test_ray():
  data_dir = os.path.join(PROJECT_DIR, "train_data", "scratch")
  log_dir = os.path.join(data_dir, "log")
  n_tfrs_per_file = 150
  writer = Writer.remote(data_dir, n_tfrs_per_file)
  count = 0
  while count < 300:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = tmpdir + "/"
      jobs = []
      for _ in range(4):
        worker = Worker.remote(writer, lambda : get_unsat_randkcnf(3,40), tmpdir)
        jobs += [worker.main.remote() for _ in range(50)]
      ray.get(jobs)
      ray.get(writer.write_log.remote())
      count += ray.get(writer.count.remote())
      print("count: ", count)
  ray.get(writer.finalize.remote())

def gen_ramsey(s=4, k=4, N=18, c=30, data_dir = os.path.join(PROJECT_DIR, "train_data", "ramsey2", "test"), n_tfrs_per_file=100, n_datapoints=1000, batch_size=100, num_threads=6):
  check_make_path(data_dir)
  get_fmla = lambda: gen_ramsey_fragment(s,k,N,c)
  writer = Writer.remote(data_dir, n_tfrs_per_file)
  count = 0
  while count < n_datapoints:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = tmpdir + "/"
      jobs = []
      jobs_per_worker = int(np.ceil((np.minimum(abs(n_datapoints - count), batch_size))/num_threads))
      for _ in range(num_threads):
        worker = Worker.remote(writer, get_fmla, tmpdir)
        # jobs = jobs + [worker.remote(writer, get_fmla, tmpdir) for _ in range(jobs_per_worker)]
        jobs = jobs + [worker.main.remote() for _ in range(jobs_per_worker)]
      ray.get(jobs)
      ray.get(writer.write_log.remote())
      count = ray.get(writer.count.remote())
      print("count: ", count)
  ray.get(writer.finalize.remote())
  print("done")

def gen_randkcnf(k=3, n=40, data_dir = os.path.join(PROJECT_DIR, "train_data", "randkcnf", "train"), n_tfrs_per_file=5000, n_datapoints = 65000, batch_size = 500, num_threads = 4):
  check_make_path(data_dir)  
  get_fmla = lambda: get_unsat_randkcnf(k,n)
  writer = Writer.remote(data_dir, n_tfrs_per_file)
  count = 0
  while count < n_datapoints:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = tmpdir + "/"
      jobs = []
      jobs_per_worker = int(np.ceil((np.minimum(abs(n_datapoints - count), batch_size))/num_threads))
      for _ in range(num_threads):
        worker = Worker.remote(writer, get_fmla, tmpdir)
        # jobs = jobs + [worker.remote(writer, get_fmla, tmpdir) for _ in range(jobs_per_worker)]
        jobs = jobs + [worker.main.remote() for _ in range(jobs_per_worker)]
      ray.get(jobs)
      ray.get(writer.write_log.remote())
      count = ray.get(writer.count.remote())
      print("count: ", count)
  ray.get(writer.finalize.remote())
  print("done")

def gen_sr(n1=10, n2=40, min_cls_len=2, data_dir = os.path.join(PROJECT_DIR, "train_data", "sr", "train"), n_tfrs_per_file=5000, n_datapoints = 185000, batch_size = 500, num_threads = 4):
  check_make_path(data_dir)  
  get_fmla = lambda: get_unsat_sr(n1,n2,min_cls_len)
  writer = Writer.remote(data_dir, n_tfrs_per_file)
  count = 0
  while count < n_datapoints:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = tmpdir + "/"
      jobs = []
      jobs_per_worker = int(np.ceil((np.minimum(abs(n_datapoints - count), batch_size))/num_threads))
      for _ in range(num_threads):
        worker = Worker.remote(writer, get_fmla, tmpdir)
        # jobs = jobs + [worker.remote(writer, get_fmla, tmpdir) for _ in range(jobs_per_worker)]
        jobs = jobs + [worker.main.remote() for _ in range(jobs_per_worker)]
      ray.get(jobs)
      ray.get(writer.write_log.remote())
      count = ray.get(writer.count.remote())
      print("count: ", count)
  ray.get(writer.finalize.remote())
  print("done")

def gen_src(n1=20, n2=100, min_cls_len=2, data_dir = os.path.join(PROJECT_DIR, "train_data", "src", "train"), n_tfrs_per_file=5000, n_datapoints = 195000, batch_size = 500, num_threads = 4):
  check_make_path(data_dir)  
  get_fmla = lambda: get_unsat_src(n1,n2,min_cls_len)
  writer = Writer.remote(data_dir, n_tfrs_per_file)
  count = 0
  while count < n_datapoints:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = tmpdir + "/"
      jobs = []
      jobs_per_worker = int(np.ceil((np.minimum(abs(n_datapoints - count), batch_size))/num_threads))
      for _ in range(num_threads):
        worker = Worker.remote(writer, get_fmla, tmpdir)
        # jobs = jobs + [worker.remote(writer, get_fmla, tmpdir) for _ in range(jobs_per_worker)]
        jobs = jobs + [worker.main.remote() for _ in range(jobs_per_worker)]
      ray.get(jobs)
      ray.get(writer.write_log.remote())
      count = ray.get(writer.count.remote())
      print("count: ", count)
  ray.get(writer.finalize.remote())
  print("done")

def ray_gen_argparse():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--sr", action="store_true", dest="sr")
  parser.add_argument("--src", action="store_true", dest="src")
  parser.add_argument("--randkcnf", action="store_true", dest="randkcnf")
  parser.add_argument("--ramsey", action="store_true", dest="ramsey")
  parser.add_argument("--datapoints", action="store", dest="n_datapoints", type=int)
  parser.add_argument("--test", action="store_true", dest="test")
  parser.add_argument("--threads", action="store", dest="num_threads", type=int)
  parser.add_argument("--batch-size", action="store", dest="batch_size", type=int, default=80)
  return parser.parse_args()

if __name__ == "__main__":
  opts = ray_gen_argparse()
  assert exactly_one([opts.sr, opts.src, opts.randkcnf, opts.ramsey, opts.test])
  ray.init()
  if opts.test:
    gen_sr(n_datapoints=300, num_threads=3, data_dir = os.path.join(PROJECT_DIR, "train_data_test/sr/train/"), batch_size=opts.batch_size)
  elif opts.sr:
    gen_sr(n_datapoints=opts.n_datapoints, num_threads=opts.num_threads, batch_size=opts.batch_size)
  elif opts.src:
    gen_src(n_datapoints=opts.n_datapoints, num_threads=opts.num_threads, batch_size=opts.batch_size)
  elif opts.ramsey:
    gen_ramsey(n_datapoints=opts.n_datapoints, num_threads=opts.num_threads, batch_size=opts.batch_size)
  elif opts.randkcnf:
    gen_randkcnf(n_datapoints=opts.n_datapoints, num_threads=opts.num_threads, batch_size=opts.batch_size)
