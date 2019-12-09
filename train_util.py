import collections
import json

# hyperparameters + checkpoint dir for getting latest checkpoint
ModelCfg = collections.namedtuple(
  "ModelCfg",
  [ "model_id",
    "mode",
    "ckpt_dir",
    "d",
    "C_res_depth",
    "C_update_depth",
    "L_update_depth",
    "V_proof_depth",
    "V_core_depth",
    "C_core_depth",
    "n_rounds",
    "tau",
    "weight_reparam",
    "norm_axis",
    "norm_eps",
    "res_layers",
    "activation",
    "LC_scale",
    "CL_scale"
  ])

# configuration for training. might want to tweak these (learning rate, etc) while training the same model
TrainCfg = collections.namedtuple(
  "TrainCfg",
  [
    "model_cfg", # stored as model_cfg_path, needs to be reconstituted as part of constructing a TrainCfg
    # perhaps not the best API
    "data_dir",
    "num_samples",
    "batch_size",
    "ckpt_freq",
    "num_epochs",
    "num_steps_per_epoch",
    "n_tfrs_per_file",
    "clipvalue",
    "clipnorm",
    "learning_rate",
    "pv_loss_scale",    
    "cv_loss_scale",
    "cc_loss_scale",
    "l2_loss_scale",
    "n_parallel_reads",
    "n_parallel_calls",
    "n_prefetch",
    "xla_jit",
    "n_steps_per_log",
    "p_log",
  ]
)

def ModelCfg_of_dict(d):
    if not "C_res_depth" in d.keys():
        d["C_res_depth"] = 1
    if not "mode" in d.keys():
        d["mode"] = "res"
    return ModelCfg(**d)

def TrainCfg_of_dict(d):
  return TrainCfg(**d)

def ModelCfg_from_file(path):
    with open(path,"r") as f:
        cfg_dict = json.load(f)
    return ModelCfg_of_dict(cfg_dict)

def TrainCfg_from_file(path):
  with open(path, "r") as f:
    cfg_dict = json.load(f)
  model_cfg = ModelCfg_from_file(cfg_dict.pop("model_cfg_path"))
  return TrainCfg(model_cfg = model_cfg, **cfg_dict)

