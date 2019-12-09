#!/bin/bash

python3 train_neurocuber.py  --train-cfg=res_models/train_res_grid_2_sr.json
python3 train_neurocuber.py  --train-cfg=res_models/train_res_grid_2_src.json
