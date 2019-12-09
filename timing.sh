#!/bin/bash

python3 cubing_eval.py ramsey z3 --n-vars=1 --cpus=2 --data-dir="cnf_data/ramsey/test/"
python3 cubing_eval.py ramsey random --n-vars=1 --cpus=2 --data-dir="cnf_data/ramsey/test/"
python3 cubing_eval.py ramsey res_models/res_grid_2_sr.json --n-vars=1 --cpus=2 --data-dir="cnf_data/ramsey/test/" --drat-head
python3 cubing_eval.py ramsey res_models/res_grid_2_sr.json --n-vars=1 --cpus=2 --data-dir="cnf_data/ramsey/test/"
python3 cubing_eval.py ramsey res_models/res_grid_2_src.json --n-vars=1 --cpus=2 --data-dir="cnf_data/ramsey/test/" --drat-head
python3 cubing_eval.py ramsey res_models/res_grid_2_src.json --n-vars=1 --cpus=2 --data-dir="cnf_data/ramsey/test/"

python3 cubing_eval.py ramsey random --n-vars=3 --cpus=8 --data-dir="cnf_data/ramsey/test/"
python3 cubing_eval.py ramsey res_models/res_grid_2_sr.json --n-vars=3 --cpus=8 --data-dir="cnf_data/ramsey/test/" --drat-head
python3 cubing_eval.py ramsey res_models/res_grid_2_sr.json --n-vars=3 --cpus=8 --data-dir="cnf_data/ramsey/test/"
python3 cubing_eval.py ramsey res_models/res_grid_2_src.json --n-vars=3 --cpus=8 --data-dir="cnf_data/ramsey/test/" --drat-head
python3 cubing_eval.py ramsey res_models/res_grid_2_src.json --n-vars=3 --cpus=8 --data-dir="cnf_data/ramsey/test/"

python3 cubing_eval.py schur z3 --n-vars=1 --cpus=2 --data-dir="cnf_data/schur/test/"
python3 cubing_eval.py schur random --n-vars=1 --cpus=2 --data-dir="cnf_data/schur/test/"
python3 cubing_eval.py schur res_models/res_grid_2_sr.json --n-vars=1 --cpus=2 --data-dir="cnf_data/schur/test/" --drat-head
python3 cubing_eval.py schur res_models/res_grid_2_sr.json --n-vars=1 --cpus=2 --data-dir="cnf_data/schur/test/"
python3 cubing_eval.py schur res_models/res_grid_2_src.json --n-vars=1 --cpus=2 --data-dir="cnf_data/schur/test/" --drat-head
python3 cubing_eval.py schur res_models/res_grid_2_src.json --n-vars=1 --cpus=2 --data-dir="cnf_data/schur/test/"

python3 cubing_eval.py schur random --n-vars=3 --cpus=8
python3 cubing_eval.py schur res_models/res_grid_2_sr.json --n-vars=3 --cpus=8 --data-dir="cnf_data/schur/test/" --drat-head
python3 cubing_eval.py schur res_models/res_grid_2_sr.json --n-vars=3 --cpus=8 --data-dir="cnf_data/schur/test/"
python3 cubing_eval.py schur res_models/res_grid_2_src.json --n-vars=3 --cpus=8 --data-dir="cnf_data/schur/test/" --drat-head
python3 cubing_eval.py schur res_models/res_grid_2_src.json --n-vars=3 --cpus=8 --data-dir="cnf_data/schur/test/"

python3 cubing_eval.py vdw z3 --n-vars=1 --cpus=2 --data-dir="cnf_data/vdw/test/"
python3 cubing_eval.py vdw random --n-vars=1 --cpus=2 --data-dir="cnf_data/vdw/test/"
python3 cubing_eval.py vdw res_models/res_grid_2_sr.json --n-vars=1 --cpus=2 --data-dir="cnf_data/vdw/test/" --drat-head
python3 cubing_eval.py vdw res_models/res_grid_2_sr.json --n-vars=1 --cpus=2 --data-dir="cnf_data/vdw/test/"
python3 cubing_eval.py vdw res_models/res_grid_2_src.json --n-vars=1 --cpus=2 --data-dir="cnf_data/vdw/test/" --drat-head
python3 cubing_eval.py vdw res_models/res_grid_2_src.json --n-vars=1 --cpus=2 --data-dir="cnf_data/vdw/test/"

python3 cubing_eval.py vdw random --n-vars=3 --cpus=8
python3 cubing_eval.py vdw res_models/res_grid_2_sr.json --n-vars=3 --cpus=8 --data-dir="cnf_data/vdw/test/" --drat-head
python3 cubing_eval.py vdw res_models/res_grid_2_sr.json --n-vars=3 --cpus=8 --data-dir="cnf_data/vdw/test/"
python3 cubing_eval.py vdw res_models/res_grid_2_src.json --n-vars=3 --cpus=8 --data-dir="cnf_data/vdw/test/" --drat-head
python3 cubing_eval.py vdw res_models/res_grid_2_src.json --n-vars=3 --cpus=8 --data-dir="cnf_data/vdw/test/"
