#!/bin/bash

python playout_eval.py ramsey z3 --cpus=16 --matches=50 --data="cnf_data/ramsey/test/"
python playout_eval.py ramsey random --cpus=16 --matches=50 --data="cnf_data/ramsey/test"
python playout_eval.py ramsey drat --model=res_models/res_grid_2_sr.json --cpus=16 --matches=50 --data="cnf_data/ramsey/test"
python playout_eval.py ramsey core --model=res_models/res_grid_2_sr.json --cpus=16 --matches=50 --data="cnf_data/ramsey/test"
python playout_eval.py ramsey drat --model=res_models/res_grid_2_src.json --cpus=16 --matches=50 --data="cnf_data/ramsey/test"
python playout_eval.py ramsey core --model=res_models/res_grid_2_src.json --cpus=16 --matches=50 --data="cnf_data/ramsey/test"

python playout_eval.py schur z3 --cpus=16 --matches=50 --data="cnf_data/schur/test/"
python playout_eval.py schur random --cpus=16 --matches=50 --data="cnf_data/schur/test"
python playout_eval.py schur drat --model=res_models/res_grid_2_sr.json --cpus=16 --matches=50 --data="cnf_data/schur/test"
python playout_eval.py schur core --model=res_models/res_grid_2_sr.json --cpus=16 --matches=50 --data="cnf_data/schur/test"
python playout_eval.py schur drat --model=res_models/res_grid_2_src.json --cpus=16 --matches=50 --data="cnf_data/schur/test"
python playout_eval.py schur core --model=res_models/res_grid_2_src.json --cpus=16 --matches=50 --data="cnf_data/schur/test"

python playout_eval.py vdw z3 --cpus=16 --matches=50 --data="cnf_data/vdw/test/"
python playout_eval.py vdw random --cpus=16 --matches=50 --data="cnf_data/vdw/test"
python playout_eval.py vdw drat --model=res_models/res_grid_2_sr.json --cpus=16 --matches=50 --data="cnf_data/vdw/test"
python playout_eval.py vdw core --model=res_models/res_grid_2_sr.json --cpus=16 --matches=50 --data="cnf_data/vdw/test"
python playout_eval.py vdw drat --model=res_models/res_grid_2_src.json --cpus=16 --matches=50 --data="cnf_data/vdw/test"
python playout_eval.py vdw core --model=res_models/res_grid_2_src.json --cpus=16 --matches=50 --data="cnf_data/vdw/test"
