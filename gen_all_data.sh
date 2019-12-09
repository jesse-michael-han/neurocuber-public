#!/bin/bash

python3 gen_labelled_data.py --test --threads=4 --datapoints=300 --batch-size=32
python3 gen_labelled_data.py --sr --threads=8 --datapoints=250000 --batch-size=80
python3 gen_labelled_data.py --src --threads=8 --datapoints=250000 --batch-size=80
python3 gen_cnf.py
