#!/bin/bash

# run this in a virtual environment with tensorflow>=2.0

# generate data
bash gen_all_data.sh

# train models
bash train.sh

# timing evaluation
bash timing.sh

# playout evaluation
bash random_playout.sh
