import os
import sys

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "train_data") # labelled data
CNF_DIR = os.path.join(PROJECT_DIR, "cnf_data") # unlabelled CNF files for evaluation
TOOLS_DIR = os.path.join(PROJECT_DIR, "tools")
CADICAL_PATH = "cadical"
