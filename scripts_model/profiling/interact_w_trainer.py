## Profiling Packages
import cProfile
import pstats
import snakeviz
import pickle as pkl

import os, sys
print(os.getcwd())
sys.path.append(os.getcwd()+"/scripts_model")

print(sys.path)

## Setup code from 2_2 cluster simulation
from setup_simu2_2 import *