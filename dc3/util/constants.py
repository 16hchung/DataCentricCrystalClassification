from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from collections import namedtuple
import numpy as np

from .util import Lattice, range_list_max, n_neighs_from_lattices

### FILE I/O

CFG_RT = Path('config')
PERF_DUMP_RT = CFG_RT / 'perf_dump'

DFLT_OUTPUT_RT = Path('default_pipeline')
CACHE_RT = Path('.dc3_cache')
EVAL_CACHE_RT = CACHE_RT / 'eval'
EVAL_CACHE_X = EVAL_CACHE_RT / 'X.npy'
EVAL_CACHE_y = EVAL_CACHE_RT / 'y.npy'

OV_OUTPUT_FMT = 'lammps/dump'
OV_CART_COLS = ['Position.X', 'Position.Y', 'Position.Z']

### TRAINING

DFLT_LATTICES = [
  Lattice(name='fcc', perfect_path=str(PERF_DUMP_RT/'dump_fcc_perfect_0.dat'), neigh_range=[0,12]),
  Lattice(name='bcc', perfect_path=str(PERF_DUMP_RT/'dump_bcc_perfect_0.dat'), neigh_range=[0,12]),
  Lattice(name='hcp', perfect_path=str(PERF_DUMP_RT/'dump_hcp_perfect_0.dat'), neigh_range=[0,8 ]),
  Lattice(name='cd' , perfect_path=str(PERF_DUMP_RT/'dump_cd_perfect_0.dat' ), neigh_range=[0,16]),
  Lattice(name='hd' , perfect_path=str(PERF_DUMP_RT/'dump_hd_perfect_0.dat' ), neigh_range=[0,16]),
  Lattice(name='sc' , perfect_path=str(PERF_DUMP_RT/'dump_sc_perfect_0.dat' ), neigh_range=[0,6 ])
]

DFLT_DISTORT_BINS = list(np.linspace(.01, .35, num=15))

# Feature computation

# TODO vvv maybe don't need this
DFLT_N_NEIGHS = n_neighs_from_lattices(DFLT_LATTICES)

MAX_NEIGH = range_list_max(DFLT_N_NEIGHS)

STEIN_MIN_NEIGH = 2
STEIN_NUM_lS = 10

SHELL_RANGE = False

N_RSF_PER_MU = 7 # DEPRECATED
RSF_MU_STEP = .05
RSF_SIGMA_SCALE = .05
RSF_USE_NEIGH_RANGE = True

FEATURE_PRECISION = 11

# Classifier
DFLT_CLF_TYPE = SVC
NN_CLF_TYPE = MLPClassifier
DFLT_CLF_KWARGS = {'C': 10,
                   'gamma': .01,
                   'max_iter': 1e5,
                   'tol': 1e-3,
                   'cache_size': 1000,
                   'class_weight': 'balanced',
                   'verbose': 2}

GS_SCORING = 'f1_weighted'
GS_NJOBS = -1
GS_VERBOSITY = 2

DFLT_OUTLIER_PCUT = 95
