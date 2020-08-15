from sklearn.svm import SVC
from pathlib import Path
from collections import namedtuple
import numpy as np

from .util import Lattice, range_list_max, n_neighs_from_lattices

### FILE I/O

CFG_RT = Path('config')
EVAL_INPT_RT = Path('eval') # TODO make subpaths for benchmarks.csv, val_data
PERF_DUMP_RT = CFG_RT / 'perf_dump'

DFLT_OUTPUT_RT = Path('default_pipeline')

OV_OUTPUT_FMT = 'lammps/dump'
OV_CART_COLS = ['Position.X', 'Position.Y', 'Position.Z']

### TRAINING

DFLT_LATTICES = [
  Lattice(name='fcc', perfect_path=PERF_DUMP_RT/'dump_fcc_perfect_0.dat', neigh_range=(0,12)),
  Lattice(name='bcc', perfect_path=PERF_DUMP_RT/'dump_bcc_perfect_0.dat', neigh_range=(0,12)),
  Lattice(name='hcp', perfect_path=PERF_DUMP_RT/'dump_hcp_perfect_0.dat', neigh_range=(0,8) ),
  Lattice(name='cd' , perfect_path=PERF_DUMP_RT/'dump_cd_perfect_0.dat' , neigh_range=(0,16)),
  Lattice(name='hd' , perfect_path=PERF_DUMP_RT/'dump_hd_perfect_0.dat' , neigh_range=(0,16)),
  Lattice(name='sc' , perfect_path=PERF_DUMP_RT/'dump_sc_perfect_0.dat' , neigh_range=(0,6) )
]

DFLT_DISTORT_BINS = list(np.linspace(.01, .35, num=15))

# Feature computation

# TODO vvv maybe don't need this
DFLT_N_NEIGHS = n_neighs_from_lattices(DFLT_LATTICES)

MAX_NEIGH = range_list_max(DFLT_N_NEIGHS)

STEIN_MIN_NEIGH = 2
STEIN_NUM_lS = 10

N_RSF_PER_MU = 7
RSF_MU_STEP = .05
RSF_SIGMA_SCALE = .05

# Classifier
DFLT_CLF_KWARGS = {'C': 10,
                   'gamma': .01,
                   'max_iter': 1e5,
                   'tol': 1e-3,
                   'cache_size': 1000,
                   'class_weight': 'balanced'}

DFLT_OUTLIER_PCUT = 95
