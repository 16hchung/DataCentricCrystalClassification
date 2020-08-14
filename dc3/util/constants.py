from pathlib import Path
from collections import namedtuple
import numpy as np

from .util import range_list_max

### FILEPATHS

CFG_RT = Path('config')
PERF_OV_RT = CFG_RT / 'ovito_perf'

OUTPUT_RT = Path('output')
TRAIN_RT = OUTPUT_RT  / 'training'
SYNTHETIC_RT = OUTPUT_RT / 'synth_features'
PERF_FEAT_RT = OUTPUT_RT / 'perf_features'


### TRAINING

Lattice = namedtuple('Lattice', ['name', 'perfect_path', 'neigh_range'])
DFLT_LATTICES = [
  Lattice(name='fcc', perfect_path=Path(TODO), neigh_range=(0,12)),
  Lattice(name='bcc', perfect_path=Path(TODO), neigh_range=(0,12)),
  Lattice(name='hcp', perfect_path=Path(TODO), neigh_range=(0,8) ),
  Lattice(name='cd' , perfect_path=Path(TODO), neigh_range=(0,16)),
  Lattice(name='hd' , perfect_path=Path(TODO), neigh_range=(0,16)),
  Lattice(name='sc' , perfect_path=Path(TODO), neigh_range=(0,6) )
]

DFLT_DISTORT_BINS = list(np.linspace(.01, .35, num=15))

# Feature computation

# TODO vvv maybe don't need this
DFLT_N_NEIGHS = list(set([l.neigh_range for l in DFLT_LATTICES]))
DFLT_N_NEIGHS.sort()

MAX_NEIGH = range_list_max(DFLT_N_NEIGHS)

STEIN_MIN_NEIGH = 2
STEIN_NUM_lS = 10

N_RSF_PER_MU = 7
RSF_MU_STEP = .05
RSF_SIGMA_SCALE = .05
