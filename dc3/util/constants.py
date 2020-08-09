from pathlib import Path

from .util import range_list_max

### FILEPATHS

### FEATURE COMPUTATION

DFLT_N_NEIGHS = [
    range(6),
    range(8),
    range(12),
    range(4,16)
]
MAX_NEIGH = range_list_max(DFLT_N_NEIGHS)

STEIN_MIN_NEIGH = 2
STEIN_NUM_lS = 10

N_RSF_PER_MU = 7
RSF_MU_STEP = .05
RSF_SIGMA_SCALE = .05
