import numpy as np

from ..util import constants as C
from ..util.features import SteinhardtNelsonConfig, RSFConfig, FeatureComputer

class DC3Pipeline:
  def __init__(self, lattices=C.DFLT_LATTICES,
                     stein_config=SteinhardtNelsonConfig(),
                     rsf_config=RSFConfig()):
    self.lattices     = lattices
    self.n_neighs     = list(set([l.neigh_range for l in lattices]))
    self.stein_config = stein_config
    self.rsf_config   = rsf_config # TODO maybe don't need
    self.feature_computer = FeatureComputer(self.n_neighs,
                                            self.stein_config,
                                            self.rsf_config)
    # TODO vvv
    self.outlier_detector = None
    self.classifier       = None
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def predict(self, ov_data_collection):
    raise NotImplementedError

  def eval(self, ov_data_collection, labels):
    raise NotImplementedError

