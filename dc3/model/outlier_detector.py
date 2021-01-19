from sklearn.metrics.pairwise import cosine_distances
import pickle as pk
import numpy as np
import numpy.linalg

from ..util import constants as C
from ..util.util import get_optimal_cutoff

class OutlierDetector:
  outlier_lbl = -1

  def __init__(self, max_neigh=C.DFLT_MAX_NEIGH,
                     percentile_cut=C.DFLT_OUTLIER_PCUT,
                     perfect_features=None,
                     label_to_cutoffs_dict=None,
                     alpha_cutoff=None,
                     amorph_sim_cut=C.DFLT_AMORPH_SIM_CUT):
    self.percentile_cut = percentile_cut
    self.perf_xs = perfect_features
    self.lbl_to_cutoff = label_to_cutoffs_dict
    self.amorph_sim_cut = amorph_sim_cut
    self.max_neigh = max_neigh
    self.alpha_cutoff = alpha_cutoff

  def fit(self, X, y, perf_xs, alpha, liq_alpha):
    self.perf_xs = perf_xs
    self.lbl_to_cutoff = {}
    # iterate over lattices
    for lbl, perf_x in enumerate(perf_xs):
      # grab features only from this class
      X_latt = X[y==lbl][:]
      D = self._distance(X_latt, perf_x)
      self.lbl_to_cutoff[lbl] = np.percentile(D, self.percentile_cut)
    self.alpha_cutoff = get_optimal_cutoff(liq_alpha, alpha)
    return self

  def predict(self, X, alpha, y, data_collection):
    # first find amorphous outliers
    y[alpha < self.alpha_cutoff] = self.outlier_lbl
    # now find crystal outliers (but not included in this model's classes)
    for lbl, perf_x in enumerate(self.perf_xs):
      cutoff = self.lbl_to_cutoff[lbl]
      # extract this lattice's labels and features
      y_latt = y[(y==lbl) & (y!=self.outlier_lbl)]
      if not len(y_latt): continue
      X_latt = X[(y==lbl) & (y!=self.outlier_lbl)][:]
      # find distances to this lattice's perfect features and det if below cutoff
      D = self._distance(X_latt, perf_x)
      y_latt[D > cutoff] = self.outlier_lbl
      # update labels in og np y array
      y[(y==lbl) & (y!=self.outlier_lbl)] = y_latt
    return y

  def save(self, save_path):
    with open(save_path, 'wb') as f:
      pk.dump({'percentile_cut': self.percentile_cut,
               'perfect_features': self.perf_xs,
               'label_to_cutoffs_dict': self.lbl_to_cutoff,
               'amorph_sim_cut': self.amorph_sim_cut,
               'max_neigh': self.max_neigh,
               'alpha_cutoff': self.alpha_cutoff
              }, f)

  @classmethod
  def from_saved_path(cls, save_path):
    with open(save_path, 'rb') as f:
      kwargs = pk.load(f)
    return cls(**kwargs)

  @staticmethod
  def _distance(X, perf_x):
    return np.linalg.norm(X - perf_x, axis=-1)
    #D = np.linalg.norm(X - perf_x, axis=-1) * \
    #    cosine_distances(X, perf_x[np.newaxis, :])[:,0]
    #return D

  @staticmethod
  def _datapoint_wise_cos_sim(X1, X2):
    cos_sim = np.tensordot(X1, X2, axes=(1,1)) / \
              (np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1))
    return cos_sim
