from sklearn.metrics.pairwise import cosine_distances
import pickle as pk
import numpy as np
import numpy.linalg

from ..util import constants as C

class OutlierDetector:
  outlier_lbl = -1

  def __init__(self, percentile_cut=C.DFLT_OUTLIER_PCUT,
                     perfect_features=None,
                     label_to_cutoffs_dict=None):
    self.percentile_cut = percentile_cut
    self.perf_xs = perfect_features
    self.lbl_to_cutoff = label_to_cutoffs_dict

  def fit(self, X, y, perf_xs):
    self.perf_xs = perf_xs
    self.lbl_to_cutoff = {}
    # iterate over lattices
    for lbl, perf_x in enumerate(perf_xs):
      # grab features only from this class
      X_latt = X[y==lbl][:]
      D = self._distance(X_latt, perf_x)
      self.lbl_to_cutoff[lbl] = np.percentile(D, self.percentile_cut)
    return self

  def predict(self, X, y):
    for lbl, perf_x in enumerate(self.perf_xs):
      cutoff = self.lbl_to_cutoff[lbl]
      # extract this lattice's labels and features
      y_latt = y[y==lbl]
      if not len(y_latt): continue
      X_latt = X[y==lbl][:]
      import pdb;pdb.set_trace() # may need to check dims
      # find distances to this lattice's perfect features and det if below cutoff
      D = self._distance(X_latt, perf_x)
      y_latt[D > cutoff] = self.outlier_lbl
      # update labels in og np y array
      y[y==lbl] = y_latt
    return y

  def save(self, save_path):
    with open(save_path, 'wb') as f:
      pk.dump({'percentile_cut': self.percentile_cut,
               'perfect_features': self.perf_xs,
               'label_to_cutoffs_dict': self.lbl_to_cutoff
              }, f)

  @classmethod
  def from_saved_path(cls, save_path):
    with open(save_path, 'rb') as f:
      kwargs = pk.load(f)
    return cls(**kwargs)

  @staticmethod
  def _distance(X, perf_x):
    D = np.linalg.norm(X - perf_x, axis=-1) * \
        cosine_distances(X, perf_x[np.newaxis, :])[:,0]
    return D
