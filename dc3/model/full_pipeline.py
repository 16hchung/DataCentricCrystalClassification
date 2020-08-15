from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from pathlib import Path
import numpy as np
import joblib

from ..util import constants as C
from ..util.util import n_neighs_from_lattices
from ..util.features import SteinhardtNelsonConfig, RSFConfig, FeatureComputer
from ..data.synthetic import distort_perfect
from .outlier_detector import OutlierDetector

class DC3Pipeline:
  def __init__(self, lattices=C.DFLT_LATTICES,
                     feature_computer=feature_computer,
                     overwrite=False,
                     scaler=StandardScaler(),
                     classifier=SVC(**C.DFLT_CLF_KWARGS),
                     outlier_detector=OutlierDetector(),
                     output_rt=C.DFLT_OUTPUT_RT): # TODO implement model saving
    self.overwrite = overwrite
    self._make_paths(output_rt)
    
    # TODO include config cache
    self.lattices         = lattices
    self.feature_computer = feature_computer
    self.scaler           = scaler
    self.outlier_detector = outlier_detector
    self.classifier       = classifier

  @staticmethod
  def from_pipeline_rt(output_rt=C.DFLT_OUTPUT_RT):
    import pdb;pdb.set_trace()
    raise NotImplementedError

  def _make_paths(self, output_rt):
    output_rt = Path(output_rt)
    output_rt.mkdir(exist_ok=True)
    # settings for this model
    self.cfg_path = output_rt / 'config.json'
    # train-related paths
    self.train_rt = output_rt / 'train'
    self.train_rt.mkdir(exist_ok=True) # TODO incorporate self.overwrite
    self.synth_dump_path = self.train_rt / 'dump'
    self.synth_feat_path = self.train_rt / 'synth_features.npy'
    self.synth_lbls_path = self.train_rt / 'synth_labels.npy'
    self.perf_feat_path  = self.train_rt / 'perfect_features.npz'
    self.weights_path    = self.train_rt / 'weights'
    self.scaler_path     = self.weights_path / 'scaler.joblib'
    self.classifier_path = self.weights_path / 'svc.joblib'
    self.outlier_path    = self.weights_path / 'outlier_detector.pkl'
    # inference-related paths
    self.inf_rt = output_rt / 'inference'
    # evaluation-related output
    self.eval_rt = output_rt / 'eval'

  def compute_synth_features(self, distort_bins=C.DFLT_DISTORT_BINS):
    self.synth_dump_path.mkdir(exist_ok=self.overwrite)
    # get synthetic training data
    Xs, ys = [], []
    for label, latt in enumerate(self.lattices):
      print(latt)
      # get distorted cartesian coords
      latt_dump_path = self.synth_dump_path / latt.name
      latt_dump_path.mkdir(exist_ok=self.overwrite)
      ov_collections = distort_perfect(latt.perfect_path,
                                       distort_bins=distort_bins,
                                       save_path=latt_dump_path)
      X_latt_list = [self.feature_computer.compute(ov_collection)
                     for ov_collection in tqdm(ov_collections)]
      X_latt = np.concatenate(X_latt_list, axis=0)
      y_latt = np.array( [float(label)] * len(X_latt) )

      Xs.append(X_latt)
      ys.append(y_latt)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.save(self.synth_feat_path, X)
    np.save(self.synth_lbls_path, y)
    return X, y

  def compute_perf_features(self):
    perf_xs = []
    for lbl, latt in enumerate(self.lattices):
      perf_x = self.feature_computer.compute_perf_from_dump(latt.perfect_path)
      perf_xs.append(perf_x)
    np.savez(self.perf_feat_path, *perf_xs)
    return perf_xs

  def load_perf_features(self):
    npzfile = np.load(self.perf_feat_path)
    perf_xs = [npzfile[key] for key in npzfile.files]
    return perf_xs

  def fit(self, X, y, perf_xs):
    self.weights_path.mkdir(exist_ok=self.overwrite)
    # fit scaler to training data
    self.scaler.fit(X)
    joblib.dump(self.scaler, self.scaler_path)
    X = self.scaler.transform(X)
    # train classifier
    X, y = shuffle(X, y)
    X_train, y_train = X[:50000,:], y[:50000]
    self.classifier.fit(X_train, y_train)
    joblib.dump(self.classifier, self.classifier_path)
    # TODO add hparam grid searching
    # train outlier detector
    perf_xs = [np.array(x) for x in self.scaler.transform(perf_xs).tolist()]
    self.outlier_detector.fit(X, y, perf_xs)
    self.outlier_detector.save(self.outlier_path)
    return self

  def fit_end2end(self, **kwargs):
    if not self.overwrite and self.synth_feat_path.exists() \
                          and self.synth_lbls_path.exists():
      X = np.load(self.synth_feat_path)
      y = np.load(self.synth_lbls_path)
    else:
      X, y = self.compute_synth_features(**kwargs)

    if not self.overwrite and self.perf_feat_path.exists():
      perf_xs = self.load_perf_features()
    else:
      perf_xs = self.compute_perf_features()

    if not self.overwrite and False:
      raise NotImplementedError # TODO load cached models
    else:
      self.fit(X, y, perf_xs)
    return self

  def predict(self, ov_data_collection):
    X = self.feature_computer.compute(ov_data_collection)
    X = self.scaler.transform(X)
    y_cand = self.classifier.predict(X)
    y = self.outlier_detector(X, y_cand)
    return y

  def eval(self, ov_data_collection, labels):
    raise NotImplementedError
