from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from pathlib import Path
import numpy as np
import warnings
import joblib

from ovito.io import import_file, export_file

from ..util import constants as C
from ..util.util import n_neighs_from_lattices
from ..util.features import Featurizer
from ..data.synthetic import distort_perfect
from ..data.file_io import recursive_in_out_file_pairs
from .outlier_detector import OutlierDetector

class DC3Pipeline:
  def __init__(self, lattices=C.DFLT_LATTICES,
                     featurizer=Featurizer(),
                     classifier=SVC(**C.DFLT_CLF_KWARGS),
                     outlier_detector=OutlierDetector(),
                     output_rt=C.DFLT_OUTPUT_RT,
                     overwrite=False):
    self._make_paths(output_rt)
    self._init_config_and_featurizer(lattices, featurizer, overwrite)
    self._init_models(classifier, outlier_detector, overwrite)
    
  def _init_config_and_featurizer(self, lattices, featurizer, overwrite):
    if self.cfg_path.exists() and not overwrite:
      self.featurizer = Featurizer.from_saved_path(self.cfg_path)
      self.lattices   = self.featurizer.lattices
      assert not lattices or self.lattices == lattices
      assert not featurizer or self.featurizer.__dict__ == featurizer.__dict__
    else:
      assert isinstance(lattices, list) and isinstance(featurizer, Featurizer)
      self.lattices         = lattices
      self.featurizer       = featurizer
      self.featurizer.save(self.cfg_path)
    self.name_to_lbl_latt = {l.name: (i,l) for i,l in enumerate(self.lattices)}
    
  def _init_models(self, classifier, outlier_detector, overwrite):
    if self.scaler_path.exists() and self.outlier_path.exists() \
                                 and self.classifier_path.exists() \
                                 and not overwrite:
      if classifier != None or outlier_detector != None:
        warnings.warn(
          'Using cached models. ' + 
          'Ignoring classifier and outlier_detector passed into DC3Pipeline constructor'
        )
      self.scaler = joblib.load(self.scaler_path)
      self.classifier = joblib.load(self.classifier_path)
      self.outlier_detector = OutlierDetector.from_saved_path(self.outlier_path)
      self.is_trained = True
    else:
      assert isinstance(outlier_detector, OutlierDetector) and \
             isinstance(classifier, SVC)
      self.scaler           = StandardScaler()
      self.outlier_detector = outlier_detector
      self.classifier       = classifier
      self.is_trained = False

  def _make_paths(self, output_rt):
    output_rt = Path(output_rt)
    output_rt.mkdir(exist_ok=True)
    # settings for this model
    self.cfg_path = output_rt / 'config.json'
    # train-related paths
    self.train_rt = output_rt / 'train'
    self.train_rt.mkdir(exist_ok=True)
    self.synth_dump_path = self.train_rt / 'dump'
    self.synth_feat_path = self.train_rt / 'synth_features.npy'
    self.synth_lbls_path = self.train_rt / 'synth_labels.npy'
    self.perf_feat_path  = self.train_rt / 'perfect_features.npz'
    self.weights_path    = self.train_rt / 'weights'
    self.scaler_path     = self.weights_path / 'scaler.joblib'
    self.classifier_path = self.weights_path / 'svc.joblib'
    self.outlier_path    = self.weights_path / 'outlier_detector.pkl'
    # inference-related paths
    self.inference_rt = output_rt / 'inference'
    # evaluation-related output
    self.eval_rt = output_rt / 'eval'

  def compute_synth_features(self, distort_bins=C.DFLT_DISTORT_BINS,
                                   overwrite=False):
    self.synth_dump_path.mkdir(exist_ok=overwrite)
    # get synthetic training data
    Xs, ys = [], []
    for label, latt in enumerate(self.lattices):
      print(latt)
      # get distorted cartesian coords
      latt_dump_path = self.synth_dump_path / latt.name
      latt_dump_path.mkdir(exist_ok=overwrite)
      ov_collections = distort_perfect(latt.perfect_path,
                                       distort_bins=distort_bins,
                                       save_path=latt_dump_path)
      X_latt_list = [self.featurizer.compute(ov_collection)
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
      perf_x = self.featurizer.compute_perf_from_dump(latt.perfect_path)
      perf_xs.append(perf_x)
    np.savez(self.perf_feat_path, *perf_xs)
    return perf_xs

  def load_perf_features(self):
    npzfile = np.load(self.perf_feat_path)
    perf_xs = [npzfile[key] for key in npzfile.files]
    return perf_xs

  def fit(self, X, y, perf_xs, overwrite=False):
    self.weights_path.mkdir(exist_ok=overwrite)
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
    self.is_trained = False
    return self

  def fit_end2end(self, distort_bins=C.DFLT_DISTORT_BINS,
                        overwrite=False):
    if not overwrite and self.synth_feat_path.exists() \
                          and self.synth_lbls_path.exists():
      X = np.load(self.synth_feat_path)
      y = np.load(self.synth_lbls_path)
    else:
      X, y = self.compute_synth_features(distort_bins=distort_bins)
    if not overwrite and self.perf_feat_path.exists():
      perf_xs = self.load_perf_features()
    else:
      perf_xs = self.compute_perf_features()

    if not overwrite and False:
      raise NotImplementedError # TODO load cached models
    else:
      self.fit(X, y, perf_xs)
    return self

  def predict(self, ov_data_collection):
    return self.predict_return_features(ov_data_collection)[1]

  def predict_return_features(self, ov_data_collection):
    assert self.is_trained
    X = self.featurizer.compute(ov_data_collection)
    X = self.scaler.transform(X)
    y_cand = self.classifier.predict(X)
    y = self.outlier_detector.predict(X, y_cand)
    return X, y

  def predict_recursive_dir(self, input_dir, output_name, ext='.gz'):
    output_dir = self.inference_rt / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for in_path, out_path in tqdm(recursive_in_out_file_pairs(input_dir,
                                                              output_dir,
                                                              ext=ext)):
      ov_data = import_file(in_path).compute()
      y = self.predict(ov_data)
      ov_data.particles_.create_property('Lattice', data=y)
      export_file(ov_data,
                  out_path,
                  'lammps/dump',
                  columns=['Position.X', 'Position.Y', 'Position.Z', 'Lattice'])
