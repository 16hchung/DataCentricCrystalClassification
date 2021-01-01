from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from pathlib import Path
import numpy as np
import pickle as pk
import warnings
import joblib
import gc
import json

from ovito.io import import_file, export_file

from ..util import constants as C
from ..util.util import n_neighs_from_lattices
from ..features import Featurizer
from ..data.synthetic import distort_perfect, make_synthetic_liq
from ..data.file_io import recursive_in_out_file_pairs
from .outlier_detector import OutlierDetector

class DC3Pipeline:
  def __init__(self, lattices=C.DFLT_LATTICES,
                     featurizer=Featurizer(),
                     clf_type=C.DFLT_CLF_TYPE,
                     clf_params=C.DFLT_CLF_KWARGS,
                     clf_param_options=None, # if not None, do grid search
                     outlier_detector=OutlierDetector(),
                     output_rt=C.DFLT_OUTPUT_RT,
                     synth_dump_path=None,
                     overwrite=False):
    self._make_paths(output_rt, clf_type)
    if synth_dump_path: self.synth_dump_path = Path(synth_dump_path)
    self._init_config_and_featurizer(lattices, featurizer, overwrite)
    self._init_models(clf_type, clf_params, clf_param_options, 
                      outlier_detector, overwrite)
    
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
    
  def _init_models(self, clf_type,
                         clf_params,
                         clf_param_options,
                         outlier_detector,
                         overwrite):
    # If not overwriting and pretrained models are in file sys, load them
    if self.scaler_path.exists() and self.outlier_path.exists() \
                                 and self.classifier_path.exists() \
                                 and not overwrite:
      if clf_params != None or outlier_detector != None:
        warnings.warn(
          'Using cached models. ' + 
          'Ignoring clf_params and outlier_detector passed into DC3Pipeline constructor'
        )
      self.scaler = joblib.load(self.scaler_path)
      self.classifier = joblib.load(self.classifier_path)
      self.outlier_detector = OutlierDetector.from_saved_path(self.outlier_path)
      self.is_trained = True
    # Otherwise, init models with default params
    else:
      assert isinstance(outlier_detector, OutlierDetector) and \
            ( isinstance(clf_params, dict) or \
              isinstance(clf_param_options, dict) or \
              self.clf_cfg_path.exists() )
      self.scaler           = StandardScaler()
      self.outlier_detector = outlier_detector
      if self.clf_cfg_path.exists():
        with open(str(self.clf_cfg_path)) as f: clf_params = json.load(f)
        if clf_params:
          warnings.warn(
            'Using cached hparams, ignoring clf_params passed into DC3Pipeline'
          )
      self.classifier = clf_type(**clf_params) if clf_params else None
      self.is_trained = False
    self._save_clf_hparams()
    self._clf_type = clf_type
    self._clf_params = clf_params
    self._clf_param_options = clf_param_options

  def _save_clf_hparams(self):
    if not self.classifier:
      warnings.warn('could notsave hparams for None classifier')
      return
    with open(str(self.clf_cfg_path), 'w') as f:
      json.dump(self.classifier.get_params(), f)

  def _make_paths(self, output_rt, clf_type):
    output_rt = Path(output_rt)
    output_rt.mkdir(exist_ok=True)
    # settings for this model
    self.cfg_path = output_rt / 'config.json'
    self.clf_cfg_path = output_rt / f'{clf_type.__name__}_config.json'
    # train-related paths
    self.train_rt = output_rt / 'train'
    self.train_rt.mkdir(exist_ok=True)
    self.synth_dump_path = self.train_rt / 'dump'
    self.synth_feat_path = self.train_rt / 'synth_features.npy'
    self.synth_alpha_path= self.train_rt / 'synth_alpha.npy'
    self.synth_lbls_path = self.train_rt / 'synth_labels.npy'
    self.liq_alpha_path  = self.train_rt / 'synth_liq_alpha.npy'
    self.perf_feat_path  = self.train_rt / 'perfect_features.npz'
    self.weights_path    = self.train_rt / 'weights'
    self.scaler_path     = self.weights_path / 'scaler.joblib'
    self.classifier_path = self.weights_path / f'{clf_type.__name__}.joblib'
    self.outlier_path    = self.weights_path / 'outlier_detector.pkl'
    # clf grid search related paths
    self.clf_gs_path     = self.train_rt / f'clf_gs_{clf_type.__name__}'
    self.clf_gs_cfg_path = self.clf_gs_path / 'param_options.json'
    self.clf_gs_res_path = self.clf_gs_path / 'results.pkl'
    # inference-related paths
    self.inference_rt = output_rt / 'inference'
    # evaluation-related output
    self.eval_rt = output_rt / 'eval'

  def compute_synth_features(self, distort_bins=C.DFLT_DISTORT_BINS,
                                   overwrite=False):
    self.synth_dump_path.mkdir(exist_ok=overwrite)
    # get synthetic training data
    Xs, ys, alphas, liq_alphas = [], [], [], []

    def get_X_alpha(ov_collections):
      all_features = [self.featurizer.compute(ov_collection)
                     for ov_collection in tqdm(ov_collections)]
      X_list, alpha_list = zip(*all_features)
      X = np.concatenate(X_list, axis=0)
      alpha = np.concatenate(alpha_list)
      return X, alpha

    for label, latt in enumerate(self.lattices):
      print(latt)
      # get distorted cartesian coords
      latt_dump_path = self.synth_dump_path / latt.name
      latt_dump_path.mkdir(exist_ok=overwrite)
      ov_collections = distort_perfect(latt.perfect_path,
                                       distort_bins=distort_bins,
                                       save_path=latt_dump_path)
      X_latt, alpha_latt = get_X_alpha(ov_collections)
      y_latt = np.array( [float(label)] * len(X_latt) )

      Xs.append(X_latt)
      ys.append(y_latt)
      alphas.append(alpha_latt)

      liq_collection = make_synthetic_liq(latt.perfect_path,
                                          save_path=latt_dump_path)
      _, alpha_liq = get_X_alpha([liq_collection])
      liq_alphas.append(alpha_liq)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    alpha = np.concatenate(alphas)
    liq_alpha = np.concatenate(liq_alphas)

    np.save(self.synth_feat_path, X)
    np.save(self.synth_lbls_path, y)
    np.save(self.synth_alpha_path, alpha)
    np.save(self.liq_alpha_path, liq_alpha)
    return X, y, alpha, liq_alpha

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

  def _gs_clf(self, X, y, overwrite):
    if not self._clf_param_options: return False # no listed paramas to try

    # path stuff
    self.clf_gs_path.mkdir(exist_ok=overwrite)
    with open(str(self.clf_gs_cfg_path), 'w') as f: 
      json.dump(self._clf_param_options, f)

    # init gs object
    gs = GridSearchCV(self._clf_type(),
                      self._clf_param_options,
                      scoring=C.GS_SCORING,
                      n_jobs=C.GS_NJOBS,
                      refit=True,
                      verbose=C.GS_VERBOSITY,
                      return_train_score=True)
    # run gs
    gs.fit(X, y)
    # set classifier to best model
    self.classifier = gs.best_estimator_
    # save results to disk
    with open(str(self.clf_gs_res_path), 'wb') as f: pk.dump(gs.cv_results_, f)
    self._save_clf_hparams()

  def _format_to_clf_lbls(self, y):
    if self._clf_type != C.NN_CLF_TYPE: return y
    else: return np.eye(len(self.lattices))[y.astype(int)]

  def _format_from_clf_lbls(self, y):
    if self._clf_type != C.NN_CLF_TYPE: return y
    else: return np.argmax(y, axis=1)

  def fit(self, X, y, perf_xs, alpha, liq_alpha, overwrite=False):
    self.weights_path.mkdir(exist_ok=overwrite)
    # fit scaler to training data
    self.scaler.fit(X)
    joblib.dump(self.scaler, self.scaler_path)
    X = self.scaler.transform(X)
    # train classifier
    X, y = shuffle(X, y)
    y_clf = self._format_to_clf_lbls(y)
    X_train, y_train = X[:50000,:], y_clf[:50000,...]
    ran_gs = self._gs_clf(X_train, y_train, overwrite)
    if not ran_gs:
      self.classifier.fit(X_train, y_train)
    joblib.dump(self.classifier, self.classifier_path)
    # TODO add hparam grid searching
    # train outlier detector
    perf_xs = [np.array(x) for x in self.scaler.transform(perf_xs).tolist()]
    self.outlier_detector.fit(X, y, perf_xs, alpha, liq_alpha)
    self.outlier_detector.save(self.outlier_path)
    self.is_trained = True
    return self

  def fit_end2end(self, distort_bins=C.DFLT_DISTORT_BINS,
                        overwrite=False):
    if not overwrite and self.synth_feat_path.exists() \
                     and self.synth_lbls_path.exists():
      X = np.load(self.synth_feat_path)
      y = np.load(self.synth_lbls_path)
      alpha = np.load(self.synth_alpha_path)
      liq_alpha = np.load(self.liq_alpha_path)
    else:
      X, y, alpha, liq_alpha = self.compute_synth_features(distort_bins=distort_bins)
    if not overwrite and self.perf_feat_path.exists():
      perf_xs = self.load_perf_features()
    else:
      perf_xs = self.compute_perf_features()

    if not overwrite and False:
      raise NotImplementedError # TODO load cached models
    else:
      self.fit(X, y, perf_xs, alpha, liq_alpha)
    return self

  def predict(self, ov_data_collection):
    return self.predict_return_features(ov_data_collection)[1]

  def predict_return_features(self, ov_data_collection):
    assert self.is_trained
    X, alpha = self.featurizer.compute(ov_data_collection)
    X = self.scaler.transform(X)
    y_cand = self.classifier.predict(X)
    y_cand = self._format_from_clf_lbls(y_cand)
    y = self.outlier_detector.predict(X, alpha, y_cand, ov_data_collection)
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
      gc.collect()
