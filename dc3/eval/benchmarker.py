from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import pickle as pk

from ..util import constants as C
from ..util.util import lazy_property
from ..data.file_io import glob_pattern_inference

class Benchmarker:
  def __init__(self, pipeline,
                     metadata,
                     dump_tmplt,
                     y_pkl_path=None,
                     X_pkl_path=None):
    self.pipeline = pipeline
    self.metadata = metadata
    self.dump_tmplt = dump_tmplt
    # TODO remove and implement feature + label caching
    #self.metadata = self.metadata[(self.metadata.T_h == 1) | (self.metadata.T_h == .2)].reset_index(drop=True)

    # construct glob patterns + other info for data loading
    self.metadata['fpattern'] = self.metadata.apply(
      lambda row: self.dump_tmplt.format(lattice=row.lattice, T_h=row.T_h),
      axis=1
    )
    self.metadata['y_true'] = self.metadata.lattice.apply(
      lambda latt: self.pipeline.name_to_lbl_latt[latt][0]
    )
    # indices in y_preds_list will be aligned with metadata index
    # shape should be: (len(self.metadata), total atoms per row)
    if y_pkl_path and X_pkl_path \
                  and Path(y_pkl_path).exists() \
                  and Path(X_pkl_path).exists():
      with open(X_pkl_path, 'rb') as Xf, open(y_pkl_path, 'rb') as yf:
        self.Xs_list      = pk.load(Xf)
        self.y_preds_list = pk.load(yf)
    else:
      self.Xs_list, self.y_preds_list = zip(*[
        glob_pattern_inference(row.fpattern, self.pipeline)
        for row in tqdm(self.metadata.itertuples(), total=len(self.metadata))
      ])
      if y_pkl_path and X_pkl_path:
        with open(X_pkl_path, 'wb') as Xf, open(y_pkl_path, 'wb') as yf:
          pk.dump(self.Xs_list,      Xf)
          pk.dump(self.y_preds_list, yf)
    tqdm.pandas()

  @classmethod
  def from_metadata_path(cls, pipeline, metadata_path, **kwargs):
    with open(metadata_path) as f:
      dump_tmplt = f.readline().strip()
      metadata = pd.read_csv(f, index_col=0)
    return cls(pipeline, metadata, dump_tmplt, **kwargs)

  def global_metrics(self):
    raise NotImplementedError

  def metrics_for_T(self, T_h):
    raise NotImplementedError

  def metrics_for_all_T(self):
    raise NotImplementedError

  def save_accuracy_comparison(self, pipeline_name, save_path):
    self._add_acc_to_metadata(pipeline_name)
    self.metadata.to_csv(save_path)

  def plot_accuracy_comparison(self, pipeline_name,
                                     save_dir,
                                     only_methods=['PTM', 'CNA'],
                                     rcParams={'font.size': 16, 
                                               'figure.autolayout': True}):
    self._add_acc_to_metadata(pipeline_name)
    if only_methods == None:
      df = self.metadata
    else:
      df = self.metadata[only_methods + [pipeline_name, 'T_h', 'lattice']]
    df = df.set_index('T_h')

    save_dir = Path(save_dir)
    save_dir.mkdir()
    for latt in self.pipeline.lattices:
      plt.rcParams.update()
      df.groupby('lattice').get_group(latt.name).plot(legend=True)
      plt.axvline(x=1,ls='--', c='k', lw=1.0)
      plt.title(f'Test Accuracy for {latt.name.upper()}')
      plt.xlabel(r'$T/T_m$')
      plt.ylabel('Accuracy')
      plt.savefig(save_dir / f'{latt.name}.png', dpi=300)
      plt.clf()

  def _add_acc_to_metadata(self, pipeline_name):
    if pipeline_name in self.metadata.columns:
      return
    def row_acc(row):
      y_pred = self.y_preds_list[row.name]
      y_true = np.array([row.y_true]*len(y_pred))
      return accuracy_score(y_true, y_pred)
    self.metadata[pipeline_name] = self.metadata.apply(row_acc, axis=1)
