from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import pickle as pk
import joblib

from ..util import constants as C
from ..util.util import lazy_property, TwoDArrayCombiner
from ..data.file_io import glob_pattern_inference, expand_metadata_by_T

class Benchmarker:
  def __init__(self, pipeline,
                     metadata,
                     y_pkl_path=None,
                     X_pkl_path=None):
    self.pipeline = pipeline
    self.summary_metadata = metadata
    # TODO remove and implement feature + label caching
    #self.metadata = self.metadata[(self.metadata.T_h == 1) | (self.metadata.T_h == .2)].reset_index(drop=True)

    # construct glob patterns + other info for data loading
    self.metadata = expand_metadata_by_T(metadata)
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
      tags, Xs_list, y_preds_list = zip(*[
        [(row.material, row.lattice, row.T_h)]
        + glob_pattern_inference(row.glob_pattern, self.pipeline)
        for row in tqdm(self.metadata.itertuples(), total=len(self.metadata))
      ])
      self.Xs_list = {tag: X for tag, X in zip(tags, Xs_list)}
      self.y_preds_list = {tag: y_pred for tag, y_pred in zip(tags, y_preds_list)}
      if y_pkl_path and X_pkl_path:
        with open(X_pkl_path, 'wb') as Xf, open(y_pkl_path, 'wb') as yf:
          pk.dump(self.Xs_list,      Xf)
          pk.dump(self.y_preds_list, yf)
    tqdm.pandas()

  @classmethod
  def from_metadata_path(cls, pipeline, metadata_path, **kwargs):
    metadata = pd.read_csv(metadata_path)
    return cls(pipeline, metadata, **kwargs)

  def global_metrics(self):
    raise NotImplementedError

  def metrics_for_T(self, T_h):
    raise NotImplementedError

  def metrics_for_all_T(self):
    raise NotImplementedError

  def save_accuracy_comparison(self, pipeline_name,
                                     competing_accuracies_path,
                                     save_path):
    accuracies = self._add_acc_to_metadata(pipeline_name,
                                           competing_accuracies_path)
    accuracies.to_csv(save_path)

  def visualize_features(self, save_path,
                               perplexities=[10, 50, 100, 200, 500, 1000]):
    # path manip
    data_path = save_path / 'data'
    plot_path = save_path / 'figures'
    data_path.mkdir(parents=True)
    plot_path.mkdir()

    # fit PCA to synthetic data
    X_tr_all = self.pipeline.load_synth_feat_lbls()[0]
    pca = PCA()
    pca.fit(X_tr_all)
    joblib.dump(pca, str(data_path / 'pca.pkl'))
    explained_variance = pca.explained_variance_ratio_.cumsum()
    N_var_99 = np.argmax(explained_variance > .99)
    with open(str(data_path / 'pca_explained_variance.pkl'), 'wb') as f:
      pk.dump({'explained_variance': explained_variance,
               'N_var_99': N_var_99}, f)
    print(f'99% of variance explained with {N_var_99} PCA components')

    # sample points to perform tsne
    X_te, y_te, X_tr, y_tr = self._sample_from_all_lattices()
    np.save(data_path / 'X_md_sample.pkl', X_te)
    np.save(data_path / 'y_md_sample.pkl', y_te)
    np.save(data_path / 'X_synthetic_sample.pkl', X_tr)
    np.save(data_path / 'y_synthetic_sample.pkl', y_tr)
    X_perf = np.row_stack(self.pipeline.outlier_detector.perf_xs)
    Xcombiner = TwoDArrayCombiner(X_te, X_tr, X_perf)
    X = Xcombiner.combined()

    # compute pca, then tsne --> plot
    X_PCA = pca.transform(X)[:,:N_var_99]
    X_te_PCA, X_tr_PCA, X_perf_PCA = Xcombiner.decompose(X_PCA)
    np.save(data_path / 'X_md_sample_PCA.pkl', X_te_PCA)
    np.save(data_path / 'X_synthetic_sample_PCA.pkl', X_tr_PCA)
    np.save(data_path / 'X_perf_PCA.pkl', X_perf_PCA)
    xlim, ylim = self._get_xlim_ylim(X_PCA)
    self._plot_scatter(X_te_PCA, y_te, X_perf_PCA,
                       plot_path / f'PCA_md.png',
                       xlim=xlim, ylim=ylim,
                       plot_type='PCA')
    self._plot_scatter(X_tr_PCA, y_tr, X_perf_PCA,
                       plot_path / f'PCA_synthetic.png',
                       xlim=xlim, ylim=ylim,
                       plot_type='PCA')
    for perplexity in perplexities:
      X_tsne = TSNE(perplexity=perplexity, verbose=2).fit_transform(X_PCA)
      X_te_tsne, X_tr_tsne, X_perf_tsne = Xcombiner.decompose(X_tsne)
      np.save(data_path / 'X_md_sample_tsne.pkl', X_te_tsne)
      np.save(data_path / 'X_synthetic_sample_tsne.pkl', X_tr_tsne)
      np.save(data_path / 'X_perf_tsne.pkl', X_perf_tsne)
      xlim, ylim = self._get_xlim_ylim(X_tsne)
      self._plot_scatter(X_te_tsne, y_te, X_perf_tsne,
                         plot_path / f'tSNE_md_{perplexity}.png',
                         xlim=xlim, ylim=ylim, plot_type='tSNE')
      self._plot_scatter(X_tr_tsne, y_tr, X_perf_tsne,
                         plot_path / f'tSNE_synthetic_{perplexity}.png',
                         xlim=xlim, ylim=ylim, plot_type='tSNE')
    #TODO temperature dependent plot (maybe diff fxn)

  def plot_accuracy_comparison(self, pipeline_name,
                                     competing_accuracies_path,
                                     save_dir,
                                     only_methods=['PTM', 'CNA'],
                                     rcParams={'font.size': 16, 
                                               'figure.autolayout': True}):
    accuracies = self._add_acc_to_metadata(pipeline_name,
                                           competing_accuracies_path)
    if only_methods == None:
      df = accuracies
    else:
      df = accuracies[only_methods + [pipeline_name, 'T_h', 'material']]
    material_to_maxT = {
      row.material:row.max_T_h for row in self.summary_metadata.itertuples()
    }
    df = df[df.apply(
      lambda row: row['T_h'] < material_to_maxT[row['material']], axis=1
    )]
    df = df.set_index('T_h')

    save_dir = Path(save_dir)
    save_dir.mkdir()
    plt.clf()
    for row in self.summary_metadata.itertuples():
      plt.rcParams.update()
      df.groupby('material').get_group(row.material).sort_index().plot(legend=True)
      plt.axvline(x=1,ls='--', c='k', lw=1.0)
      plt.title(f'Test Accuracy for {row.material} ({row.lattice.upper()})')
      plt.xlabel(r'$T/T_m$')
      plt.ylabel('Accuracy')
      plt.savefig(save_dir / f'{row.material}.png', dpi=300)
      plt.clf()
 
  def _get_xlim_ylim(self, X):
    xlim = 1.1*np.array([min(X[:,0]), max(X[:,0])])
    ylim = 1.1*np.array([min(X[:,1]), max(X[:,1])])
    return xlim, ylim

  def _plot_scatter(self, X, y, X_perf,
                          title,
                          xlim=None, ylim=None,
                          fontsize=18,
                          plot_type='tSNE'):
    if xlim is None or ylim is None:
      xlim, ylim = self._get_xlim_ylim(X)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_axes([.06, .06, .9, .9])
    for ilatt, latt in enumerate(self.pipeline.lattices):
      ax.plot(X[y==ilatt,0], X[y==ilatt,1], f'C{ilatt}{C.MARKERS[ilatt]}', 
              alpha=.15, mew=0, zorder=0-ilatt, label=latt.name.upper())
      ax.plot(X_perf[ilatt,0], X_perf[ilatt,1], f'k{C.MARKERS[ilatt]}', zorder=100)
    ax.set_xlabel(f'First {plot_type} dimension', fontsize=18)
    ax.set_ylabel(f'Second {plot_type} dimension', fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    legend = plt.legend(handletextpad=0.1, loc='best', ncol=1, 
                        frameon=True, fontsize=14)
    for l in legend.legendHandles: 
      l._legmarker.set_alpha(1)
    fig.savefig(title, dpi=300)
    plt.close()
    plt.clf()

  def _add_acc_to_metadata(self, pipeline_name, competing_accuracies_path):
    def row_acc(row):
      y_pred = self.y_preds_list[(row.material, row.lattice, row.T_h)]
      y_true = np.array([row.y_true]*len(y_pred))
      return accuracy_score(y_true, y_pred)

    competing = pd.read_csv(competing_accuracies_path)
    if pipeline_name not in self.metadata.columns:
      self.metadata[pipeline_name] = self.metadata.apply(row_acc, axis=1)
    return pd.merge(self.metadata, competing, how='outer', on=['material', 'T_h'])

  def _sample_from_all_lattices(self, N=10000):
    # collect all into same array (to shuffle later)
    Xs, ys = [], []
    for row in self.metadata.itertuples():
      X = self.Xs_list[(row.material, row.lattice, row.T_h)]
      y = np.array([row.y_true]*len(X))
      Xs.append(X)
      ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    X_te, y_te = shuffle(X, y)
    X_tr, y_tr = shuffle(*self.pipeline.load_synth_feat_lbls(scaled=True))
    return X_te[:N,...], y_te[:N,...], X_tr[:N,...], y_tr[:N,...]
