from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd
import numpy as np
import pickle as pk
import joblib

from ..util import constants as C
from ..util.util import lazy_property, TwoDArrayCombiner, bootstrap
from ..data.file_io import glob_pattern_inference, expand_metadata_by_T

class Benchmarker:
  def __init__(self, pipeline,
                     metadata,
                     y_pkl_path=None,
                     X_pkl_path=None,
                     alpha_pkl_path=None):
    self.pipeline = pipeline
    self.summary_metadata = metadata
    self._material_to_maxT = {
      row.material:row.max_T_h for row in self.summary_metadata.itertuples()
    }
    # construct glob patterns + other info for data loading
    self.metadata = expand_metadata_by_T(metadata)
    self.metadata['y_true'] = self.metadata.lattice.apply(
      lambda latt: self.pipeline.name_to_lbl_latt[latt][0]
    )
    # indices in y_preds_list will be aligned with metadata index
    # shape should be: (len(self.metadata), total atoms per row)
    if X_pkl_path and Path(X_pkl_path).exists() \
                  and alpha_pkl_path \
                  and Path(alpha_pkl_path).exists():
      self._load_test_features(X_pkl_path, alpha_pkl_path)

      if y_pkl_path and Path(y_pkl_path).exists():
        self._load_test_preds(y_pkl_path)
      # feature computation is bottleneck => pass cached features to clf
      else:
        self._compute_test_preds(y_pkl_path)
    # neither features nor preds are cached
    else:
      self._compute_test_features_and_preds(X_pkl_path,
                                            alpha_pkl_path,
                                            y_pkl_path)
    tqdm.pandas()

  ### INITIALIZATION HELPERS ###

  def _load_test_features(self, X_pkl_path, alpha_pkl_path):
    with open(X_pkl_path, 'rb') as Xf, open(alpha_pkl_path, 'rb') as af:
      self.Xs_list     = pk.load(Xf)
      self.alphas_list = pk.load(af)

  def _load_test_preds(self, y_pkl_path):
    with open(y_pkl_path, 'rb') as f:
      self.y_preds_list = pk.load(f)

  def _compute_test_preds(self, y_pkl_path):
    tags = [
      (row.material, row.lattice, row.T_h)
      for row in self.metadata.itertuples()
    ]
    self.y_preds_list = {
      tag : self.pipeline.predict_from_features(self.Xs_list[tag], 
                                                self.alphas_list[tag])
      for tag in tqdm(tags, desc='Inference on cached features:')
    }
    with open(y_pkl_path, 'wb') as f:
      pk.dump(self.y_preds_list, f)

  def _compute_test_features_and_preds(self, 
                                       X_pkl_path, 
                                       alpha_pkl_path, 
                                       y_pkl_path):
    tags, Xs_list, alphas_list, y_preds_list = zip(*[
      [(row.material, row.lattice, row.T_h)]
      + glob_pattern_inference(row.glob_pattern, self.pipeline)
      for row in tqdm(self.metadata.itertuples(), total=len(self.metadata))
    ])
    self.Xs_list = {tag: X for tag, X in zip(tags, Xs_list)}
    self.alphas_list = {tag: X for tag, X in zip(tags, alphas_list)}
    self.y_preds_list = {tag: y_pred for tag, y_pred in zip(tags, y_preds_list)}
    if y_pkl_path and X_pkl_path and alpha_pkl_path:
      with open(X_pkl_path, 'wb') as Xf, \
           open(alpha_pkl_path, 'wb') as af, \
           open(y_pkl_path, 'wb') as yf:
        pk.dump(self.Xs_list,      Xf)
        pk.dump(self.y_preds_list, yf)
        pk.dump(self.alphas_list,  af)

  @classmethod
  def from_metadata_path(cls, pipeline, metadata_path, **kwargs):
    metadata = pd.read_csv(metadata_path)
    return cls(pipeline, metadata, **kwargs)

  ### TEXT OUTPUT METRICS ###

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

  ### FEATURE VISUALIZATIONS - Train and Test comparisons ###

  def viz_dim_reduction(self, save_path,
                              perplexities=C.TSNE_PERPLEX):
    # path manip
    data_path = save_path / 'data'
    plot_path = save_path / 'figures'
    data_path.mkdir(exist_ok=True, parents=True)
    plot_path.mkdir(exist_ok=True)

    X_tr_all = self.pipeline.load_synth_feat_lbls(scaled=True)[0]
    pca, N_var_99 = self._fit_PCA(X_tr_all, data_path)
    print(f'Fit PCA, N_var_99: {N_var_99}')

    # sample points to perform tsne
    X_te, _, y_te, X_tr, _, y_tr = self._sample_from_all_lattices()
    np.save(data_path / 'X_md_sample.pkl', X_te)
    np.save(data_path / 'y_md_sample.pkl', y_te)
    np.save(data_path / 'X_synthetic_sample.pkl', X_tr)
    np.save(data_path / 'y_synthetic_sample.pkl', y_tr)
    X_perf = np.row_stack(self.pipeline.outlier_detector.perf_xs)
    Xcombiner = TwoDArrayCombiner(X_te, X_tr, X_perf)
    # TODO clean up scaling discrepencies
    X = Xcombiner.combined()

    # compute pca, then tsne --> plot
    X_PCA = pca.transform(X)[:,:N_var_99]
    self._plot_joined_transformed(Xcombiner, X_PCA, y_te, y_tr,
                                  data_path, plot_path,
                                  f_suffix='PCA', plot_type='PCA')
    for perplexity in tqdm(perplexities, desc='tSNE perplexities:'):
      X_tsne = TSNE(n_jobs=-1,
                    perplexity=perplexity,
                    verbose=20).fit_transform(X_PCA)
      self._plot_joined_transformed(Xcombiner, X_tsne, y_te, y_tr,
                                    data_path, plot_path,
                                    f_suffix=f'tSNE_{perplexity}',
                                    plot_type='tSNE')

  def viz_Tdependent(self, save_path, perplexities=C.TSNE_PERPLEX):
    save_path = save_path / 't_dependent'
    fig_path = save_path / 'figures' # TODO: save data?
    data_path = save_path / 'data'
    fig_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    X_te,_,y_te,T_te,_,_,_ = self._sample_from_all_lattices(return_te_Ts=True)
    for ilatt, latt in enumerate(self.pipeline.lattices):
      X_perf = self.pipeline.outlier_detector.perf_xs[ilatt][np.newaxis,:]
      X_latt = X_te[y_te==ilatt]
      T_latt = T_te[y_te==ilatt]

      Xcombiner = TwoDArrayCombiner(X_latt, X_perf)
      pca, N_var_99 = self._fit_PCA(X_latt, data_path)
      X_PCA = pca.transform(Xcombiner.combined())[:,:N_var_99]
      for p in tqdm(perplexities, desc=f'tSNE perplexities for {latt.name}:'):
        X_tsne = TSNE(n_jobs=-1, perplexity=p, verbose=20).fit_transform(X_PCA)
        X_tsne, X_perf_tsne = Xcombiner.decompose(X_tsne)
        self._plot_with_color_map(X_perf_tsne, X_tsne, T_latt,
                                  fig_path / f'{latt.name}_tSNE_{p}.png',
                                  perf_lbl=f'Ideal {latt.name} structure',
                                  xlabel='First tSNE dimension',
                                  ylabel='Second tSNE dimension',
                                  colorbar_label='$T/T_\mathrm{m}$')

  def _plot_with_color_map(self, X_perf, X, color_lbls,
                                 save_path,
                                 perf_style='C3o',
                                 perf_lbl='',
                                 xlabel='',
                                 ylabel='',
                                 colorbar_label=''):
    plt.clf()
    plt.rcParams.update(C.RCPARAMS)
    fig = plt.figure()
    ax  = fig.add_axes([0.05, 0.06, 0.85, 0.90])

    # plot color-coded features
    color = C.DFLT_CM(color_lbls / color_lbls.max())
    ax.scatter(X[:,0], X[:,1], marker='^', alpha=.3, color=color, lw=0, s=40)
    # plot perfect features
    ax.plot(X_perf[0,0], X_perf[0,1], perf_style, 
            zorder=100, markersize=9, label=perf_lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    # Setup colorbar.
    ax_bar = fig.add_axes([0.92,0.05,0.015,0.90])
    norm = mpl.colors.Normalize(vmin=0, vmax=color_lbls.max())
    cb = mpl.colorbar.ColorbarBase(ax_bar, 
                                   cmap=C.DFLT_CM,
                                   norm=norm,
                                   orientation='vertical',
                                   spacing='proportional')
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, color_lbls.max()])
    cb.set_ticklabels(
      ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '%.2f' % color_lbls.max()]
    )
    cb.ax.tick_params(labelsize=10)
    cb.ax.set_title(colorbar_label, fontsize=12)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    plt.clf()

  def _plot_joined_transformed(self, Xcombiner, X, y_te, y_tr,
                                     data_path, plot_path,
                                     f_suffix='PCA',
                                     plot_type='PCA'):
    X_te, X_tr, X_perf = Xcombiner.decompose(X)
    np.save(data_path / f'X_md_sample_{f_suffix}.pkl', X_te)
    np.save(data_path / f'X_synthetic_sample_{f_suffix}.pkl', X_tr)
    np.save(data_path / f'X_perf_{f_suffix}.pkl', X_perf)
    xlim, ylim = self._get_xlim_ylim(X)
    self._plot_scatter(X_te, y_te, X_perf,
                       plot_path / f'md_{f_suffix}.png',
                       xlim=xlim, ylim=ylim, plot_type=plot_type)
    self._plot_scatter(X_tr, y_tr, X_perf,
                       plot_path / f'synthetic_{f_suffix}.png',
                       xlim=xlim, ylim=ylim, plot_type=plot_type)

  def _fit_PCA(self, X, data_path): # TODO move to util?
    pca = PCA()
    pca.fit(X)
    joblib.dump(pca, str(data_path / 'pca.pkl'))
    explained_variance = pca.explained_variance_ratio_.cumsum()
    N_var_99 = np.argmax(explained_variance > .99)
    with open(str(data_path / 'pca_explained_variance.pkl'), 'wb') as f:
      pk.dump({'explained_variance': explained_variance,
               'N_var_99': N_var_99}, f)
    print(f'99% of variance explained with {N_var_99} PCA components')
    return pca, N_var_99

  def viz_outlier(self, save_path):
    # TODO also save data
    fig_path = save_path / 'figures'
    # histograms of alphas
    X_te, A_te, y_te, X_tr, A_tr, y_tr = self._sample_from_all_lattices()
    A_liq = self.pipeline.load_synth_liq_alphas()
    self._plot_hist(fig_path / 'alphas_histogram.png', 
                    (A_tr, f'Train (rand distort)'),
                    (A_te, f'Test (MD)'),
                    (A_liq,'Train (rand distort liq)'),
                    xlabel=r'Coherence factor $\alpha$',
                    ylabel='Frequency',
                    vline_x=self.pipeline.outlier_detector.alpha_cutoff,
                    vline_lbl=r'$\alpha$ cutoff')

    # histograms of distances from perfect lattice
    X_perf = self.pipeline.outlier_detector.perf_xs
    dist_fn = self.pipeline.outlier_detector.distance
    plt.clf()
    fig_path = fig_path / 'unk_crystal'
    for ilatt, latt in enumerate(self.pipeline.lattices):
      dist_te = dist_fn(X_te[y_te==ilatt], X_perf[ilatt])
      dist_tr = dist_fn(X_tr[y_tr==ilatt], X_perf[ilatt])
      cut = self.pipeline.outlier_detector.lbl_to_cutoff[ilatt]
      self._plot_hist(fig_path / f'{latt.name}_dist_from_perf.png',
                      (dist_tr, 'Synthetic data set'),
                      (dist_te, 'Molecular Dynamics'),
                      xlabel=r'Distance $\delta_{y_i}(\mathbf{x}_i)$',
                      ylabel=r'Density distribution',
                      vline_x=cut,
                      vline_lbl='99th percentile')

  ### PLOTTING PERFORMANCE ###

  def plot_accuracy_comparison(self, pipeline_name,
                                     competing_accuracies_path,
                                     save_dir,
                                     pipeline_color='k',
                                     only_methods={'PTM':'b','CNA':'g'},
                                     rcParams=C.RCPARAMS):
    acc = self._add_acc_to_metadata(pipeline_name, competing_accuracies_path)
    acc = acc[acc.apply(
      lambda row: row['T_h'] < self._material_to_maxT[row['material']], axis=1
    )]

    assert only_methods and isinstance(only_methods, dict), \
           'must pass in methods to compare'
    only_methods[pipeline_name] = pipeline_color
    df = acc[['T_h', 'material']]
    for m in only_methods:
      df[m] = acc[f'{m}_acc']
    df = df.set_index('T_h')

    save_dir = Path(save_dir)
    save_dir.mkdir()
    plt.clf()
    for row in self.summary_metadata.itertuples():
      plt.rcParams.update(rcParams)
      df.groupby('material').get_group(row.material) \
                            .sort_index() \
                            .plot(legend=True, style=only_methods)
      for m, color in only_methods.items():
        this_acc = acc[acc.material==row.material].sort_values(by=['T_h'])
        if len(this_acc[this_acc[f'{m}_low'].isnull()]):
          continue
        plt.fill_between(this_acc.T_h,this_acc[f'{m}_low'],this_acc[f'{m}_high'],
                         color=color, alpha=.3, lw=0)
      # add error bars
      plt.axvline(x=1,ls='--', c='k', lw=1.0)
      plt.title(f'Test Accuracy for {row.material} ({row.lattice.upper()})')
      plt.xlabel(r'$T/T_m$')
      plt.ylabel('Accuracy')
      plt.savefig(save_dir / f'{row.material}.png', dpi=300)
      plt.clf()

  ### GENERAL HELPERS ###

  def _get_xlim_ylim(self, X):
    xlim = 1.1*np.array([min(X[:,0]), max(X[:,0])])
    ylim = 1.1*np.array([min(X[:,1]), max(X[:,1])])
    return xlim, ylim

  def _plot_hist(self, save_path, *X_and_lbls,
                       bins=50,
                       xlabel='',
                       ylabel='',
                       vline_x=None,
                       vline_lbl=None,
                       histtype='step'):
    plt.clf()
    plt.rcParams.update(C.RCPARAMS)
    for X,lbl in X_and_lbls:
      plt.hist(X, bins=bins, histtype=histtype, label=lbl)

    if vline_x is not None and vline_lbl is not None:
      plt.axvline(vline_x,
                  color='r', linestyle='dotted', label=vline_lbl, 
                  linewidth=2.5)

    plt.legend(fontsize='medium')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=300)
    plt.clf()

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
      acc, low_acc, high_acc = bootstrap(y_pred, row.y_true)
      return acc, low_acc, high_acc

    competing = pd.read_csv(competing_accuracies_path)
    if f'{pipeline_name}_acc' not in self.metadata.columns:
      acc, low, high = zip(*self.metadata.apply(row_acc, axis=1))
      self.metadata[f'{pipeline_name}_acc'] = acc
      self.metadata[f'{pipeline_name}_low'] = low
      self.metadata[f'{pipeline_name}_high'] = high
    return pd.merge(self.metadata, competing, how='outer', on=['material', 'T_h'])

  def _sample_from_all_lattices(self, N=C.DFLT_FEATURE_VIZ_N,
                                      transform=True,
                                      return_te_Ts=False):
    # collect all into same array (to shuffle later)
    Xs, alphas, ys, Ts = [], [], [], []
    for row in self.metadata.itertuples():
      if row.T_h > self._material_to_maxT[row.material]: continue
      X = self.Xs_list[(row.material, row.lattice, row.T_h)]
      alpha = self.alphas_list[(row.material, row.lattice, row.T_h)]
      y = np.array([row.y_true]*len(X))
      T = np.array([row.T_h]*len(X))

      Xs.append(X)
      alphas.append(alpha)
      ys.append(y)
      Ts.append(T)

    X = np.concatenate(Xs, axis=0)
    alpha = np.concatenate(alphas, axis=0)
    y = np.concatenate(ys, axis=0)
    T = np.concatenate(Ts, axis=0)
    X_te, a_te, y_te, T_te = shuffle(X, alpha, y, T)
    X_tr, a_tr, y_tr = shuffle(
      *self.pipeline.load_synth_feat_lbls(scaled=False)
    )
    if transform:
      X_te = self.pipeline.scaler.transform(X_te)
      X_tr = self.pipeline.scaler.transform(X_tr)
    ret = [X_te[:N,...], a_te[:N,...], y_te[:N,...]]
    if return_te_Ts:
      ret.append(T_te[:N,...])
    ret.extend([X_tr[:N,...], a_tr[:N,...], y_tr[:N,...]])
    return ret
