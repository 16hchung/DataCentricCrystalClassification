from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
import itertools
import traceback
import warnings
import json
pd.options.mode.chained_assignment = None

from ..util import constants as C

class GridSearchVisualiser:
  prefix = 'param_'
  def __init__(self, save_path,
                     results_pkl_path, 
                     clf_param_options, 
                     N):
    self.clf_param_options = clf_param_options
    with open(str(results_pkl_path), 'rb') as f:
      self.cv_results = pd.DataFrame(pk.load(f))
    self.N = N
    self.save_path = Path(save_path)
    self.save_path.mkdir(exist_ok=True)

  def _product_plot(self, x_axis, xlbl, save_path, df=None):
    if df is None:
      df = self.cv_results
    knobs = {k:v for k,v in self.clf_param_options.items()
                 if len(df[f'{self.prefix}{k}'].unique()) > 1 
                    and k != x_axis}
    cols = [f'{self.prefix}{k}' for k in knobs.keys()]
    results = df[cols + C.LC_YAXES]
    results[xlbl] = df[f'{self.prefix}{x_axis}']
    results = results.set_index(xlbl)
    plt.clf()
    for vals in itertools.product(*knobs.values()):
      vals = [tuple(v) if isinstance(v,list) else v for v in vals]
      kv_str_pairs = [f'{k}-{v}'for k,v in zip(knobs.keys(), vals)]
      fname = '_'.join(kv_str_pairs) + '.png'
      title = ' '.join(kv_str_pairs)
      grp = tuple(vals) if len(vals) > 1 else vals[0]
      results.groupby(cols).get_group(grp)[C.LC_YAXES] \
                           .sort_index() \
                           .plot(legend=True)
      plt.ylim([.06, .9])
      plt.title(title)
      plt.savefig(save_path / fname, dpi=300)
      plt.clf()

  def plot_learning_curves(self, overwrite=False):
    clf_gs_lc_path  = self.save_path / 'learning_curves'
    clf_gs_lc_path.mkdir(exist_ok=overwrite)
    df = self.cv_results.copy()
    x_key = f'{self.prefix}{C.LC_XAXIS}'
    df = self._convert_val_frac(df)
    self._product_plot(C.LC_XAXIS, 'training_examples', clf_gs_lc_path, df=df)

  def plot_model_size(self, overwrite=False):
    save_path = self.save_path / 'model_sz_vs_acc'
    save_path.mkdir(exist_ok=overwrite)
    df = self._filter_to_most_trained()
    df = self._reduce_mod_size(df)
    self._product_plot(C.MOD_SIZE_XAXIS, 'model size', save_path, df=df)

  def plot_lr_init(self, overwrite=False):
    save_path = self.save_path / 'lr_vs_acc'
    save_path.mkdir(exist_ok=overwrite)
    df = self._filter_to_most_trained()
    self._product_plot(C.LR_XAXIS, 'learning rate init', save_path, df=df)

  def plot_learning_curves_together(self, overwrite=False):
    save_path = self.save_path / 'lc_mod_szs_together'
    save_path.mkdir(exist_ok=overwrite)
    xlbl ='training examples'
    results = self._convert_val_frac(self.cv_results, new_key=xlbl)
    yaxis = C.LC_YAXES[-1] # only want test acc
    k_lr = f'{self.prefix}{C.LR_XAXIS}'
    for lr in results[k_lr].unique():
      df = results[results[k_lr]==lr]
      k_mod_size = f'{self.prefix}{C.MOD_SIZE_XAXIS}'
      # only keep columns we need
      df = df[[k_mod_size, xlbl, yaxis]].rename(columns={
                                                k_mod_size:C.MOD_SIZE_XAXIS}) \
                                        .drop_duplicates([C.MOD_SIZE_XAXIS,xlbl]) \
                                        .sort_values(xlbl, ignore_index=True)
      # "transpose" df so mod_sz are columns
      df_by_mod_sz = df[[xlbl]].drop_duplicates(ignore_index=True).set_index(xlbl)
      for mod_sz in np.sort(df[C.MOD_SIZE_XAXIS].unique()):
        filtered = df[df[C.MOD_SIZE_XAXIS]==mod_sz].set_index(xlbl)[yaxis]
        df_by_mod_sz[mod_sz] = filtered
      # plot
      plt.clf()
      df_by_mod_sz.plot(legend=True)
      #plt.yscale('function', functions=(lambda x:1000**x,
      #                                  lambda x: np.log(x)/np.log(1000)))
      plt.ylim([df_by_mod_sz.drop((5,5,5), axis=1).min().min() - .01,
                df_by_mod_sz.max().max() + .01])
      plt.savefig(save_path / f'lr_{lr}.png', dpi=300)
      plt.clf()

  def _filter_to_most_trained(self):
    k_val_frac = f'{self.prefix}{C.LC_XAXIS}'
    min_val_frac = self.cv_results[k_val_frac].min()
    df = self.cv_results[self.cv_results[k_val_frac] == min_val_frac]
    return df
  
  def _reduce_mod_size(self, df):
    k_mod_size = f'{self.prefix}{C.MOD_SIZE_XAXIS}'
    df[k_mod_size] = [tup[0] for tup in df[k_mod_size]]
    return df

  def _convert_val_frac(self, df, new_key=None):
    x_key = f'{self.prefix}{C.LC_XAXIS}'
    if new_key is None: new_key = x_key
    df[new_key] = self.N * (1 - df[x_key])
    return df
