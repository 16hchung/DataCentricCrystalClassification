'''
Script to train our default pipeline, evaluate/reproduce our figures, and run
inference on your own simulation output files
'''

from pathlib import Path
from tqdm import tqdm
import json
import fire

from dc3.model.full_pipeline import DC3Pipeline
from dc3.util import constants as C
from dc3.util.features import Featurizer
from dc3.eval.benchmarker import Benchmarker
from dc3.data.file_io import recursive_in_out_file_pairs

def train(overwrite=False, 
          output_rt=C.DFLT_OUTPUT_RT,
          clf_param_opt_json=None,
          **featurizer_kwargs):
  featurizer = Featurizer(**featurizer_kwargs)
  clf_param_options = None
  if clf_param_opt_json:
    with open(clf_param_opt_json) as f: clf_param_options = json.load(f)
    for k,v in clf_param_options.items():
      if isinstance(v[0], list): 
        clf_param_options[k] = [tuple(x) for x in v] # make items copyable
  pipeline = DC3Pipeline(overwrite=overwrite,
                         featurizer=featurizer,
                         clf_type=C.NN_CLF_TYPE,
                         clf_params={},
                         clf_param_options=clf_param_options,
                         output_rt=output_rt)
  if not pipeline.is_trained:
    pipeline.fit_end2end()

def eval(metadata_path,
         results_path,
         output_rt=C.DFLT_OUTPUT_RT,
         pipeline_name='dc3',
         overwrite=False):
  # make result paths
  results_path = Path(results_path)
  results_path.mkdir(exist_ok=True)
  acc_comparison_path = results_path / 'accuracy_comparison.csv'
  plt_comparison_path = results_path / 'accuracy_plots'
  X_cache_path = results_path / 'X_cache.pkl'
  y_cache_path = results_path / 'y_cache.pkl'

  pipeline_kwargs = {'output_rt': output_rt,
                     'clf_type': C.NN_CLF_TYPE}
  if output_rt != C.DFLT_OUTPUT_RT: pipeline_kwargs['featurizer'] = None
  pipeline = DC3Pipeline(**pipeline_kwargs)
  benchmarker = Benchmarker.from_metadata_path(pipeline,
                                               metadata_path,
                                               X_pkl_path=X_cache_path,
                                               y_pkl_path=y_cache_path)
  benchmarker.plot_accuracy_comparison(pipeline_name, plt_comparison_path)
  benchmarker.save_accuracy_comparison(pipeline_name, acc_comparison_path)

#@profile
def inference(input_dir, output_name):
  pipeline = DC3Pipeline()
  pipeline.predict_recursive_dir(input_dir, output_name, ext='.gz')

# TODO prioritize non-scripting version (ovito UI version)

if __name__=='__main__':
  fire.Fire()
