from pathlib import Path
from tqdm import tqdm
import fire

from ovito.io import import_file, export_file

from dc3.model.full_pipeline import DC3Pipeline
from dc3.util import constants as C
from dc3.eval.benchmarker import Benchmarker
from dc3.data.file_io import recursive_in_out_file_pairs

def train(overwrite=False):
  pipeline = DC3Pipeline(overwrite=overwrite)
  if not pipeline.is_trained:
    pipeline.fit_end2end()

def eval(metadata_path, results_path, pipeline_name='dc3', overwrite=False):
  # make result paths
  results_path = Path(results_path)
  results_path.mkdir(exist_ok=True)
  acc_comparison_path = results_path / 'accuracy_comparison.csv'
  plt_comparison_path = results_path / 'accuracy_plots'
  X_cache_path = results_path / 'X_cache.pkl'
  y_cache_path = results_path / 'y_cache.pkl'

  pipeline = DC3Pipeline()
  benchmarker = Benchmarker.from_metadata_path(pipeline,
                                               metadata_path,
                                               X_pkl_path=X_cache_path,
                                               y_pkl_path=y_cache_path)
  benchmarker.plot_accuracy_comparison(pipeline_name, plt_comparison_path)
  benchmarker.save_accuracy_comparison(pipeline_name, acc_comparison_path)

def inference(input_dir, output_name):
  pipeline = DC3Pipeline()
  output_dir = pipeline.inference_rt / output_name
  output_dir.mkdir(parents=True, exist_ok=True)
  for in_path, out_path in tqdm(recursive_in_out_file_pairs(input_dir,
                                                            output_dir,
                                                            ext='.gz')):
    ov_data = import_file(in_path).compute()
    y = pipeline.predict(ov_data)
    ov_data.particles_.create_property('Lattice', data=y)
    export_file(ov_data,
                out_path,
                'lammps/dump',
                columns=['Position.X', 'Position.Y', 'Position.Z', 'Lattice'])

# TODO prioritize non-scripting version (ovito UI version)

if __name__=='__main__':
  fire.Fire()
