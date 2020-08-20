'''
Script to train a pipeline with your own defined crystal lattices, still using
our default featurization settings.

FIRST: add your lattices to the list marked TODO below
THEN: Some sample commands...

To use default arguments:
  povitos examples/custom_lattices.py

To use your own arguments:
  povitos examples/custom_lattices.py --pipeline_output_rt=OUTPUT_ROOT_DIRECTORY \
                                      --input_dump_rt=DIRECTORY_CONTAINING_INPUT_SIM_FILES \
                                      --output_name=DIRNAME_FOR_INFERENCE_OUTPUT \
                                      --dump_file_ext=SIM_FILE_EXTENSION
This will 
1. Train your model to recognize the lattices specified below
2. Recursively find all files with dump_file_ext extension within the input_dump_rt directory
3. Run inference on found files and output to <pipeline_output_rt>/inference/<output_name>/
'''

from pathlib import Path
from tqdm import tqdm
import fire

from ovito.io import import_file, export_file

from dc3.util.util import Lattice
from dc3.model.full_pipeline import DC3Pipeline
from dc3.util.features import Featurizer
from dc3.data.file_io import recursive_in_out_file_pairs

def train_and_inference(pipeline_output_rt='pipeline',
                        input_dump_rt='example_lammps/dump',
                        output_name='example_predictions',
                        dump_file_ext='.gz'):
  lattices = [
    # TODO add Lattice objects here, eg...
    # Lattice(name='fcc', perfect_path=str(PERF_DUMP_RT/'dump_fcc_perfect_0.dat'), neigh_range=[0,12])
  ]
  featurizer = Featurizer(lattices=lattices)
  pipeline = DC3Pipeline(lattices=lattices,
                         featurizer=featurizer,
                         output_rt=pipeline_output_rt)
  if not pipeline.is_trained:
    pipeline.fit_end2end()
  pipeline.predict_recursive_dir(input_dump_rt, output_name, ext=dump_file_ext)

if __name__=='__main__':
  fire.Fire(train_and_inference)
