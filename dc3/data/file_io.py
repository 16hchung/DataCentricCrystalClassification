from pathlib import Path
from tqdm import tqdm
import numpy as np

from ovito.io import import_file

from ..features import Featurizer

def recursive_in_out_file_pairs(input_dir, output_dir, ext='.dump'):
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)
  input_files = input_dir.glob(f'**/*{ext}')
  in_out_pairs = []
  for infile in input_files:
    rel_infile = infile.relative_to(input_dir)
    outfile = output_dir / rel_infile
    outfile.parent.mkdir(exist_ok=True, parents=True)
    in_out_pairs.append( (str(infile), str(outfile)) )
  return in_out_pairs

def glob_pattern_inference(glob_pattern, dc3_pipeline):
  files = [str(p) for p in Path('.').glob(glob_pattern)]
  ov_pipeline = import_file(files)
  Xs = []
  y_preds = []
  pbar = tqdm(range(ov_pipeline.source.num_frames))
  for frame in pbar:
    pbar.set_description(f'computing features for files {glob_pattern})')
    data_collection = ov_pipeline.compute(frame)
    X, y_pred = dc3_pipeline.predict_return_features(data_collection)
    Xs.append(X)
    y_preds.append(y_pred)
  return np.concatenate(Xs, axis=0), np.concatenate(y_preds)
