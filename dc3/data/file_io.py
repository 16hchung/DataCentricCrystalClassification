from pathlib import Path
from tqdm import tqdm
from parse import parse
import pandas as pd
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

def files_from_pattern(glob_pattern):
  return [str(p) for p in Path('.').glob(glob_pattern)]

def expand_metadata_by_T(metadata):
  def meta_from_path(path):
    r = parse(row.fname_tmplt, Path(path).name)
    return r.named

  all_names, all_lattices, all_Ts, all_globs = [], [], [], []
  for row in metadata.itertuples():
    all_files = files_from_pattern(row.glob_pattern)
    path_tmplt = str(Path(row.glob_pattern).parent / row.fname_tmplt)
    Ts = list(set([meta_from_path(dump)['T_h'] for dump in all_files]))
    globs_by_T = [path_tmplt.format(T_h=T, ts='*') for T in Ts]

    all_names.extend(len(Ts) * [row.name])
    all_lattices.extend(len(Ts) * [row.lattice])
    all_Ts.extend(Ts)
    all_globs.extend(globs_by_T)

  return pd.DataFrame({'name': all_names,
                       'lattice': all_lattices,
                       'T_h': all_Ts,
                       'glob_pattern': all_globs})

def glob_pattern_inference(glob_pattern, dc3_pipeline):
  files = files_from_pattern(glob_pattern)
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
