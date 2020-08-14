from copy import deepcopy
import numpy as np
import numpy.random
import numpy.linalg

from ovito.io import import_file, export_file
from ovito.data import NearestNeighborFinder

from ..util import constants as C
from ..util.features import FeatureComputer

def distort_perfect(perfect_ovfile,
                    distort_bins=C.DFLT_DISTORT_BINS,
                    save_path=None):
  # First get first_neigh_d which we'll use to determine displacement sampling
  pipeline = import_file(perfect_ovfile)
  perf_data = pipeline.compute()
  n_atoms = perf_data.particles.count

  finder = NearestNeighborFinder(1, perf_data)
  first_neigh_d = min([finder.find(i)[0].distance for i in range(n_atoms)])

  # Include each bin as a separate "frame" in ovito pipeline (typically
  # used to handle time evolution)
  dup_pipeline = import_file([perfect_ovfile] * len(distort_bins))
  def pipeline_add_offsets(i_frame, data):
    distort_scale = distort_bins[i_frame]

    positions = daata.particles_.positions
    n_total_points = positions[:].shape[0]
    # generate unit vectors in random directions
    displacements = np.random.randn(3, n_total_points).T
    norms = np.linalg.norm(displacements, axis=1, keepdims=True)
    # generate uni distributed random displacement magnitudes to apply
    mags = np.random.uniform(0,
                             first_neigh_d * distort_scale,
                             size=n_total_points) \
                    .reshape(norms.shape)
    displacements = displacements / norms * mags
    data.particles_.positions_ += displacements

  pipeline.modifiers.append(pipeline_add_offsets)
  ov_collections = [
    pipeline.compute(i) for i in range(pipeline.source.num_frames)
  ]
  return ov_collections

def train_features_from_distorted(distorted_collections,
                                  save_path,
                                  feature_computer=FeatureComputer()):
  # TODO maybe also take feature calculator as arg?
  assert save_path
  raise NotImplementedError # returns  np array
