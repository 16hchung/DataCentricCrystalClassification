from tqdm import tqdm
import numpy as np
import numpy.random
import numpy.linalg
import itertools

# TODO consider pyshtools?
from scipy.special import sph_harm
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder

from . import constants as C

def compute(ov_data_collection, **kwargs):
  Q = compute_steinhardt(ov_data_collection, **kwargs)

def compute_steinhardt(ov_data_collection,
                       max_neigh=C.MAX_NEIGH,
                       n_ls=C.NUM_lS,
                       **unused_kwargs):
  '''
  TODO documentation
  '''
  n_atoms = ov_data_collection.particles.count

  # generate R_cart: matrix of deltas to neighbors in cartesion coords
  #           shape: (n_atoms, max_neigh, 3)
  finder = NearestNeighborFinder(max_neigh, ov_data_collection)
  import pdb;pdb.set_trace() # confirm shape
  R_list = [
    [neigh.delta for neigh in finder.find(iatom)]
    for iatom in tqdm(range(n_atoms))
  ]
  R_cart = np.array(R_list)
  # convert to spherical coords
  phi = np.arctan2(R_cart[:,:,1], R_cart[:,:,0]) # shape: (n_atoms, max_neigh)
  theta = np.arccos(R_cart[:,:,2] / np.linalg.norm(R_cart, axis=2))
  # compute for l's separately to make dimensions of sph_harm easier
  q_ls_list = []
  Ns = np.arange(1, max_neigh+1).reshape((1,max_neigh,1), dtype=complex)
  for l in range(n_ls):
    # compute spherical harmonics, shape: (n_atoms, max_neigh, 2l+1)
    # TODO make sure dtype is complex here...
    Y_lm = sph_harm(np.arange(-l, l+1), # range of m's
                    l,
                    np.expand_dims(phi,   axis=-1),
                    np.expand_dims(theta, axis=-1))
    # sum over neighbors (cumsum to cover all possible n_neigh) to get q_lm
    q_lm = np.cumsum(Y, axis=1) / Ns # TODO confirm this is doing the right thing
    # ^^^ has shape: (n_atoms, max_neigh, 2l+1)
    q_lm_sqr = q_lm * np.conjugate(q_lm)
    # sum over m to get main portion of q
    k = 4*np.pi / (2*l+1)
    q_l = np.sqrt(np.real(
            k * np.sum(q_lm_sqr, axis=-1)
    )) # shape: (n_atoms, max_neigh)
    q_ls_list.append(q_l)
  q_l_3d = np.stack(q_ls_list, axis=1) # shape: (n_atoms, l, max_neigh)
  # TODO make sure all of q_l_3d[:,:,0] are 1s (ie 1 neighbor)
  q_l_3d = q_l_3d[:,:,C.MIN_NEIGH:]
  n_possible_neigh = max_neigh - C.MIN_NEIGH + 1
  assert q_l_3d.shape[-1] == n_possible_neigh
  Q = np.reshape(q_l_3d, (n_atoms, n_possible_neigh*n_ls), order='C')
  return Q # shape: (n_atoms, n_features --dflt 150)


