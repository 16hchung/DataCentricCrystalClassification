from copy import deepcopy
from inspect import signature
from tqdm import tqdm
import pickle as pk
import pandas as pd
import numpy as np
import numpy.random
import numpy.linalg
import itertools
import json

from ovito.io import import_file
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder
# TODO consider pyshtools?
from scipy.special import sph_harm
from scipy.stats import norm as sp_norm

from ..util import constants as C
from ..util.util import (Lattice,
                   range_list_max,
                   n_neighs_from_lattices,
                   split_kwargs_among_fxns)

class Featurizer: # TODO make more accessible from higher level dir
  ''' Computes features from an ovito DataCollection '''

  def __init__(self, lattices=C.DFLT_LATTICES,
                     steinhardt_n_ls=C.STEIN_NUM_lS,
                     shell_range=C.SHELL_RANGE, # eg use 4-16 instead of -16
                     n_rsf_per_mu=C.N_RSF_PER_MU,
                     rsf_mu_step=C.RSF_MU_STEP,
                     rsf_sigma_scale=C.RSF_SIGMA_SCALE,
                     rsf_use_neighbor_range=C.RSF_USE_NEIGH_RANGE,
                     coherence_ls_list=C.COH_L_LIST):
    # add args automatically as instance variables
    kwargs = locals()
    self.__dict__.update(kwargs)
    del self.__dict__['self']

    # check that n_neighs are fine
    assert rsf_use_neighbor_range, 'using preset # neigh for rsfs is deprecated'
    self.n_neighs = n_neighs_from_lattices(lattices)
    self._check_n_neighs(self.n_neighs)
    self.max_neigh = range_list_max(self.n_neighs)

  def save(self, save_path):
    init_args = [p for p in signature(self.__init__).parameters]
    save_args = { k:v for k,v in self.__dict__.items() if k in init_args }
    with open(str(save_path), 'w') as f:
      # TODO: look int using custom json serializer
      # https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
      json.dump(save_args, f)

  @classmethod
  def from_saved_path(cls, save_path):
    with open(str(save_path)) as f:
      kwargs = json.load(f)
    (_,kwargs), = split_kwargs_among_fxns(cls, **kwargs)
    lattices = [Lattice(*l_args) for l_args in kwargs['lattices']]
    kwargs['lattices'] = lattices
    return cls(**kwargs)

  @staticmethod
  def _check_n_neighs(n_neighs):
    prev_start, prev_end = None, None
    for start, end in n_neighs:
      assert (prev_start == prev_end == None) or \
             (prev_start <= start < prev_end < end)
      prev_start, prev_end = start, end

  def compute(self, ov_data_collection):
    feature_sets = []
    R_cart = self._get_deltas(ov_data_collection)
    Q, Q_decomp = self.compute_steinhardt(ov_data_collection,
                                          R_cart=R_cart)
    alpha = self.compute_coherence(ov_data_collection, Q_decomp)
    R = self.compute_rsf(ov_data_collection, R_cart=R_cart)
    return np.concatenate([Q,R], axis=1), alpha

  def compute_perf_from_dump(self, dump_path):
    ov_data_collection = import_file(str(dump_path)).compute()
    X_all, _ = self.compute(ov_data_collection)
    # turn into dataframe to use pd functionality to find most common x
    df = pd.DataFrame(X_all)
    perf_x = ( df.round(decimals=C.FEATURE_PRECISION)
                 .groupby(list(df.columns)) # find unique feature vectors
                 .size().sort_values()      # sort counts for unique vctors
                 .index.tolist()[-1] )      # only grab most common x
    return np.array(perf_x)

  def compute_steinhardt(self, ov_data_collection, R_cart=None, **kwargs):
    '''
    TODO documentation
    '''
    max_neigh = self.max_neigh
    n_ls      = self.steinhardt_n_ls
  
    if not isinstance(R_cart, np.ndarray):
      # shape: (n_atoms, max_neigh, 3)
      R_cart = self._get_deltas(ov_data_collection, **kwargs)
    n_atoms = len(R_cart)
    # convert to spherical coords
    phi = np.arctan2(R_cart[:,:,1], R_cart[:,:,0]) # shape: (n_atoms, max_neigh)
    theta = np.arccos(R_cart[:,:,2] / np.linalg.norm(R_cart, axis=2))
    # compute for l's separately to make dimensions of sph_harm easier
    q_ls_list = []
    q_lms_list = []
    for l in range(1, n_ls+1):
      # compute spherical harmonics, shape: (n_atoms, max_neigh, 2l+1)
      Y_lm = sph_harm(np.arange(-l, l+1), # range of m's
                      l,
                      np.expand_dims(phi,   axis=-1),
                      np.expand_dims(theta, axis=-1))
      # sum over neighbors (cumsum to cover all possible n_neigh) to get q_lm
      q_lm = self._accumulate(Y_lm)
      # ^^^ has shape: (n_atoms, max_neigh, 2l+1)
      if l in self.coherence_ls_list:
        q_lms_list.append(q_lm[:,-1,:])
      q_lm_sqr = q_lm * np.conjugate(q_lm)
      # sum over m to get main portion of q
      k = 4*np.pi / (2*l+1)
      q_l = np.sqrt(np.real(
              k * np.sum(q_lm_sqr, axis=-1)
      )) # shape: (n_atoms, max_neigh)
      q_ls_list.append(q_l)
    q_l_3d = np.stack(q_ls_list, axis=1) # shape: (n_atoms, l, max_neigh)
    q_l_3d = q_l_3d[:,:,C.MIN_NEIGH-1:]
    n_possible_neigh = max_neigh - C.MIN_NEIGH + 1
    assert q_l_3d.shape[-1] == n_possible_neigh
    Q = np.reshape(q_l_3d, (n_atoms, n_possible_neigh*n_ls), order='C')
    Q_decomp = np.hstack(q_lms_list)
    return Q, Q_decomp # shape: (n_atoms, n_features --dflt 150)

  def compute_coherence(self, ov_data_collection, Q_decomp):
    Q_decomp = Q_decomp / np.linalg.norm(Q_decomp, axis=1)[:,np.newaxis]
    finder = NearestNeighborFinder(self.max_neigh, ov_data_collection)
    n_atoms = len(Q_decomp)
    alphas = []
    for iatom in range(n_atoms):
      v = Q_decomp[iatom,:]
      alphas.append(sum([
        np.vdot(Q_decomp[neigh.index,:], v).real
        for neigh in finder.find(iatom)
      ]))
    return np.array(alphas) / self.max_neigh

  def compute_rsf(self, ov_data_collection, R_cart=None, **kwargs):
    '''
    TODO documentation
    '''
    max_neigh    = self.max_neigh
    n_neighs     = self.n_neighs
    n_rsf_per_mu = self.n_rsf_per_mu
    mu_step      = self.rsf_mu_step
    sigma_scale  = self.rsf_sigma_scale
    use_neigh_rng= self.rsf_use_neighbor_range
   
    if not isinstance(R_cart, np.ndarray):
      # shape: (n_atoms, max_neigh, 3)
      R_cart = self._get_deltas(ov_data_collection, **kwargs)
    n_atoms = len(R_cart)
    # convert to from xyz to magnitude
    R = np.linalg.norm(R_cart, axis=2) # shape: (n_atoms, max_neigh)
    # Mus shape: (n_atoms, len(n_neighs) or max_neigh,  n_rsf_per_mu)
    # Sigmas shape: (n_atoms, len(n_neighs) or max_neigh)
    Mus, Sigmas, n_features = self._rsf_get_mus_sigmas(R)
    # use maximual r_cut with some buffer
    r_cut = np.max(Mus) + 4*np.max(Sigmas)
    # retrieve distances for each
    # shapes: (n_atoms, max neigh w/in cutoff per atom)
    distances = self._get_variable_neigh_distances(ov_data_collection, 
                                                   r_cut,
                                                   **kwargs)
    # NOTE: expanding dims so they can be broadcast together in norm.pdf
    # (n_atoms, len(n_neighs), n_rsf_per_mu, max neigh w/in cutoff per atom)
    soft_counts = sp_norm.pdf(distances[:, np.newaxis, np.newaxis, :],
                              loc=Mus[:, :, :, np.newaxis],
                              scale=Sigmas[:, :, np.newaxis, np.newaxis])
    # (n_atoms, len(n_neighs), n_rsf_per_mu)
    G_stacked = np.sum(soft_counts, axis=-1)
    G_stacked = np.sqrt(2*np.pi*np.square(Sigmas[:,:,np.newaxis])) * G_stacked
    G = np.reshape(G_stacked, (n_atoms, n_features), order='C')
    return G

  def _rsf_get_mus_sigmas(self, R):
    """ R  shape: (n_atoms, max_neigh) """
    max_neigh    = self.max_neigh
    sigma_scale  = self.rsf_sigma_scale
    use_neigh_rng= self.rsf_use_neighbor_range
    n_rsf_per_mu = self.n_rsf_per_mu
    mu_step      = self.rsf_mu_step

    # Use 6/8/12/16 neighbors
    # used for generating mus centered around main mus, eg if n_rsf_per_mu is 5, 
    #   want range to be -2, -1, 0, 1, 2
    mu_incr_range = range(int(n_rsf_per_mu / -2),
                          int(n_rsf_per_mu / 2) + 1)
    # calculate mus and sigmas
    Mu_list = []
    Sigma_list = []
    for n_neigh in range(C.MIN_NEIGH, max_neigh + 1):
      Mu_center = np.mean(R[:,:n_neigh], axis=-1) # appended shape: (n_atoms)
      Sigma_list.append(sigma_scale * Mu_center)
      # generate a bunch of mus centered around this one
      center_Mu_list = []
      for mu_incr in mu_incr_range:
        scaled_incr = mu_incr * mu_step
        center_Mu_list.append(Mu_center + scaled_incr * Mu_center)
      # shape: (n_atoms, n_rsf_per_mu)
      Mu_list.append( np.stack(center_Mu_list, axis=-1) ) 
    # stack mulist and sigma list
    # shape: (n_atoms, # n_neighs,  n_rsf_per_mu)
    Mus = np.stack(Mu_list, axis=1)
    Sigmas = np.stack(Sigma_list, axis=-1) # shape: (n_atoms, # n_neighs)
    return Mus, Sigmas, (max_neigh - C.MIN_NEIGH + 1) * n_rsf_per_mu

  def _accumulate(self, X, dtype=complex):
    ''' X shape: (n_atoms, max_neigh, ...) '''
    max_neigh = self.max_neigh
    n_neighs  = self.n_neighs

    assert len(X.shape) in [2, 3]
    Ns_shape = (1,max_neigh,1) if len(X.shape) == 3 else (1,max_neigh)
    Ns = np.arange(1, max_neigh+1, dtype=dtype).reshape(Ns_shape)
    X_accum = np.cumsum(X, axis=1)
    if not self.shell_range:
      return X_accum / Ns

    X_shell_accum = X_accum.copy()
    Ns_shell = Ns.copy()
    prev_start, prev_end = 0,0
    for (start, end) in n_neighs:
      if not start: 
        prev_start, prev_end = start, end
        continue
      X_shell_accum[:, prev_end:end, ...] -= X_accum[:, start-1:start, ...]
      Ns_shell[:, prev_end:end, ...] -= Ns[:, start-1:start, ...]
      prev_start, prev_end = start, end

    return X_shell_accum / Ns_shell

  def _get_deltas(self, ov_data_collection):
    # generate R_cart: matrix of deltas to neighbors in cartesion coords
    #           shape: (n_atoms, max_neigh, 3)
    # TODO find non-ovito version of this
    finder = NearestNeighborFinder(self.max_neigh, ov_data_collection)
    R_list = [
      [neigh.delta for neigh in finder.find(iatom)]
      for iatom in range(ov_data_collection.particles.count)
    ]
    R_cart = np.array(R_list)
    return R_cart

  def _get_variable_neigh_distances(self, ov_data_collection,
                                          r_cut,
                                          pad=np.inf):
    # only choose neighbors of certain type
    finder = CutoffNeighborFinder(cutoff=r_cut,
                                  data_collection=ov_data_collection)
    # retrieve neighbor distances from ovito finder (all elements may
    # be different lengths, shape: list len = n_atoms, each el len = ?
    distances_list = [
        np.array( [neigh.distance for neigh in finder.find(iatom)] )
        for iatom in range(ov_data_collection.particles.count)
    ]
    max_len = max([len(l) for l in distances_list])
    padded_list = [
        np.pad(l, (0, max_len - len(l)), mode='constant', constant_values=pad)
        for l in distances_list
    ]
    distances = np.stack(padded_list, axis=0) # shape: (n_atoms, max_len)
    return distances
