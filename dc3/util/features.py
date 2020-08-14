from tqdm import tqdm
import numpy as np
import numpy.random
import numpy.linalg
import itertools

from ovito.data import NearestNeighborFinder, CutoffNeighborFinder
# TODO consider pyshtools?
from scipy.special import sph_harm
from scipy.stats import norm as sp_norm

from . import constants as C

class SteinhardtNelsonConfig:
  def __init__(self, max_neigh=C.MAX_NEIGH,
                     max_1st_shell=C.MAX_1ST_SHELL
                     n_ls=C.STEIN_NUM_lS):
    self.n_ls = n_ls


class RSFConfig:
  def __init__(self, n_rsf_per_mu=C.N_RSF_PER_MU,
                     mu_step=C.RSF_MU_STEP,
                     sigma_scale=C.RSF_SIGMA_SCALE):
    self.n_rsf_per_mu = n_rsf_per_mu
    self.mu_step      = mu_step
    self.sigma_scale  = sigma_scale


class FeatureComputer: # TODO make more accessible from higher level dir
  ''' Computes features from an ovito DataCollection '''
  def __init__(self, n_neighs=C.DFLT_N_NEIGHS,
                     stein_config=SteinhardtNelsonConfig(),
                     rsf_config=RSFConfig()):
    self._check_n_neighs(n_neighs)
    self.n_neighs     = n_neighs
    self.stein_config = stein_config
    self.rsf_config   = rsf_config

  @staticmethod
  def _check_n_neighs(n_neighs):
    prev_start, prev_end = 0, 9
    for start, end in n_neighs:
      assert prev_start < start < prev_end < end

  @property
  def max_neigh(self):
    return self._max_neigh

  def __getattr__(self, attr):
    if attr == '_max_neigh':
      self._max_neigh = range_list_max(self.n_neighs)
      return self._max_neigh

  def compute(self, ov_data_collection):
    R_cart = self._get_deltas(ov_data_collection)
    Q = self.compute_steinhardt(ov_data_collection, R_cart=R_cart)
    G = self.compute_rsf(ov_data_collection, R_cart=R_cart)
    return np.concatenate((Q,G), axis=1)

  def compute_steinhardt(self, ov_data_collection, R_cart=None):
    '''
    TODO documentation
    '''
    max_neigh = self.max_neigh
    n_ls      = self.stein_config.n_ls
  
    import pdb;pdb.set_trace()
    if not isinstance(R_cart, np.array):
      R_cart = self._get_deltas(ov_data_collection) # shape: (n_atoms, max_neigh, 3)
    n_atoms = len(R_cart)
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

  def compute_rsf(ov_data_collection, R_cart=R_cart):
    '''
    TODO documentation
    '''
    max_neigh    = self.max_neigh
    n_neighs     = self.n_neighs
    n_rsf_per_mu = self.rsf_config.n_rsf_per_mu
    mu_step      = self.rsf_config.mu_step
    sigma_scale  = self.rsf_config.sigma_scale
   
    import pdb;pdb.set_trace()
    if not isinstance(R_cart, np.array):
      R_cart = self._get_deltas(ov_data_collection) # shape: (n_atoms, max_neigh, 3)
    n_atoms = len(R)
    # convert to from xyz to magnitude
    R = np.linalg.norm(R_cart, axis=2) # shape: (n_atoms, max_neigh)
    # used for generating mus centered around main mus, eg if n_rsf_per_mu is 5, 
    #   want range to be -2, -1, 0, 1, 2
    mu_incr_range = range(int(n_rsf_per_mu / -2),
                          int(n_rsf_per_mu / 2) + 1)
    # calculate mus and sigmas
    Mu_list = []
    Sigma_list = []
    for start, end in n_neighs:
      Mu_center = np.mean(R[start:end], axis=-1) # appended shape: (n_atoms)
      Sigma_list.append(sigma_scale * Mu_center)
      # generate a bunch of mus centered around this one
      center_Mu_list = []
      for mu_incr in mu_incr_range:
        scaled_incr = mu_incr * mu_step
        center_Mu_list.append(Mu_center + scaled_incr * Mu_center)
      Mu_list.append( np.stack(center_Mu_list, axis=-1) ) # shape: (n_atoms, n_rsf_per_mu)
    # stack mulist and sigma list
    Mus    = np.stack(Mu_list,    axis=1)  # shape: (n_atoms, len(n_neighs),  n_rsf_per_mu)
    Sigmas = np.stack(Sigma_list, axis=-1) # shape: (n_atoms, len(n_neighs))

    # use maximual r_cut with some buffer
    r_cut = np.max(Mus) + 4*np.max(Sigms)
    # retrieve distances for each
    # shapes: (n_atoms, max neigh w/in cutoff per atom)
    distances = self._get_variable_neigh_distances(ov_data_collection, r_cut)
    # NOTE: expanding dims in arrays so they can be broadcast together in norm.pdf
    # shape: (n_atoms, len(n_neighs), n_rsf_per_mu, max neigh w/in cutoff per atom)
    soft_counts = sp_norm.pdf(distances[:, np.newaxis, np.newaxis, :],
                              loc=Mus[:, :, :, np.newaxis],
                              scale=Sigmas[:, :, np.newaxis, np.newaxis])
    G_stacked = np.sum(soft_counts, axis=-1) # (n_atoms, len(n_neighs), n_rsf_per_mu)
    G = np.reshape(G_stacked, (n_atoms, len(n_neighs)*n_rsf_per_mu), order='C')
    return G

  def _get_deltas(self, ov_data_collection):
    n_atoms = ov_data_collection.particles.count

    # generate R_cart: matrix of deltas to neighbors in cartesion coords
    #           shape: (n_atoms, max_neigh, 3)
    finder = NearestNeighborFinder(self._max_neigh, ov_data_collection)
    R_list = [
      [neigh.delta for neigh in finder.find(iatom)]
      for iatom in tqdm(range(n_atoms))
    ]
    R_cart = np.array(R_list)
    return R_cart

  def _get_variable_neigh_distances(self, ov_data_collection, r_cut, pad=np.inf):
    n_atoms = ov_data_collection.particles.count

    finder = CutoffNeighborFinder(cutoff=r_cut,
                                  data_collection=ov_data_collection)
    # retrieve neighbor distances from ovito finder (all elements may
    # be different lengths, shape: list len = n_atoms, each el len = ?
    distances_list = [
        finder.neighbor_distances(iatom) for iatom in tqdm(range(n_atoms))
    ]
    max_len = max([len(l) for l in distances_list])
    padded_list = [
        np.pad(l, (0, max_len - len(l)), mode='constant', constant_values=pad)
        for l in distances_list
    ]
    distances = np.stack(padded_list, axis=0) # shape: (n_atoms, max_len)
    return distances
