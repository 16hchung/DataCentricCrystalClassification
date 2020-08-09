from tqdm import tqdm
import numpy as np
import numpy.random
import numpy.linalg
import itertools

# TODO consider pyshtools?
from scipy.special import sph_harm
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder

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
    prev_r_min, prev_r_max = 0, 9
    for r in n_neighs:
      r_min, r_max = min(r), max(r)
      assert prev_r_min < r_min < prev_r_max < r_max

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
    # TODO concat
    raise NotImplementedError

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
    max_neigh    = self.max_neigh
    n_neighs     = self.n_neighs
    n_rsf_per_mu = self.rsf_config.n_rsf_per_mu
    mu_step      = self.rsf_config.mu_step
    sigma_scale  = self.rsf_config.sigma_scale
    
    if not isinstance(R_cart, np.array):
      R_cart = self._get_deltas(ov_data_collection) # shape: (n_atoms, max_neigh, 3)
    n_atoms = len(R)

    raise NotImplementedError

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
