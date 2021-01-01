import numpy as np

from ovito.data import NearestNeighborFinder, CutoffNeighborFinder

from .featurizer import Featurizer
from ..util import constants as C
from ..util.util import (Lattice,
                         range_list_max,
                         n_neighs_from_lattices,
                         split_kwargs_among_fxns)

class BinaryFeaturizer(Featurizer):

  def compute(self, ov_data_collection):
    feature_sets = []
    R_cart_same = self._get_deltas(ov_data_collection, same_species=True)
    Q_same, Q_decomp = self.compute_steinhardt(ov_data_collection,
                                               R_cart=R_cart_same,
                                               same_species=True)
    alpha = self.compute_coherence(ov_data_collection, Q_decomp)
    R_same = self.compute_rsf(ov_data_collection, 
                              R_cart=R_cart_same,
                              same_species=True)
    R_cart_diff = self._get_deltas(ov_data_collection, same_species=False)
    Q_diff, _ = self.compute_steinhardt(ov_data_collection,
                                        R_cart=R_cart_diff,
                                        same_species=False)
    R_diff = self.compute_rsf(ov_data_collection, 
                              R_cart=R_cart_diff, 
                              same_species=False)
    return np.concatenate([Q_same, R_same, Q_diff, R_diff], axis=1), alpha

  @staticmethod
  def _get_filtered_data_collection(ov_data_collection, particle_type):
    ov_neigh_data = ov_data_collection.clone()
    ov_neigh_data.cell_.pbc = (False, False, False) # make this copy a deep copy
    ov_neigh_data.apply(ExpressionSelectionModifier(expression=f'ParticleType == {particle_type}'))
    ov_neigh_data.apply(DeleteSelectedModifier())
    return ov_neigh_data

  def _get_deltas(self, ov_data_collection, same_species=True):
    all_types = ov_data_collection.particles['Particle Type']
    types = np.unique(all_types)
    assert len(types) == 2
    type0 = types[0 if same_species else 1]
    type1 = types[1 if same_species else 0]
    data0 = self._get_filtered_data_collection(ov_data_collection, type0)
    data1 = self._get_filtered_data_collection(ov_data_collection, type1)

    # generate R_cart: matrix of deltas to neighbors in cartesion coords
    #           shape: (n_atoms, max_neigh, 3)
    # TODO find non-ovito version of this
    finder0 = NearestNeighborFinder(self.max_neigh + 1, data0)
    finder1 = NearestNeighborFinder(self.max_neigh + 1, data1)
    R_list = [
      [neigh.delta for ineigh, neigh in enumerate(
         (finder0 if all_types[iatom] == types[0] else finder1).find_at(atom_pos)
       ) if ineigh > 0]
       for iatom, atom_pos in enumerate(ov_data_collection.particles.positions)
    ]

    #ov_neigh_data = ov_data_collection
    ## generate R_cart: matrix of deltas to neighbors in cartesion coords
    ##           shape: (n_atoms, max_neigh, 3)
    ## TODO find non-ovito version of this
    #finder = NearestNeighborFinder(self.max_neigh + 1, ov_neigh_data)
    #R_list = [
    #  [neigh.delta for i, neigh in enumerate(finder.find_at(atom_pos)) if i > 0]
    #  for atom_pos in ov_data_collection.particles.positions
    #]
    R_cart = np.array(R_list)
    return R_cart

  def _get_variable_neigh_distances(self, ov_data_collection,
                                          r_cut,
                                          pad=np.inf,
                                          same_species=True):
    types = ov_data_collection.particles['Particle Type']
    finder = CutoffNeighborFinder(cutoff=r_cut,
                                  data_collection=ov_data_collection)
    types_match = lambda a,b: (a == b) == same_species
    # retrieve neighbor distances from ovito finder (all elements may
    # be different lengths, shape: list len = n_atoms, each el len = ?
    distances_list = [
        np.array( [neigh.distance for neigh in finder.find(iatom) 
                   if types_match(types[neigh.index], types[iatom])] )
        for iatom in range(ov_data_collection.particles.count)
    ]
    #distances_list = [
    #    np.array( [neigh.distance for neigh in finder.find(iatom)] )
    #    for iatom in range(ov_data_collection.particles.count)
    #]
    max_len = max([len(l) for l in distances_list])
    padded_list = [
        np.pad(l, (0, max_len - len(l)), mode='constant', constant_values=pad)
        for l in distances_list
    ]
    distances = np.stack(padded_list, axis=0) # shape: (n_atoms, max_len)
    return distances
