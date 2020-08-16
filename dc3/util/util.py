'''
NOTE: util/constants.py imports functions from this module => do not import other
custom modules! (this should have no custom dependencies)
'''
from collections import namedtuple

Lattice = namedtuple('Lattice', ['name', 'perfect_path', 'neigh_range'])

def range_list_max(range_list):
  return max([end for start, end in range_list])

def n_neighs_from_lattices(lattices):
  n_neighs = list(set([tuple(l.neigh_range) for l in lattices]))
  n_neighs.sort()
  return n_neighs
