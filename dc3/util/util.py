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

def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    from: https://stevenloria.com/lazy-properties/
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

