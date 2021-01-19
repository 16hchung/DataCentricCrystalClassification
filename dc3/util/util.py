'''
NOTE: util/constants.py imports functions from this module => do not import other
custom modules! (this should have no custom dependencies)
'''
import numpy as np
from collections import namedtuple
from inspect import signature, Parameter
from sklearn import metrics

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

def split_kwargs_among_fxns(*fxns, **all_kwargs):
  fxn_arg_pairs = []
  for fxn in fxns:
    fxn_params = signature(fxn).parameters
    kwargs = { arg_name: arg_val 
               for arg_name, arg_val in all_kwargs.items()
               if arg_name in fxn_params and
                  fxn_params[arg_name].kind == Parameter.POSITIONAL_OR_KEYWORD }
    fxn_arg_pairs.append( (fxn, kwargs) )
  return fxn_arg_pairs

def generate_arg_combos(**kwargs):
  exclude = [] \
            if not exclude_no_option \
            else [k for k,v in kwargs.items() if len(v) <= 1]
  keys = kwargs.keys()
  vals = kwargs.values()
  for instance in itertools.product(*vals):
    this_kwargs = dict(zip(keys, instance))
    name_kwargs = {k:v for k,v in this_kwargs.items() if k not in exclude}
    stringified = stringify_args(**name_kwargs)
    yield stringified, this_kwargs

def stringify_args(**kwargs):
  arg_strs = []
  for k,v in kwargs.items():
    arg_strs.append(f'{k}-{v}')
  return '_'.join(arg_strs)

def get_optimal_cutoff(X_pos, X_neg):
  labels = np.array([1.] * len(X_pos) + [0.] * len(X_neg))
  X = np.append(X_pos, X_neg)
  fpr, tpr, thresholds = metrics.roc_curve(labels, X)
  optimal_idx = np.argmax(tpr - fpr)
  optimal_threshold = thresholds[optimal_idx]
  return optimal_threshold

class TwoDArrayCombiner:
  '''Stacks np arrays that should be transformed together, then decomposes 
  the transformed array to be split into the correct dimensions'''

  def __init__(self, *Xs): 
    self._Xs = Xs
    # all Xs should have same # features
    d = None
    for X in Xs:
      assert len(X.shape) == 2
      assert d is None or X.shape[1] == d
      d = X.shape[1]
    self._d = d
    self._Ns = [X.shape[0] for X in Xs]

  def combined(self):
    return np.concatenate(self._Xs, axis=0)

  def decompose(self, transformedXs):
    splitXs = []
    last_idx = 0
    for N in self._Ns:
      splitXs.append(transformedXs[last_idx:last_idx+N,:])
      last_idx += N
    assert last_idx == len(transformedXs) == sum([len(X) for X in self._Xs])
    return splitXs
