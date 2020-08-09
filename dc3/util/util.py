'''
NOTE: util/constants.py imports functions from this module => do not import other
custom modules! (this should have no custom dependencies)
'''

def range_list_max(range_list):
  return max([max(r) + 1 for r in range_list])
