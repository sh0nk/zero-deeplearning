#!/usr/bin/env python3
import numpy as np

def mean_squared_error(y, t):
  """
  loss function

  >>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,]
  >>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
  >>> mean_squared_error(y, t)
  Traceback (most recent call last):
  ...
  AttributeError: 'list' object has no attribute 'ndim'
  >>> mean_squared_error(np.array(y), np.array(t))
  0.097500000000000031
  >>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]
  >>> mean_squared_error(np.array(y), np.array(t))
  0.72250000000000003
  >>> y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], \
           [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]]
  >>> mean_squared_error(np.array(y), np.array(t))
  0.41000000000000003
  """
  if y.ndim == 1:
    y = y.reshape(1, y.size)
    t = t.reshape(1, t.size)
  batch_size = y.shape[0] 
  return 0.5 * np.sum((y - t) ** 2) / batch_size

def cross_entropy_error(y, t):
  """
  loss function(cross entropy)
  if y is 0, np.log(y) will be -inf.
  In this case, we cannot precise value because of this special num.
  For preventing it, adding delta to y.

  >>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,]
  >>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
  >>> cross_entropy_error(y, t)
  Traceback (most recent call last):
  ...
  AttributeError: 'list' object has no attribute 'ndim'
  >>> cross_entropy_error(np.array(y), np.array(t))
  0.51082545709933802
  >>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]
  >>> cross_entropy_error(np.array(y), np.array(t))
  2.3025840929945458
  >>> y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], \
           [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]]
  >>> cross_entropy_error(np.array(y), np.array(t))
  1.4067047750469419
  """
  delta = 1e-7
  if y.ndim == 1:
    y = y.reshape(1, y.size)
    t = t.reshape(1, t.size)
  batch_size = y.shape[0] 
  return -np.sum(t * np.log(y + delta)) / batch_size

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
