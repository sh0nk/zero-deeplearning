#!/usr/bin/env python3
import numpy as np

def mean_squared_error(x, t):
  """
  loss function

  >>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,]
  >>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
  >>> mean_squared_error(y, t)
  Traceback (most recent call last):
  ...
  TypeError: unsupported operand type(s) for -: 'list' and 'list'
  >>> mean_squared_error(np.array(y), np.array(t))
  0.097500000000000031
  >>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]
  >>> mean_squared_error(np.array(y), np.array(t))
  0.72250000000000003
  """
  return 0.5 * np.sum((x - t) ** 2)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
