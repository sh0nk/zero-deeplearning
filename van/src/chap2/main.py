#!/usr/bin/env python3
import numpy as np

def AND(x1, x2):
  """
  simple logic circuit (AND)

  >>> AND(0, 0)
  0
  >>> AND(1, 0)
  0
  >>> AND(0, 1)
  0
  >>> AND(1, 1)
  1
  """
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
