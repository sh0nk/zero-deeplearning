#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pylab as plt

def step_function(x):
  """
  >>> step_function(-1)
  array(0)
  >>> step_function(0)
  array(0)
  >>> step_function(1)
  array(1)
  >>> step_function(np.array([-4, -0.1, 0, 0.1, 5]))
  array([0, 0, 0, 1, 1])
  """
  return np.array(x > 0, dtype=np.int)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
  x = np.arange(-5.0, 5.0, 0.1)
  y = step_function(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 1.1)
  filename = "step_function.png"
  plt.savefig(filename)
