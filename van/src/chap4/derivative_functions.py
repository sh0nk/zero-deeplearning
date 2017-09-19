#!/usr/bin/env python3
import numpy as np

def numerical_diff(f, x):
  """
  Approximate dirivative function

  >>> numerical_diff(__test_function_1, 5)
  0.1999999999990898
  >>> numerical_diff(__test_function_1, 10)
  0.2999999999986347
  """
  h = 1e-4 #0.0001
  return (f(x + h) - f(x - h)) / (2 * h)

def __test_function_1(x):
  """
  function for test

  >>> __test_function_1(1)
  0.11
  """
  return 0.01 * x ** 2 + 0.1 * x

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pylab as plt
  x = np.arange(0.0, 20.0, 0.1)
  # draw __test_function_1
  y = __test_function_1(x)
  plt.plot(x, y)
  plt.xlim(0.0, 20.0)
  plt.ylim(-1.0, 6.0)
  # draw derivative lines
  xs = [5, 10]
  colors = ["orange", "forestgreen"]
  for i in range(2):
    tx = xs[i]
    c = colors[i]
    ty = __test_function_1(tx)
    plt.plot(x, numerical_diff(__test_function_1, tx) * (x - tx) + ty, color=c)
    plt.plot(tx, ty, "o", color=c)
    plt.plot([0.0, tx], [ty, ty], "--", color=c)
    plt.plot([tx, tx], [-1.0, ty], "--", color=c)
  plt.savefig("derivative_function.png")
