#!/usr/bin/env python3
import numpy as np

def numerical_diff(f, x):
  """
  Approximate dirivative function

  >>> numerical_diff(__test_function_1, 5)
  0.1999999999990898
  >>> numerical_diff(__test_function_1, 10)
  0.2999999999986347
  >>> numerical_diff(__test_function_2([None, 4.0]), 3.0)
  6.00000000000378
  >>> numerical_diff(__test_function_2([3.0, None]), 4.0)
  7.999999999999119
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

def __test_function_2(x):
  """
  funcstion with 2 variables

  >>> __test_function_2([0.0, 0.0])
  0.0
  >>> __test_function_2([0.0, 2.0])
  4.0
  >>> __test_function_2([2.5, 0.0])
  6.25
  >>> __test_function_2([2.0, 2.5])
  10.25
  >>> __test_function_2([5, None])(2)
  29
  >>> __test_function_2([None, 3])(4)
  25
  >>> __test_function_2([None, None])(4)
  Traceback (most recent call last):
  ...
  Exception: At least one parameter is needed
  """
  
  x0, x1 = x[0], x[1]
  if x0 == None:
    if x1 == None:
      raise Exception("At least one parameter is needed")
    return lambda t: t ** 2 + x1 ** 2
  elif x1 == None:
    return lambda t: x0 ** 2 + t ** 2
  else:
    return x0 ** 2 + x1 ** 2

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
