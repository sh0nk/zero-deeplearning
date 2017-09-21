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

def numerical_gradient(f, x):
  """
  gradient functions (enable to deal multi variables)

  >>> numerical_gradient(__test_function_2, np.array([3.0, 4.0]))
  array([ 6.,  8.])
  >>> numerical_gradient(__test_function_2, np.array([0.0, 2.0]))
  array([ 0.,  4.])
  >>> numerical_gradient(__test_function_2, np.array([3.0, 0.0]))
  array([ 6.,  0.])
  """
  h = 1e-4
  grad = np.zeros_like(x)

  for idx in range(x.size):
    val = x[idx]
    x[idx] = val + h
    fxh1 = f(x)
    x[idx] = val - h
    fxh2 = f(x)
    grad[idx] = (fxh1 - fxh2) / (2 * h)
    x[idx] = val #restore value
  return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
  """
  implementation of gradient descent.

  >>> init_x = np.array([-3.0, 4.0])
  >>> gradient_descent(__test_function_2, init_x, lr=0.1)
  array([ -6.11110793e-10,   8.14814391e-10])
  >>> gradient_descent(__test_function_2, init_x, lr=10.0)
  array([ -2.58983747e+13,  -1.29524862e+12])
  >>> gradient_descent(__test_function_2, init_x, lr=1e-10)
  array([-2.99999994,  3.99999992])
  """
  x = np.copy(init_x)
  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x

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

  # draw gradient map
  plt.clf()
  plt.xlim(-2.0, 2.0)
  plt.ylim(-2.0, 2.0)
  x0, x1 = np.mgrid[-2.0:2.1:0.25, -2.0:2.1:0.25]
  x0 = x0.flatten()
  x1 = x1.flatten()
  g0, g1 = np.copy(x0), np.copy(x1)
  for idx in range(x0.size):
    grad = numerical_gradient(__test_function_2, np.array([x0[idx], x1[idx]])) 
    g0[idx] -= grad[0]
    g1[idx] -= grad[1]
  plt.quiver(x0, x1, g0, g1, angles="xy", scale_units="xy", scale=10)
  plt.grid(which='major',color='lightgray',linestyle='--', linewidth=1, alpha=90)
  plt.savefig("gradient_map.png")

  # draw gradient descent
  plt.clf()
  x = np.array([-3.0, 4.0])
  plt.plot(x[0], x[1], "o", color="steelblue")
  for _ in range(1000):
    x = gradient_descent(__test_function_2, x, lr=0.1, step_num=1)
    plt.plot(x[0], x[1], "o", color="steelblue")
  n = 100
  x = np.linspace(-3.5, 3.5, n)
  y = np.linspace(-4, 4, n)
  X, Y = np.meshgrid(x, y)

  Z = X**2 + Y**2 - 1
  plt.contour(X, Y, Z, levels=[0], colors="lightgray", linestyles="dashed")
  Z = X**2 + Y**2 - 4
  plt.contour(X, Y, Z, levels=[0], colors="lightgray", linestyles="dashed")
  Z = X**2 + Y**2 - 9
  plt.contour(X, Y, Z, levels=[0], colors="lightgray", linestyles="dashed")
  Z = X**2 + Y**2 - 16
  plt.contour(X, Y, Z, levels=[0], colors="lightgray", linestyles="dashed")
  plt.xlim(-3.5, 3.5)
  plt.ylim(-4.5, 4.5)
  plt.savefig("gradient_descent.png")
