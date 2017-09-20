#!/usr/bin/env python3
import numpy as np

def numerical_gradient(f, x):
  """  
  calc gradient(enable to deal algebra)
  >>> from chap4.simplenet import SimpleNet
  >>> net = SimpleNet()
  >>> x = np.array([1.0, 0.8])
  >>> t = np.array([0, 0, 1])
  >>> f = lambda W: net.loss(x, t)
  >>> numerical_gradient(f, net.W).shape
  (2, 3)
  """

  h = 1e-4
  grad = np.zeros_like(x)

  # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.nditer.html
  it = np.nditer(grad, flags=["multi_index"])
  while not it.finished:
    idx = it.multi_index
    val = x[idx]
    x[idx] = val + h
    fxh1 = f(x)
    x[idx] = val - h
    fxh2 = f(x)
    grad[idx] = (fxh1 - fxh2) / (2 * h)
    x[idx] = val
    it.iternext()
  return grad

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
