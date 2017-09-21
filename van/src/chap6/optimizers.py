#!/usr/bin/env python3
import numpy as np

class SGD:
  def __init__(self, lr=0.01):
    """
    >>> SGD(lr=1.0).lr
    1.0
    """
    self.lr = lr

  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> SGD().update(params, grads)
    >>> params
    {'test': array([[ 1.995,  0.992],
           [-2.01 ,  0.495]]), 'test2': array([ 0.17 ,  0.385])}
    """
    for key in params:
      params[key] -= self.lr * grads[key]

class Momentum:
  def __init__(self, lr=0.01, momentum=0.9):
    """
    >>> opt = Momentum(lr=1.0, momentum=0.5)
    >>> opt.lr
    1.0
    >>> opt.momentum
    0.5
    """
    self.lr = lr
    self.momentum = momentum
    self.v = None

  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> Momentum().update(params, grads)
    >>> params
    {'test': array([[ 1.995,  0.992],
           [-2.01 ,  0.495]]), 'test2': array([ 0.17 ,  0.385])}
    """
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)

    for key in params.keys():
      self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
      params[key] += self.v[key]

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
