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
    >>> params['test']
    array([[ 1.995,  0.992],
           [-2.01 ,  0.495]])
    >>> params['test2']
    array([ 0.17 ,  0.385])
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
    >>> params['test']
    array([[ 1.995,  0.992],
           [-2.01 ,  0.495]])
    >>> params['test2']
    array([ 0.17 ,  0.385])
    """
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)

    for key in params.keys():
      self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
      params[key] += self.v[key]

class AdaGrad:
  def __init__(self, lr=0.01):
    """
    >>> opt = AdaGrad(lr=1.0)
    >>> opt.lr
    1.0
    """
    self.lr = lr
    self.h = None

  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> AdaGrad().update(params, grads)
    >>> params['test']
    array([[ 1.99,  0.99],
           [-2.01,  0.49]])
    >>> params['test2']
    array([ 0.19,  0.39])
    """
    if self.h is None:
      self.h = {}
      for key, val in params.items():
        self.h[key] = np.zeros_like(val)
    for key in params.keys():
      self.h[key] += grads[key] * grads[key]
      params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
