#!/usr/bin/env python3
class SGD:
  def __init__(self, lr=0.01):
    """
    >>> SGD(lr=1.0).lr
    1.0
    """
    self.lr = lr

  def update(self, params, grads):
    """
    >>> import numpy as np
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> SGD().update(params, grads)
    >>> params
    {'test': array([[ 1.995,  0.992],
           [-2.01 ,  0.495]]), 'test2': array([ 0.17 ,  0.385])}
    """
    for key in params:
      params[key] -= self.lr * grads[key]

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
