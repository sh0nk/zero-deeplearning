#!/usr/bin/env python3
import numpy as np
from chap3.activation_functions import softmax
from chap4.loss_functions import cross_entropy_error
from chap4.gradient import numerical_gradient

class SimpleNet:
  """
  simple neural network for studying learning network

  >>> network = SimpleNet()
  >>> network.W.shape
  (2, 3)
  >>> x = np.array([1.0, 0.8])
  >>> network.predict(x).shape
  (3,)
  >>> x = np.array([[1.0, 0.8], [2.2, 0.6]])
  >>> network.predict(x).shape
  (2, 3)
  >>> t = np.array([[1, 0, 0], [0, 0, 1]])
  >>> type(network.loss(x, t))
  <class 'numpy.float64'>

  def f(W):
  return net.loss(x, t)
  dW = numerical_gradient(f, net.W)
  """
  def __init__(self):
    self.W = np.random.randn(2, 3)

  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    return cross_entropy_error(y, t)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
