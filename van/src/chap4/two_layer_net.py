#!/usr/bin/env python3
import numpy as np
from chap3.activation_functions import sigmoid, softmax
from chap4.loss_functions import cross_entropy_error
from chap4.gradient import numerical_gradient

class TwoLayerNet:
  """
  >>> net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
  >>> net.params["W1"].shape
  (784, 100)
  >>> net.params["b1"].shape
  (100,)
  >>> net.params["W2"].shape
  (100, 10)
  >>> net.params["b2"].shape
  (10,)
  >>> x = np.random.rand(100, 784)
  >>> t = np.random.rand(100, 10)
  >>> net.predict(x).shape
  (100, 10)
  >>> type(net.loss(x, t))
  <class 'numpy.float64'>
  >>> 0.0 <= net.accuracy(x, t) <= 1.0
  True
  >>> grads = net.numerical_gradient(x[0:10], t[0:10])
  >>> grads["W1"].shape
  (784, 100)
  >>> grads["b1"].shape
  (100,)
  >>> grads["W2"].shape
  (100, 10)
  >>> grads["b2"].shape
  (10,)
  """

  def __init__ (self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # init weight
    self.params = {}
    self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params["b1"] = np.zeros(hidden_size)
    self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params["b2"] = np.zeros(output_size)

  def predict(self, x):
    a1 = np.dot(x, self.params["W1"]) + self.params["b1"]
    z1 = sigmoid(a1)
    a2 = np.dot(z1, self.params["W2"]) + self.params["b2"]
    return softmax(a2)

  def loss(self, x, t):
    y = self.predict(x)
    return cross_entropy_error(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
    grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
    grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
    grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
    return grads

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
