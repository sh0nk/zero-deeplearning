#!/usr/bin/env python3

import numpy as np
from chap4.gradient import numerical_gradient
from chap5.layer_native import *
from collections import OrderedDict

class MultiLayerNet:
  def __init__(self, input_size, hidden_size_list, output_size):
    """
    >>> net = MultiLayerNet(30, [20], 10)
    >>> net.params["W1"].shape
    (30, 20)
    >>> net.params["b1"].shape
    (20,)
    >>> net.params["W2"].shape
    (20, 10)
    >>> net.params["b2"].shape
    (10,)
    >>> len(net.layers)
    3
    >>> type(net.layers["Affine1"])
    <class 'chap5.layer_native.Affine'>
    >>> type(net.layers["Relu1"])
    <class 'chap5.layer_native.Relu'>
    >>> type(net.layers["Affine2"])
    <class 'chap5.layer_native.Affine'>
    >>> type(net.lastLayer)
    <class 'chap5.layer_native.SoftmaxWithLoss'>
    """
    size_list = [input_size] + hidden_size_list + [output_size]
    self.params = {}
    for i in range(1, len(size_list)):
      self.params["W{}".format(i)] = np.random.randn(size_list[i - 1], size_list[i]) * np.sqrt(2 / size_list[i - 1])
      self.params["b{}".format(i)] = np.random.randn(size_list[i]) * np.sqrt(2 / size_list[i - 1])

    # layer
    self.layers = OrderedDict()
    for i in range(1, len(size_list)):
      self.layers["Affine{}".format(i)] = Affine(self.params["W{}".format(i)], self.params["b{}".format(i)])
      if i + 1 < len(size_list):
        self.layers["Relu{}".format(i)] = Relu()
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    """
    >>> x = np.random.randn(10, 30)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> net.predict(x).shape
    (10, 10)
    """
    for layer in self.layers.values():
      x = layer.forward(x)
    return x

  def loss(self, x, t):
    """
    >>> x = np.random.randn(10, 30)
    >>> t = np.random.randn(10, 10)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> type(net.loss(x, t))
    <class 'numpy.float64'>
    """
    y = self.predict(x)
    _, loss = self.lastLayer.forward(y, t)
    return loss

  def accuracy(self, x, t):
    """
    >>> x = np.random.randn(10, 30)
    >>> t = np.random.randn(10, 10)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> 0.0 <= net.accuracy(x, t) <= 1.0
    True
    """
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1:
      t = np.argmax(t, axis=1)
    return np.sum(y == t) / float(x.shape[0])

  def numerical_gradient(self, x, t):
    """
    >>> x = np.random.randn(10, 30)
    >>> t = np.random.randn(10, 10)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> grads = net.numerical_gradient(x, t)
    >>> grads["W1"].shape == net.params["W1"].shape
    True
    >>> grads["b1"].shape == net.params["b1"].shape
    True
    >>> grads["W2"].shape == net.params["W2"].shape
    True
    >>> grads["b2"].shape == net.params["b2"].shape
    True
    """
    loss_W = lambda W: self.loss(x, t)
    grads = {}
    for key in self.params.keys():
      grads[key] = numerical_gradient(loss_W, self.params[key])
    return grads

  def gradient(self, x, t):
    """
    fast calculation of gradient

    >>> x = np.random.randn(10, 30)
    >>> t = np.random.randn(10, 10)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> grads = net.gradient(x, t)
    >>> grads["W1"].shape == net.params["W1"].shape
    True
    >>> grads["b1"].shape == net.params["b1"].shape
    True
    >>> grads["W2"].shape == net.params["W2"].shape
    True
    >>> grads["b2"].shape == net.params["b2"].shape
    True
    >>> from lib.mnist import load_mnist
    >>> (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    >>> network = MultiLayerNet(input_size=784, hidden_size_list=[50], output_size=10)
    >>> x_batch = x_train[:3]
    >>> t_batch = t_train[:3]
    >>> grad_numerical = network.numerical_gradient(x_batch, t_batch)
    >>> grad_backprop = network.gradient(x_batch, t_batch)
    >>> diffs = list(map(lambda idx: (idx, np.abs(grad_backprop[idx] - grad_numerical[idx])), grad_numerical.keys()))
    >>> sum(map(lambda diff: np.sum(diff[1] >= 1e-3), diffs))
    0
    """
    # forward
    self.loss(x, t)

    # backward
    dout = self.lastLayer.backward(1)
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    # set gradient
    grads = {}
    for i in range(1, int((len(self.layers) + 1) / 2) + 1):
      grads["W{}".format(i)] = self.layers["Affine{}".format(i)].dW
      grads["b{}".format(i)] = self.layers["Affine{}".format(i)].db
    return grads

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
