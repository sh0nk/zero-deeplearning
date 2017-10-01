#!/usr/bin/env python3

import numpy as np
from chap4.gradient import numerical_gradient
from chap5.layer_native import *
from chap6.dropout import Dropout
from collections import OrderedDict

class MultiLayerNet:
  def __init__(self, input_size, hidden_size_list, output_size, weight_decay_lambda=0.0, use_dropout=False, dropout_ratio=0.5):
    """
    >>> net = MultiLayerNet(30, [20], 10, use_dropout=True)
    >>> net.params["W1"].shape
    (30, 20)
    >>> net.params["b1"].shape
    (20,)
    >>> net.params["W2"].shape
    (20, 10)
    >>> net.params["b2"].shape
    (10,)
    >>> len(net.layers)
    4
    >>> type(net.layers["Affine1"])
    <class 'chap5.layer_native.Affine'>
    >>> type(net.layers["Relu1"])
    <class 'chap5.layer_native.Relu'>
    >>> type(net.layers["Dropout1"])
    <class 'chap6.dropout.Dropout'>
    >>> type(net.layers["Affine2"])
    <class 'chap5.layer_native.Affine'>
    >>> type(net.lastLayer)
    <class 'chap5.layer_native.SoftmaxWithLoss'>
    """
    self.weight_decay_lambda = weight_decay_lambda
    self.use_dropout = use_dropout

    size_list = [input_size] + hidden_size_list + [output_size]
    self.params = {}
    for i in range(1, len(size_list)):
      self.params["W{}".format(i)] = np.random.randn(size_list[i - 1], size_list[i]) * np.sqrt(2 / size_list[i - 1])
      self.params["b{}".format(i)] = np.random.randn(size_list[i]) * np.sqrt(2 / size_list[i - 1])

    # layer
    size_list = [input_size] + hidden_size_list
    self.layers = OrderedDict()
    for i in range(1, len(size_list)):
      self.layers["Affine{}".format(i)] = Affine(self.params["W{}".format(i)], self.params["b{}".format(i)])
      self.layers["Relu{}".format(i)] = Relu()
      if self.use_dropout:
        self.layers["Dropout{}".format(i)] = Dropout(dropout_ratio)
    # add for output layer
    i = len(size_list)
    self.layers["Affine{}".format(i)] = Affine(self.params["W{}".format(i)], self.params["b{}".format(i)])

    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x, is_training=False):
    """
    >>> x = np.random.randn(10, 30)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> net.predict(x).shape
    (10, 10)
    >>> net = MultiLayerNet(30, [20], 10, use_dropout=True)
    >>> net.predict(x).shape
    (10, 10)
    >>> net = MultiLayerNet(30, [20], 10, use_dropout=True)
    >>> net.predict(x, is_training=True).shape
    (10, 10)
    """
    for key, layer in self.layers.items():
      if "Dropout" in key:
        x = layer.forward(x, is_training)
      else:
        x = layer.forward(x)
    return x

  def loss(self, x, t, is_training=False):
    """
    >>> x = np.random.randn(10, 30)
    >>> t = np.random.randn(10, 10)
    >>> net = MultiLayerNet(30, [20], 10)
    >>> type(net.loss(x, t))
    <class 'numpy.float64'>
    """
    weight_decay = 0.0
    for i in range(1, len(self.params) // 2 + 1):
      W = self.params["W{}".format(i)]
      weight_decay += np.sum(W * W)
    weight_decay = self.weight_decay_lambda * weight_decay / 2
    y = self.predict(x, is_training)
    _, loss = self.lastLayer.forward(y, t)
    return loss + weight_decay

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
    >>> grad_backprop = network.gradient(x_batch, t_batch)
    >>> len(grad_backprop)
    4
    """
    # forward
    self.loss(x, t, is_training=True)

    # backward
    dout = self.lastLayer.backward(1)
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    # set gradient
    grads = {}
    for i in range(1, len(self.params) // 2 + 1):
      layer = self.layers["Affine{}".format(i)]
      grads["W{}".format(i)] = layer.dW + self.weight_decay_lambda * layer.W
      grads["b{}".format(i)] = layer.db
    return grads

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
