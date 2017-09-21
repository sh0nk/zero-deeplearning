#!/usr/bin/env python3

import numpy as np
from chap4.gradient import numerical_gradient
from chap5.layer_native import *
from collections import OrderedDict

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    """
    >>> net = TwoLayerNet(30, 20, 10)
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
    self.params = {}
    self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params["b1"] = weight_init_std * np.random.randn(hidden_size)
    self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params["b2"] = weight_init_std * np.random.randn(output_size)

    # layer
    self.layers = OrderedDict()
    self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
    self.layers["Relu1"] = Relu()
    self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    """
    >>> x = np.random.randn(10, 30)
    >>> net = TwoLayerNet(30, 20, 10)
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
    >>> net = TwoLayerNet(30, 20, 10)
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
    >>> net = TwoLayerNet(30, 20, 10)
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
    >>> net = TwoLayerNet(30, 20, 10)
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
    grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
    grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
    grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
    grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
    return grads

  def gradient(self, x, t):
    """
    fast calculation of gradient

    >>> x = np.random.randn(10, 30)
    >>> t = np.random.randn(10, 10)
    >>> net = TwoLayerNet(30, 20, 10)
    >>> grads = net.gradient(x, t)
    >>> grads["W1"].shape == net.params["W1"].shape
    True
    >>> grads["b1"].shape == net.params["b1"].shape
    True
    >>> grads["W2"].shape == net.params["W2"].shape
    True
    >>> grads["b2"].shape == net.params["b2"].shape
    True
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
    grads["W1"] = self.layers["Affine1"].dW
    grads["b1"] = self.layers["Affine1"].db
    grads["W2"] = self.layers["Affine2"].dW
    grads["b2"] = self.layers["Affine2"].db
    return grads

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
