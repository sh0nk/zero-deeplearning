#!/usr/bin/env python3
import numpy as np
from chap5.layer_native import *
from chap7.layers import *
from collections import OrderedDict

class SimpleConvNet:
  """
  >>> x = np.random.rand(2, 1, 12, 12)
  >>> t = np.random.rand(2, 10)
  >>> network = SimpleConvNet(input_dim=(1, 12, 12), \
                     conv_param={"filter_num": 4, "filter_size": 5, "pad": 0, "stride": 1}, \
                     hidden_size=20)
  >>> network.predict(x).shape
  (2, 10)
  >>> softmax, loss = network.loss(x, t)
  >>> softmax.shape
  (2, 10)
  >>> type(loss)
  <class 'numpy.float64'>
  >>> network.gradient(x, t)["W1"].shape
  (4, 1, 5, 5)
  >>> network.gradient(x, t)["b1"].shape
  (4,)
  >>> network.gradient(x, t)["W2"].shape
  (64, 20)
  >>> network.gradient(x, t)["b2"].shape
  (20,)
  >>> network.gradient(x, t)["W3"].shape
  (20, 10)
  >>> network.gradient(x, t)["b3"].shape
  (10,)
  >>> 0.0 <= network.accuracy(x, t) <= 1.0
  True
  """
  def __init__(self, input_dim=(1, 28, 28),
                     conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
                     hidden_size = 100, output_size = 10, weight_init_std = 0.01):
    filter_num = conv_param["filter_num"]
    filter_size = conv_param["filter_size"]
    filter_pad = conv_param["pad"]
    filter_stride = conv_param["stride"]
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
    pool_output_size = (filter_num * (conv_output_size // 2) * (conv_output_size // 2))

    self.params = {}
    self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    self.params["b1"] = np.zeros(filter_num)
    self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
    self.params["b2"] = np.zeros(hidden_size)
    self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params["b3"] = np.zeros(output_size)

    self.layers = OrderedDict()
    self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], filter_stride, filter_pad)
    self.layers["Relu1"] = Relu()
    self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
    self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
    self.layers["Relu2"] = Relu()
    self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    return x

  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)

  def gradient(self, x, t):
    self.loss(x, t)

    dout = 1
    dout = self.lastLayer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    grads = {}
    grads["W1"] = self.layers["Conv1"].dW
    grads["b1"] = self.layers["Conv1"].db
    grads["W2"] = self.layers["Affine1"].dW
    grads["b2"] = self.layers["Affine1"].db
    grads["W3"] = self.layers["Affine2"].dW
    grads["b3"] = self.layers["Affine2"].db
    return grads

  def accuracy(self, x, t, batch_size=100):
    if t.ndim != 1 : t = np.argmax(t, axis=1)

    acc = 0.0

    for i in range(int(x.shape[0] / batch_size)):
      tx = x[i*batch_size:(i+1)*batch_size]
      tt = t[i*batch_size:(i+1)*batch_size]
      y = self.predict(tx)
      y = np.argmax(y, axis=1)
      acc += np.sum(y == tt)

    return acc / x.shape[0]

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
  print("start mnist training")
  from chap6.optimizers import SGD
  from lib.mnist import load_mnist

  (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, flatten=False)
  optimizer = SGD()

  max_epochs = 201
  train_size = x_train.shape[0]
  batch_size = 100

  def train(network):
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    print(iter_per_epoch)
    epoch_cnt = 0

    for i in range(1000000000):
      batch_mask = np.random.choice(train_size, batch_size)
      x_batch = x_train[batch_mask]
      t_batch = t_train[batch_mask]

      grads = network.gradient(x_batch, t_batch)
      optimizer.update(network.params, grads)

      train_loss_list.append(network.loss(x_batch, t_batch))

      if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("Accuracy: {}, {}".format(train_acc, test_acc))
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
          break
    return train_loss_list, train_acc_list, test_acc_list

  network = SimpleConvNet()
  (train_loss_list, train_acc_list, test_acc_list) = train(network)
  print(train_acc_list[-1], test_acc_list[-1])
