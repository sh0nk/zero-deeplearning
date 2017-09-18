#!/usr/bin/env python3
import numpy as np
import pickle
import os, sys
#sys.path.append(os.pardir)
from lib.mnist import load_mnist
from chap3.activation_functions import *

def get_data():
  """
  >>> x_test, t_test = get_data()
  >>> np.shape(x_test)
  (10000, 784)
  >>> np.shape(t_test)
  (10000,)
  """
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test

def init_network():
  """
  >>> network = init_network()
  >>> "W1" in network
  True
  >>> np.shape(network["W1"])
  (784, 50)
  >>> "W2" in network
  True
  >>> np.shape(network["W2"])
  (50, 100)
  >>> "W3" in network
  True
  >>> np.shape(network["W3"])
  (100, 10)
  >>> "b1" in network
  True
  >>> np.shape(network["b1"])
  (50,)
  >>> "b2" in network
  True
  >>> np.shape(network["b2"])
  (100,)
  >>> "b3" in network
  True
  >>> np.shape(network["b3"])
  (10,)
  """
  d = os.path.dirname(__file__)
  f = "sample_weight.pkl"
  with open(d + os.sep + f if len(d) > 0 else f, "rb") as f:
    network = pickle.load(f)
  return network

def predict(network, x):
  W1, W2, W3 = network["W1"], network["W2"], network["W3"]
  b1, b2, b3 = network["b1"], network["b2"], network["b3"]

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  return softmax(a3)

def main():
  """
  >>> main()
  Accuracy:0.9352
  """
  x, t = get_data()
  network = init_network()
  accuracy_cnt = 0
  for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y)
    if p == t[i]:
      accuracy_cnt += 1

  print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
