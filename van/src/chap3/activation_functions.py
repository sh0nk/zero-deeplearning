#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pylab as plt

def step_function(x):
  """
  >>> step_function(-1)
  array(0)
  >>> step_function(0)
  array(0)
  >>> step_function(1)
  array(1)
  >>> step_function(np.array([-4, -0.1, 0, 0.1, 5]))
  array([0, 0, 0, 1, 1])
  """
  return np.array(x > 0, dtype=np.int)

def sigmoid(x):
  """
  >>> sigmoid(0)
  0.5
  >>> sigmoid(np.array([-1.0, 1.0, 2.0]))
  array([ 0.26894142,  0.73105858,  0.88079708])
  """
  return 1 / (1 + np.exp(-x))

def relu(x):
  """
  >>> relu(-1)
  0
  >>> relu(3)
  3
  >>> relu(np.array([-3, -1, -0.1, 0, 0.1, 99]))
  array([  0. ,   0. ,   0. ,   0. ,   0.1,  99. ])
  """
  return np.maximum(0, x)

def identity_function(x):
  """
  >>> identity_function(-5)
  -5
  >>> identity_function(-1)
  -1
  >>> identity_function(0)
  0
  >>> identity_function(1)
  1
  >>> identity_function(5)
  5
  """
  return x

def softmax(a):
  """
  >>> softmax(np.array([0.3, 2.9, 4.0]))
  array([ 0.01821127,  0.24519181,  0.73659691])
  >>> softmax(123)
  1.0
  >>> softmax([0.5, 2])
  array([ 0.18242552,  0.81757448])
  """
  exp_a = np.exp(a - np.max(a))
  return exp_a / np.sum(exp_a)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
  x = np.arange(-5.0, 5.0, 0.1)
  # output step function
  y = step_function(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 1.1)
  filename = "step_function.png"
  plt.savefig(filename)
  # output sigmoid
  plt.clf()
  y = sigmoid(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 1.1)
  filename = "sigmoid.png"
  plt.savefig(filename)
  # output relu
  plt.clf()
  y = relu(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 5.1)
  filename = "relu.png"
  plt.savefig(filename)
