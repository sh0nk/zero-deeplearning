import numpy as np
if __name__ == "__main__":
  from activation_functions import *
else:
  from chap3.activation_functions import *

def init_network():
  network={}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def forward(network, x):
  """
  >>> forward(init_network(), np.array([1.0, 0.5]))
  array([ 0.31682708,  0.69627909])
  """
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  return identity_function(a3)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
