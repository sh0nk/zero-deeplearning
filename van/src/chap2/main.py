#!/usr/bin/env python3
import numpy as np

def perceptron(x, w, b):
  if np.sum(w*x) + b <= 0:
    return 0
  else:
    return 1

def AND(x1, x2):
  """
  simple logic circuit (AND)

  >>> AND(0, 0)
  0
  >>> AND(1, 0)
  0
  >>> AND(0, 1)
  0
  >>> AND(1, 1)
  1
  """
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  return perceptron(x, w, b)

def NAND(x1, x2):
  """
  simple logic circuit (NAND)

  >>> NAND(0, 0)
  1
  >>> NAND(1, 0)
  1
  >>> NAND(0, 1)
  1
  >>> NAND(1, 1)
  0
  """
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.7
  return perceptron(x, w, b)

def OR(x1, x2):
  """
  simple logic circuit (OR)

  >>> OR(0, 0)
  0
  >>> OR(1, 0)
  1
  >>> OR(0, 1)
  1
  >>> OR(1, 1)
  1
  """
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2
  return perceptron(x, w, b)

def XOR(x1, x2):
  """
  logic circuit (XOR)

  >>> XOR(0, 0)
  0
  >>> XOR(1, 0)
  1
  >>> XOR(0, 1)
  1
  >>> XOR(1, 1)
  0
  """
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  return AND(s1, s2)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
