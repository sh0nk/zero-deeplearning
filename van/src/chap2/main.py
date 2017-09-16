#!/usr/bin/env python3

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
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
