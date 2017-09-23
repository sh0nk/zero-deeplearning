#!/usr/bin/env python3
import numpy as np

class Dropout:
  """
  >>> x = np.random.randn(2, 3)
  >>> drop = Dropout()
  >>> drop.forward(x).shape
  (2, 3)
  >>> drop.forward(x, train_flg=True).shape
  (2, 3)
  >>> drop.backward(1).shape
  (2, 3)
  """
  def __init__(self, dropout_ratio=0.5):
    self.dropout_ratio = dropout_ratio
    self.mask = None

  def forward(self, x, train_flg=False):
    if train_flg:
      self.mask = np.random.rand(*x.shape) > self.dropout_ratio
      return x * self.mask
    else:
      return x * (1.0 - self.dropout_ratio)

  def backward(self, dout):
    return dout * self.mask

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
