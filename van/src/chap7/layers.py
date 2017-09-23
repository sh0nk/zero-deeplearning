#!/usr/bin/env python3
import numpy as np
from lib.utils import im2col

class Convolution:
  """
  >>> N = 10
  >>> C = 2
  >>> width, height = (20, 20)
  >>> FN = 5
  >>> FW, FH = (4, 4)
  >>> x = np.random.rand(N, C, height, width)
  >>> W = np.random.rand(FN, C, FH, FW)
  >>> b = np.random.rand(FN)
  >>> stride = 1
  >>> pad = 1
  >>> out_h = (height + 2 * pad - FH) // stride + 1
  >>> out_w = (width  + 2 * pad - FW) // stride + 1
  >>> Convolution(W, b, stride, pad).forward(x).shape == (N, FN, out_h, out_w)
  True
  """
  def __init__(self, W, b, stride=1, pad=0):
    self.W = W # filter
    self.b = b # bias
    self.stride = stride
    self.pad = pad

  def forward(self, x):
    FN, C, FH, FW = self.W.shape
    N, C, H, W = x.shape
    out_h = (H + 2 * self.pad - FH) // self.stride + 1
    out_w = (W + 2 * self.pad - FW) // self.stride + 1
    col = im2col(x, FH, FW, self.stride, self.pad) # (N * out_h * out_w, C * FH * FW)
    col_W = self.W.reshape(FN, -1).T # (C * FH * FW, FN)
    out = np.dot(col, col_W) + self.b
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    return out

class Pooling:
  """
  >>> x = np.array([[ \
    [                 \
      [1, 2, 3, 0],   \
      [0, 1, 2, 4],   \
      [1, 0, 4, 2],   \
      [3, 2, 0, 1]    \
    ],                \
    [                 \
      [3, 0, 6, 5],   \
      [4, 2, 4, 3],   \
      [3, 0, 1, 0],   \
      [2, 3, 3, 1]    \
    ],                \
    [                 \
      [4, 2, 1, 2],   \
      [0, 1, 0, 4],   \
      [3, 0, 6, 2],   \
      [4, 2, 4, 5]    \
    ]                 \
  ]])
  >>> pool_h = 2
  >>> pool_w = 2
  >>> stride = 2
  >>> Pooling(pool_h, pool_w, stride).forward(x).reshape(-1)
  array([ 2.,  4.,  3.,  4.,  4.,  6.,  3.,  3.,  4.,  4.,  4.,  6.])
  """
  def __init__(self, pool_h, pool_w, stride=1):
    self.pool_h = pool_h
    self.pool_w = pool_w
    self.stride = stride

  def forward(self, x):
    N, C, H, W = x.shape
    out_h = (H - self.pool_h) // self.stride + 1
    out_w = (W - self.pool_w) // self.stride + 1
    col = im2col(x, self.pool_h, self.pool_w, self.stride, 0)
    col = col.reshape(-1, self.pool_h * self.pool_w)
    out = np.max(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
    return out

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
