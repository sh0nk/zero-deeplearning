#!/usr/bin/env python3
import numpy as np
from lib.utils import im2col, col2im

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
  >>> layer = Convolution(W, b, stride, pad)
  >>> layer.forward(x).shape == (N, FN, out_h, out_w)
  True
  >>> dout = np.random.rand(N, FN, out_h, out_w)
  >>> layer.backward(dout).shape == (N, C, width, height)
  True
  """
  def __init__(self, W, b, stride=1, pad=0):
    self.W = W # filter
    self.b = b # bias
    self.stride = stride
    self.pad = pad

    self.x = None   
    self.col = None
    self.col_W = None

    self.dW = None
    self.db = None

  def forward(self, x):
    FN, C, FH, FW = self.W.shape
    N, C, H, W = x.shape
    out_h = (H + 2 * self.pad - FH) // self.stride + 1
    out_w = (W + 2 * self.pad - FW) // self.stride + 1
    col = im2col(x, FH, FW, self.stride, self.pad) # (N * out_h * out_w, C * FH * FW)
    col_W = self.W.reshape(FN, -1).T # (C * FH * FW, FN)
    out = np.dot(col, col_W) + self.b
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    self.x = x
    self.col = col
    self.col_W = col_W

    return out

  def backward(self, dout):
    FN, C, FH, FW = self.W.shape
    dout = dout.transpose(0,2,3,1).reshape(-1, FN)

    self.db = np.sum(dout, axis=0)
    self.dW = np.dot(self.col.T, dout)
    self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

    dcol = np.dot(dout, self.col_W.T)
    dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

    return dx

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
  >>> layer = Pooling(pool_h, pool_w, stride)
  >>> layer.forward(x).reshape(-1)
  array([ 2.,  4.,  3.,  4.,  4.,  6.,  3.,  3.,  4.,  4.,  4.,  6.])
  >>> dout = np.random.rand(1, 3, 2, 2)
  >>> layer.backward(dout).shape == (1, 3, 4, 4)
  True
  """
  def __init__(self, pool_h, pool_w, stride=1):
    self.pool_h = pool_h
    self.pool_w = pool_w
    self.stride = stride

    self.x = None
    self.arg_max = None

  def forward(self, x):
    N, C, H, W = x.shape
    out_h = (H - self.pool_h) // self.stride + 1
    out_w = (W - self.pool_w) // self.stride + 1
    col = im2col(x, self.pool_h, self.pool_w, self.stride, 0)
    col = col.reshape(-1, self.pool_h * self.pool_w)
    out = np.max(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    self.x = x
    self.arg_max = np.argmax(col, axis=1)

    return out

  def backward(self, dout):
    dout = dout.transpose(0, 2, 3, 1)
    
    pool_size = self.pool_h * self.pool_w
    dmax = np.zeros((dout.size, pool_size))
    dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,)) 
    
    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride)
    
    return dx

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
