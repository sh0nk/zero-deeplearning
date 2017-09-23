#!/usr/bin/env python3
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
  # get shape of input data
  N, C, H, W = input_data.shape
  # calc output shape
  out_h = (H + (2 * pad) - filter_h) // stride + 1
  out_w = (W + (2 * pad) - filter_w) // stride + 1
  
  # padding data
  # not to pad about N and C (only about H and W)
  img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

  # make column for convolution
  col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
  for y in range(filter_h):
    y_max = y + stride * out_h
    for x in range(filter_w):
      x_max = x + stride * out_w
      # set value for each filter in each stride
      col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

  # reshape colmun
  col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
  return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
  """
  Copied from https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/util.py
  Used for backward
  """
  N, C, H, W = input_shape
  out_h = (H + 2*pad - filter_h)//stride + 1
  out_w = (W + 2*pad - filter_w)//stride + 1
  col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

  img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
  for y in range(filter_h):
      y_max = y + stride*out_h
      for x in range(filter_w):
          x_max = x + stride*out_w
          img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

  return img[:, :, pad:H + pad, pad:W + pad]
