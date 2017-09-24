#!/usr/bin/env python3
import numpy as np
from chap3.activation_functions import softmax
from chap4.loss_functions import cross_entropy_error

class MulLayer:
  """
  multiplication layer for calc graph

  >>> apple = 100
  >>> apple_num = 2
  >>> tax = 1.1
  >>> mul_apple_layer = MulLayer()
  >>> mul_tax_layer = MulLayer()
  >>> apple_price = mul_apple_layer.forward(apple, apple_num)
  >>> price = mul_tax_layer.forward(apple_price, tax)
  >>> int(price)
  220
  >>> dprice = 1
  >>> dapple_price, dtax = mul_tax_layer.backward(dprice)
  >>> dapple, dapple_num = mul_apple_layer.backward(dapple_price)
  >>> print(dapple, int(dapple_num), int(dtax))
  2.2 110 200
  """
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    return x * y

  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    return dx, dy

class AddLayer:
  """
  addition layer for calc graph

  >>> apple = 200
  >>> orange = 150
  >>> add_fruit_layer = AddLayer()
  >>> add_fruit_layer.forward(apple, orange)
  350
  >>> dfruit = 10
  >>> add_fruit_layer.backward(dfruit)
  (10, 10)
  """
  def __init__(self):
    pass

  def forward(self, x, y):
    return x + y

  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy

def __sample():
  """
  >>> apple = 100
  >>> apple_num = 2
  >>> orange = 150
  >>> orange_num = 3
  >>> tax = 1.1

  ### layer
  >>> mul_apple_layer = MulLayer()
  >>> mul_orange_layer = MulLayer()
  >>> add_apple_orange_layer = AddLayer()
  >>> mul_tax_layer = MulLayer()

  ### forward
  >>> apple_price = mul_apple_layer.forward(apple, apple_num)
  >>> orange_price = mul_orange_layer.forward(orange, orange_num)
  >>> fruit_price = add_apple_orange_layer.forward(apple_price, orange_price)
  >>> price = mul_tax_layer.forward(fruit_price, tax)
  >>> print(apple_price, orange_price, fruit_price, int(price))
  200 450 650 715

  ### backward
  >>> dprice = 1
  >>> dfruit_price, dtax = mul_tax_layer.backward(dprice)
  >>> dapple_price, dorange_price = add_apple_orange_layer.backward(dfruit_price)
  >>> dorange, dorange_num = mul_orange_layer.backward(dorange_price)
  >>> dapple, dapple_num = mul_apple_layer.backward(dapple_price)
  >>> print(dapple, int(dapple_num), dorange, int(dorange_num), dfruit_price, dprice)
  2.2 110 3.3000000000000003 165 1.1 1

  >>> __sample()
  """
  pass

class Relu:
  """
  layer class for activation function.

  >>> x = np.array([[1.0, -0.5], [-2.0, 3.0]])
  >>> relu = Relu()
  >>> relu.forward(x)
  array([[ 1.,  0.],
         [ 0.,  3.]])
  >>> dout = np.array([[-3.0, 2.0], [1.5, 0.0]])
  >>> relu.backward(dout)
  array([[-3.,  0.],
         [ 0.,  0.]])
  """
  def __init__(self):
    self.mask = None

  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out

  def backward(self, dout):
    dx = dout.copy()
    dx[self.mask] = 0
    return dx

class Sigmoid:
  """
  >>> x = np.array([[1.0, -0.5], [-2.0, 3.0]])
  >>> sig = Sigmoid()
  >>> sig.forward(x)
  array([[ 0.73105858,  0.37754067],
         [ 0.11920292,  0.95257413]])
  >>> dout = np.array([[-3.0, 2.0], [1.5, 0.0]])
  >>> sig.backward(dout)
  array([[-0.5898358 ,  0.47000742],
         [ 0.15749038,  0.        ]])
  """
  def __init__(self):
    self.out = None

  def forward(self, x):
    self.out = 1 / (1 + np.exp(-x))
    return self.out

  def backward(self, dout):
    return dout * self.out * (1.0 - self.out)

class Affine:
  """
  Affine transition

  >>> x = np.random.randn(2, 2)
  >>> W = np.random.randn(2, 3)
  >>> b = np.random.randn(3)
  >>> aff = Affine(W, b)
  >>> out = aff.forward(x)
  >>> out.shape
  (2, 3)
  >>> dx = aff.backward(out / 10)
  >>> dx.shape == x.shape
  True
  >>> aff.dW.shape == W.shape
  True
  >>> aff.db.shape == b.shape
  True
  """
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    self.original_x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    self.x = x
    return np.dot(x, self.W) + self.b

  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)
    dx = dx.reshape(*self.original_x_shape)
    return dx

class SoftmaxWithLoss:
  """
  softmax + cros_entropy_loss

  >>> x = np.array([[2.0, 3.0, 0.5], [4.0, -0.5, 1.0]])
  >>> t = np.array([[0, 0, 1], [1, 0, 0]])
  >>> layer = SoftmaxWithLoss()
  >>> y, loss = layer.forward(x, t)
  >>> y.shape
  (2, 3)
  >>> type(loss)
  <class 'numpy.float64'>
  >>> layer.backward().shape
  (2, 3)
  """
  def __init__(self):
    self.y = None
    self.t = None

  def forward(self, x, t):
    self.y = softmax(x)
    self.t = t
    loss = cross_entropy_error(self.y, self.t)
    return self.y, loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    return (self.y - self.t) / batch_size

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
