#!/usr/bin/env python3
import numpy as np

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

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
