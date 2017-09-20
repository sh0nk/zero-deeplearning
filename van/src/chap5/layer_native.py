#!/usr/bin/env python3

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

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()
