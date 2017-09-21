#!/usr/bin/env python3
import numpy as np

class SGD:
  def __init__(self, lr=0.01):
    """
    >>> SGD(lr=1.0).lr
    1.0
    """
    self.lr = lr

  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> SGD().update(params, grads)
    >>> params['test']
    array([[ 1.995,  0.992],
           [-2.01 ,  0.495]])
    >>> params['test2']
    array([ 0.17 ,  0.385])
    """
    for key in params:
      params[key] -= self.lr * grads[key]

class Momentum:
  def __init__(self, lr=0.01, momentum=0.9):
    """
    >>> opt = Momentum(lr=1.0, momentum=0.5)
    >>> opt.lr
    1.0
    >>> opt.momentum
    0.5
    """
    self.lr = lr
    self.momentum = momentum
    self.v = None

  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> Momentum().update(params, grads)
    >>> params['test']
    array([[ 1.995,  0.992],
           [-2.01 ,  0.495]])
    >>> params['test2']
    array([ 0.17 ,  0.385])
    """
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)

    for key in params.keys():
      self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
      params[key] += self.v[key]

class AdaGrad:
  def __init__(self, lr=0.01):
    """
    >>> opt = AdaGrad(lr=1.0)
    >>> opt.lr
    1.0
    """
    self.lr = lr
    self.h = None

  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> AdaGrad().update(params, grads)
    >>> params['test']
    array([[ 1.99,  0.99],
           [-2.01,  0.49]])
    >>> params['test2']
    array([ 0.19,  0.39])
    """
    if self.h is None:
      self.h = {}
      for key, val in params.items():
        self.h[key] = np.zeros_like(val)
    for key in params.keys():
      self.h[key] += grads[key] * grads[key]
      params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:
  """
  Adam (http://arxiv.org/abs/1412.6980v8)

  Copied from https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/optimizer.py
  """

  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
    """
    >>> opt = Adam(lr=1.0, beta1=2.0, beta2=3.0)
    >>> opt.lr
    1.0
    >>> opt.beta1
    2.0
    >>> opt.beta2
    3.0
    """
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.iter = 0
    self.m = None
    self.v = None
      
  def update(self, params, grads):
    """
    >>> params = {"test": np.array([[2.0, 1.0], [-2.0, 0.5]]), "test2": np.array([0.2, 0.4])}
    >>> grads = {"test": np.array([[0.5, 0.8], [1.0, 0.5]]), "test2": np.array([3.0, 1.5])}
    >>> AdaGrad().update(params, grads)
    >>> params['test']
    array([[ 1.99,  0.99],
           [-2.01,  0.49]])
    >>> params['test2']
    array([ 0.19,  0.39])
    """
    if self.m is None:
      self.m, self.v = {}, {}
      for key, val in params.items():
        self.m[key] = np.zeros_like(val)
        self.v[key] = np.zeros_like(val)

    self.iter += 1
    lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         

    for key in params.keys():
      #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
      #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
      self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
      self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

      params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

      #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
      #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
      #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

def _test():
  import doctest
  doctest.testmod()

if __name__ == "__main__":
  _test()

  import matplotlib as mpl
  mpl.use("agg")
  from matplotlib import pyplot as plt
  fig, [[axL, axR], [axLL, axLR]] = plt.subplots(ncols=2, nrows=2, figsize=(10,8))

  f = lambda x, y: ((x ** 2) / 20) + (y ** 2)
  dfdx = lambda x: x / 10
  dfdy = lambda y: 2 * y
  dfdw = lambda w: {"x": dfdx(w["x"]), "y": dfdy(w["y"])}
  def plot(ax, f):
    n = 100
    x = np.linspace(-10, 10, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)
    for r in (1, 2, 3, 4, 5, 6, 7):
      Z = f(X, Y) - r
      ax.contour(X, Y, Z, levels=[0], colors="lightgray")
      #, linestyles="dashed")

  for t in [("SGD", SGD(lr=0.95), axL), ("Momentum", Momentum(lr=0.1), axR),
            ("AdaGrad", AdaGrad(lr=1.5), axLL), ("Adam", Adam(lr=0.3), axLR)]:
    optimizer = t[1]
    w = {"x": -7.0, "y": 2.0}
    w_list = [(w["x"], w["y"])]
    for i in range(30):
      dw = dfdw(w)
      optimizer.update(w, dw)
      w_list.append((w["x"], w["y"]))
    ax = t[2]
    ax.set_title(t[0])
    ax.plot(list(map(lambda w: w[0], w_list)), list(map(lambda x: x[1], w_list)), marker=".")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    plot(ax, f)
  fig.savefig("optimizer.png")
