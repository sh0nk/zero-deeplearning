#!/usr/bin/env python3
import numpy as np
from lib.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

# hyper parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  #grad = network.numerical_gradient(x_batch, t_batch)
  grad = network.gradient(x_batch, t_batch)

  for key in ("W1", "b1", "W2", "b2"):
    network.params[key] -= learning_rate * grad[key]

  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
axL.set_xlim(0, iters_num)
axL.plot(np.arange(iters_num), train_loss_list)
axR.set_xlim(0, 1000)
axR.plot(np.arange(1000), train_loss_list[0:1000])
fig.savefig("train_two_layer_net_loss.png")
