#!/usr/bin/env python3
import numpy as np
from lib.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []
train_acc_list = []
test_acc_list = []

# hyper parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

#epoch size
iter_per_epoch = max(train_size / batch_size, 1)

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

  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc | {} {}".format(train_acc, test_acc))

import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
fig, [[axL, axR], [axLL, axLR]] = plt.subplots(ncols=2, nrows=2, figsize=(10,4))
axL.set_xlim(0, iters_num)
axL.plot(np.arange(iters_num), train_loss_list)
axR.set_xlim(0, 1000)
axR.plot(np.arange(1000), train_loss_list[0:1000])
axLL.set_xlim(0, len(train_acc_list) - 1)
axLL.plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
axLL.plot(np.arange(len(test_acc_list)), test_acc_list, "--", label="test acc")
axLL.legend()
fig.savefig("train_two_layer_net_loss.png")
