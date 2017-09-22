#!/usr/bin/env python3
import numpy as np
from lib.mnist import load_mnist
from chap6.multi_layer_net import MultiLayerNet
from optimizers import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:300]
t_train = t_train[:300]
network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], output_size=10)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
#iter_per_epoch = 100
epoch_cnt = 0

for i in range(1000000000):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  grads = network.gradient(x_batch, t_batch)
  optimizer.update(network.params, grads)

  train_loss_list.append(network.loss(x_batch, t_batch))

  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    epoch_cnt += 1
    if epoch_cnt >= max_epochs:
      break

import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 6))
axL.set_xlim(0, len(train_loss_list))
axL.plot(np.arange(len(train_loss_list)), train_loss_list)
axR.set_xlim(0, len(train_acc_list) - 1)
axR.plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
axR.plot(np.arange(len(test_acc_list)), test_acc_list, "--", label="test acc")
axR.legend()
fig.savefig("overfit_weight_decay.png")
