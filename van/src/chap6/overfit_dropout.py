#!/usr/bin/env python3
import numpy as np
from lib.mnist import load_mnist
from chap6.multi_layer_net import MultiLayerNet
from chap6.overfit_weight_decay import train
from optimizers import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:300]
t_train = t_train[:300]
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

# train
network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], output_size=10)
(train_loss_list, train_acc_list, test_acc_list) = train(network)
network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], output_size=10, use_dropout=True, dropout_ratio=0.2)
(train_loss_list_decay, train_acc_list_decay, test_acc_list_decay) = train(network)

# draw out
import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
fig, [[axL, axR], [axLL, axLR]] = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
axL.set_xlim(0, len(train_loss_list))
axL.plot(np.arange(len(train_loss_list)), train_loss_list)
axR.set_xlim(0, len(train_acc_list))
axR.set_ylim(0, 1) 
axR.plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
axR.plot(np.arange(len(test_acc_list)), test_acc_list, "--", label="test acc")
axR.legend()
axLL.set_xlim(0, len(train_loss_list_decay))
axLL.plot(np.arange(len(train_loss_list_decay)), train_loss_list_decay)
axLR.set_xlim(0, len(train_acc_list_decay))
axLR.set_ylim(0, 1) 
axLR.plot(np.arange(len(train_acc_list_decay)), train_acc_list_decay, label="train acc")
axLR.plot(np.arange(len(test_acc_list_decay)), test_acc_list_decay, "--", label="test acc")
axLR.legend()
fig.savefig("overfit_dropout.png")
