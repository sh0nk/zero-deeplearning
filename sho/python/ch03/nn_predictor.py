import numpy as np
import matplotlib.pylab as plt

class ActiveFunctions():
	def step_function(x):
		return np.array(x > 0, dtype = np.int)

	def relu(x):
		return np.maximum(0, x)

	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def identeity_function(x):
		return x

	def softmax(x):
		c = np.max(x)
		exp_x = np.exp(x - c)
		sum_exp_x = np.sum(exp_x)
		return exp_x / sum_exp_x


class TriNetwork:
	def init_network(self):
		self.W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
		self.b1 = np.array([0.1, 0.2, 0.3])
		self.W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
		self.b2 = np.array([0.1, 0.2])
		self.W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
		self.b3 = np.array([0.1, 0.2])

	def init_network_from_file(self, file):
		import pickle
		with open(file, 'rb') as f:
			o = pickle.load(f)

		self.W1, self.W2, self.W3 = o['W1'], o['W2'], o['W3']
		self.b1, self.b2, self.b3 = o['b1'], o['b2'], o['b3']

	def forward(self, x):
		a1 = np.dot(x, self.W1) + self.b1
		z1 = ActiveFunctions.sigmoid(a1)
		a2 = np.dot(z1, self.W2) + self.b2
		z2 = ActiveFunctions.sigmoid(a2)
		a3 = np.dot(z2, self.W3) + self.b3
		# y = ActiveFunctions.identeity_function(a3)
		y = ActiveFunctions.softmax(a3)

		return y

def get_data(pardir):
	import sys, os
	sys.path.append(pardir)
	from dataset.mnist import load_mnist

	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

if __name__ == "__main__":
	# x = np.arange(-5.0, 5.0, 0.1)
	# # y = step_function(x)
	# y = sigmoid(x)
	# plt.plot(x, y)
	# plt.ylim(-0.1, 1.1)
	# plt.show()

	network = TriNetwork()
	# network.init_network()
	# r = network.forward(np.array([1.0, 0.5]))
	# print(r)

	x, t = get_data("../../../../deep-learning-from-scratch")

	network.init_network_from_file("../../../../deep-learning-from-scratch/ch03/sample_weight.pkl")

	# Accuracy
	correct = 0
	for i in range(len(x)):
		y = network.forward(x[i])
		p = np.argmax(y)
		if p == t[i]:
			correct += 1
	print("correctness %f" % (float(correct) / len(x)))

