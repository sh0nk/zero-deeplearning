import numpy as np
import matplotlib.pylab as plt
from neural import helper


def draw_step():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid.step_func(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # fix y range
    plt.show()


def draw_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid.sigmoid_func(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # fix y range
    plt.show()
