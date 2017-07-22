import numpy as np


def step_func(x):
    """
    x: int, float, np.array, np.generic
    """
    if isinstance(x, (int, float)):
        if x > 0:
            return 1
        else:
            return 0
    elif isinstance(x, (np.ndarray, np.generic)):
        y = x > 0
        return y.astype(np.int)


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)
