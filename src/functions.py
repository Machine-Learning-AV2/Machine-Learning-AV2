import numpy as np

def sigmoid(x: float | np.array) -> float | np.array:
    return 1 / (1 + np.exp(-x))


def sigmoid_diff(x: float | np.array) -> float | np.array:
    return sigmoid(x) / (1 - sigmoid(x))


def tanh(x: float | np.array) -> float | np.array:
    return np.tanh(x)


def tanh_diff(x: float | np.array) -> float | np.array:
    return 1 -  tanh(x)**2


def ReLU(x: float | np.array) -> float | np.array:
    return max(0, x)

def ReLU_diff(x: float | np.array) -> float | np.array:
    return 0 if x < 0 else 1
