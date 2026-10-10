from numpy import exp, tanh, array

def sigmoid(x: float | array) -> float | array:
    return 1/(1+exp(-x))


def sigmoid_diff(x: float | array) -> float | array:
    return sigmoid(x)/(1 - sigmoid(x))


def tanh(x: float | array) -> float | array:
    return tanh(x)

def tanh_diff(x: float | array) -> float | array:
    return 1 -  tanh(x)**2


def ReLU(x: float | array) -> float | array:
    return max(0, x)

def ReLU_diff(x: float | array) -> float | array:
    return 0 if x < 0 else 1
