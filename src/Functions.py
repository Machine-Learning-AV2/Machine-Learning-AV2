import numpy as np
from typing import Literal


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return 1 / (1 + np.exp(-x))


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.array(scores)  
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))

    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def sigmoid_diff(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return np.tanh(x)


def tanh_diff(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return 1 -  tanh(x)**2


def ReLU(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return np.maximum(0, x)


def ReLU_diff(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return (x > 0).astype(int)


def  identity(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return x


def  identity_diff(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x)

    return np.ones_like(x)


functions = {
    'sigmoid': (sigmoid, sigmoid_diff),
    'softmax': (softmax, None),
    'tanh': (tanh, tanh_diff),
    'ReLU': (ReLU, ReLU_diff),
    'identity': (identity, identity_diff)
}

functions_name = Literal['identity', 'ReLU', 'sigmoid', 'softmax', 'tanh']

if __name__ == "__main__":
    x1, x2 = 0, np.pi

    print(f'Teste das funções para x1 = {x1}')
    print(f'{sigmoid(x1) = }')  
    print(f'{sigmoid_diff(x1) = }')
    print(f'{tanh(x1) = }')
    print(f'{tanh_diff(x1) = }')
    print(f'{ReLU(x1) = }')
    print(f'{ReLU_diff(x1) = }')
    print(f'{identity(x1) = }')
    print(f'{identity_diff(x1) = }')
    print()

    print(f'Teste das funções para x2 = {x2}')
    print(f'{sigmoid(x2) = }')
    print(f'{sigmoid_diff(x2) = }')
    print(f'{tanh(x2) = }')
    print(f'{tanh_diff(x2) = }')
    print(f'{ReLU(x2) = }')
    print(f'{ReLU_diff(x2) = }')
    print(f'{identity(x2) = }')
    print(f'{identity_diff(x2) = }')
    print()
