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
    return int(x > 0)


if __name__ == "__main__":
    x1, x2 = 0, np.pi

    print(f'Teste das funções para x1 = {x1}')
    print(f'{sigmoid(x1) = }')
    print(f'{sigmoid_diff(x1) = }')
    print(f'{tanh(x1) = }')
    print(f'{tanh_diff(x1) = }')
    print(f'{ReLU(x1) = }')
    print(f'{ReLU_diff(x1) = }')
    print()

    print(f'Teste das funções para x2 = {x2}')
    print(f'{sigmoid(x2) = }')
    print(f'{sigmoid_diff(x2) = }')
    print(f'{tanh(x2) = }')
    print(f'{tanh_diff(x2) = }')
    print(f'{ReLU(x2) = }')
    print(f'{ReLU_diff(x2) = }')
    print()

