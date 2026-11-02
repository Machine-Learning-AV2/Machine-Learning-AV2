import numpy as np
from functions import functions, functions_name
from metricas import MSE
from typing import Callable


class NeuralNetwork:
    def __init__(self, layers: list[int], activate_function: functions_name | list[functions_name], 
                 output_activation: functions_name = 'identity', function_dictionary: dict[str, tuple[Callable, Callable]] = functions,  
                 bias_interval: tuple[float, float] = (0, 1), weight_interval: tuple[float, float] = (0, 1), 
                 seed:int = 782) -> None:
        """_summary_

        Args:
            layers (list[int]): _description_
            activate_function (Callable[[float  |  np.ndarray], float  |  np.ndarray]): _description_
            activate_diff_function (Callable[[float  |  np.ndarray], float  |  np.ndarray]): _description_
            seed (int, optional): _description_. Defaults to 782.
        """

        self.original_seed = seed
        self._rng = np.random.default_rng(seed)
        self.layers = layers

        self._state_linear: list[np.ndarray] = [np.array([]) for _ in range(1, len(layers))]
        self._state_layers: list[np.ndarray] = [np.array([]) for _ in range(len(layers))]

        self.bias: list[np.ndarray] = [self.random_bayes(i, *bias_interval) for i in range(1, len(layers))]
        self.weight: list[np.ndarray] = [self.random_weight(i, *weight_interval) for i in range(1, len(layers))]

        if isinstance(activate_function, list):
            if len(activate_function) != len(layers)-2:
                raise KeyError("ERROR: Quantidade de funções diferente dos layers ocultos")
            self.activate_functions: list[str] = activate_function.copy()
        elif isinstance(activate_function, str):
            self.activate_functions: list[str] = [activate_function] * (len(layers) - 2)
        else :
            raise TypeError("ERROR: Tipo da activete_function está incorreto")

        self.activate_functions += [output_activation]
        self._function_dictionary = function_dictionary.copy()

        self.check_activate(self.activate_functions, self._function_dictionary)

    @staticmethod
    def check_activate(activate_functions: list[str], function_dictionary: dict[str, tuple[Callable, Callable]]) -> None:
        for function_name in activate_functions:
            if function_name not in function_dictionary:
                raise ValueError(f"ERROR: Função de ativação inválida: {function_name}")

    @staticmethod
    def check_input(x: np.ndarray) -> np.reshape:
        if x.ndim > 2:
            raise ValueError("Entrada invalida, x deve ser um vetor ou matriz")
        elif x.ndim == 1:
            x = x.reshape(-1, 1) 

        return x

    def random_bayes(self, level: int, low: float = 0, high: float = 1) -> np.ndarray:
        """_summary_

        Args:
            level (int): _description_
            low (float, optional): _description_. Defaults to 0.
            high (float, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """

        array = self._rng.uniform(low, high, (self.layers[level], 1))
        
        return array

    def random_weight(self, level: int, low: float = 0, high: float = 1) -> np.ndarray:
        """_summary_

        Args:
            level (int): _description_
            low (float, optional): _description_. Defaults to 0.
            high (float, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """

        array = self._rng.uniform(low, high, (self.layers[level], self.layers[level-1]))
        
        return array

    def train(self, X: np.ndarray, y: np.ndarray, learn_rate: float = 0.1, 
              batch: int = 10, epochs: int = 100) -> None:
        """_summary_

        Args:
            x (np.ndarray): _description_
            y (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_
        """

        loss_function: str = 'MSE' #COLOCAR PARA PARAMETROS

        X = self.check_input(X)
        y = self.check_input(y)


        index: np.ndarray = np.arange(X.shape[1])

        for epoch in range(epochs):
            self._rng.shuffle(index)

            batchs_index: list[np.ndarray] = np.array_split(index, max(1, X.shape[1] // batch))

            for B in batchs_index:
                self.backpropagation(X[:, B], y[:, B], loss_function = loss_function, learn_rate = learn_rate)

    def backpropagation(self, x: np.ndarray, y: np.ndarray, loss_function: str = 'MSE', learn_rate: float = 0.1) -> None:

        N = y.shape[1]
        loss = {
            'MSE': (MSE, lambda y, y_pred: 2*(y_pred - y)) 
        }

        self.prevision(x, save_states=True)
        cost: list[np.ndarray] = [np.array([]) for _ in range(1, len(self.layers))]

        activate_function = self.activate_functions[-1]
        cost[-1] = loss[loss_function][1](y, self._state_layers[-1]) * self._function_dictionary[activate_function][1](self._state_linear[-1])

        for level in range(len(self.layers)-3, -1, -1):
            activate_function = self.activate_functions[level]
            cost[level] =  self.weight[level+1].T @ cost[level+1] * self._function_dictionary[activate_function][1](self._state_linear[level])

        for level in range(len(self.layers)-1):
            print(cost[level].shape)
            self.weight[level] -= learn_rate * (cost[level] @ self._state_layers[level].T) / N
            self.bias[level] -= learn_rate * np.mean(cost[level], axis=1, keepdims=True)


    def prevision(self, x: float | np.ndarray, save_states: bool = False) -> np.ndarray:
        """_summary_

        Args:
            x (float | np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float | np.ndarray: _description_
        """

        x = np.array(x)

        x = self.check_input(x)

        y: np.ndarray = x.copy()

        if save_states:
            self._state_layers[0] = y.copy()

        for level in range(len(self.layers)-1):
            bayes: np.ndarray = self.bias[level]
            weight: np.ndarray = self.weight[level]
            activate_func = self._function_dictionary[self.activate_functions[level]][0]

            linear = bayes + weight @ y
            layer = activate_func(linear)
            if save_states:
                self._state_linear[level] = linear
                self._state_layers[level + 1] = layer

            y = layer

        return y


if __name__ == "__main__":
    rn = NeuralNetwork([2, 4, 5, 3], activate_function="sigmoid", output_activation="identity", bias_interval = (-3, 3), weight_interval = (-10, 10))

    # X = np.arange(512).reshape((1, 512))
    # y = np.arange(512).reshape((1, 512))

    X = np.array([[9, 3], [1, 5]])
    y = np.array([[2, 1], [2, 1], [2, 1]])

    print(f"\ny = {rn.prevision(np.array([[100], [1]]))}")
    rn.train(X, y)

    print(f"\ny = {rn.prevision(np.array([[100], [1]]))}")