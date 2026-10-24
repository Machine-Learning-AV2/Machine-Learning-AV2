import numpy as np
from functions import functions, functions_name
from metricas import MSE
from typing import Callable


class NeuralNetwork:
    def __init__(self, layers: list[int], activate_function: functions_name | list[functions_name], 
                 output_activation: functions_name = 'identity', function_dictionary: dict[str, tuple[Callable, Callable]] = functions,  
                 bayes_interval: tuple[float, float] = (0, 1), weight_interval: tuple[float, float] = (0, 1), 
                 seed:int = 782) -> None:
        """_summary_

        Args:
            layers (list[int]): _description_
            activate_function (Callable[[float  |  np.ndarray], float  |  np.ndarray]): _description_
            activate_diff_function (Callable[[float  |  np.ndarray], float  |  np.ndarray]): _description_
            seed (int, optional): _description_. Defaults to 782.
        """

        self._seed = seed
        self.original_seed = seed
        self.layers = layers

        self.bayes: list[np.ndarray] = [self.random_bayes(i, *bayes_interval) for i in range(1, len(layers))]
        self.weight: list[np.ndarray] = [self.random_weight(i, *weight_interval) for i in range(1, len(layers))]

        if isinstance(activate_function, list):
            if len(activate_function) != len(layers)-2:
                raise KeyError("ERROR: Quantidade de funções diferente dos layers ocultos")
            self._activate_function: list[functions_name] = activate_function.copy()
        else:
            self._activate_function: list[functions_name] = [activate_function] * (len(layers) - 2)

        self._activate_function += [output_activation]
        self._function_dictionary = function_dictionary.copy()

        self.check_activate(self._activate_function, self._function_dictionary)

    @staticmethod
    def check_activate(activate_functions: list[str], activate_function_dictionary: dict[str, tuple[Callable, Callable]]) -> None:
        for function_name in activate_functions:
            if function_name not in activate_function_dictionary:
                raise ValueError(f"ERROR: Função de ativação inválida: {function_name}")

    @staticmethod
    def check_input(x: np.ndarray) -> None:
        if x.ndim > 2:
            raise ValueError("Entrada invalida, x deve ser um vetor ou matriz")

    def random_bayes(self, level: int, low: float = 0, high: float = 1) -> np.ndarray:
        """_summary_

        Args:
            level (int): _description_
            low (float, optional): _description_. Defaults to 0.
            high (float, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """

        np.random.seed(self._seed)
        array = np.random.uniform(low, high, (self.layers[level], 1))
        
        self._seed += 1
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

        np.random.seed(self._seed)
        array = np.random.uniform(low, high, (self.layers[level], self.layers[level-1]))
        
        self._seed += 1
        return array

    def train(self, X: np.ndarray, y: np.ndarray, learn_rate: float = 0.1, 
              batch: int = 10, epochs: int = 100, 
              loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = MSE) -> None:
        """_summary_

        Args:
            x (np.ndarray): _description_
            y (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError

    def prevision(self, x: float | np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (float | np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float | np.ndarray: _description_
        """

        x = np.array(x)

        self.check_input(x)

        y: np.ndarray = x.copy()

        for level in range(1, len(self.layers)):
            bayes: np.ndarray = self.bayes[level-1]
            weight: np.ndarray = self.weight[level-1]
            activate_func = self._function_dictionary[self._activate_function[level-1]][0]

            y = activate_func(bayes + weight @ y)

        return y


if __name__ == "__main__":
    rn = NeuralNetwork([2, 4, 5, 2], activate_function="ReLU", output_activation="sigmoid", bayes_interval = (-3, 3), weight_interval = (-10, 10))
    print(f"\ny = {rn.prevision(np.array([[0.5], [0.2]]))}")