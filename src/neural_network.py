import numpy as np
import functions as func
from collections.abc import Callable

from pprint import pprint


class NeuralNetwork:
    def __init__(self, layers: list[int], activate_function: Callable[[float | np.ndarray], float | np.ndarray],
                 activate_diff_function: Callable[[float | np.ndarray], float | np.ndarray], seed:int = 782) -> None:
        """_summary_

        Args:
            layers (list[int]): _description_
            activate_function (Callable[[float  |  np.ndarray], float  |  np.ndarray]): _description_
            activate_diff_function (Callable[[float  |  np.ndarray], float  |  np.ndarray]): _description_
            seed (int, optional): _description_. Defaults to 782.
        """

        self._seed = seed
        self._activate_function = activate_function
        self._activate_diff_function = activate_diff_function
        self.layers = layers

        self.bayes: list[np.ndarray] = [self.random_bayes(i) for i in range(1, len(layers))]
        self.weight: list[np.ndarray] = [self.random_weight(i) for i in range(1, len(layers))]
        
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

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """_summary_

        Args:
            x (np.ndarray): _description_
            y (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError

    def prevision(self, x: float | np.ndarray) -> float | np.ndarray:
        """_summary_

        Args:
            x (float | np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float | np.ndarray: _description_
        """

        x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 2:
            if x.shape[0] == 1 and x.shape[1] != 1:
                x = x.T
        else:
            raise ValueError("Entrada invalida, x deve ser um vetor ou matriz")

        y: float | np.ndarray = x.copy()
        activate_func = self._activate_function

        for level in range(1, len(self.layers)):
            bayes: np.ndarray = self.bayes[level-1]
            weight: np.ndarray = self.weight[level-1]

            y = activate_func(bayes + weight @ y)

        return y


if __name__ == "__main__":
    rn = NeuralNetwork([2, 4, 5, 2], func.sigmoid, func.sigmoid_diff)
    print(f"\ny = {rn.prevision(np.array([[0.5, 6], [0.2, 7]]))}")