import numpy as np
from functions import functions, functions_name
from metricas import MSE
from typing import Callable


class NeuralNetwork:
    def __init__(self, layers: list[int], activate_function: functions_name | list[functions_name], 
                 output_activation: functions_name = 'identity', function_dictionary: dict[str, tuple[Callable, Callable]] = functions,  
                 bias_interval: tuple[float, float] = (0, 1), weight_interval: tuple[float, float] = (0, 1), 
                 seed:int = 782) -> None:
        """
            Inicializa a classe NeuralNetwork
            
            Parâmetros
            -----------
            layers list[int] : Quantas layers tem e qual a largura de cada;
            activate_function str | list[str] : Quais funções de ativação serão usadas a cada camada oculta;
            output_activation str | list[str] : Qual a função ativação da camada de saida (identidade por padrão);
            function_dictionary dict[str, tuple[Callable, Callable]] : Mapa de funções de ativação e suas derivadas;
            baias_interval tuple[float, float] : Qual intervalo os vieses serão gerado ((0, 1) por padrão);
            weight_interval tuple[float, float] : Qual intervalo os pesos serão gerado ((0, 1) por padrão);
            seed int : Seed usada para gerar os vieses e pesos
        """

        self.original_seed = seed
        self._rng = np.random.default_rng(seed)
        self.layers = layers

        self._state_linear: list[np.ndarray] = [np.array([]) for _ in range(1, len(layers))]
        self._state_layers: list[np.ndarray] = [np.array([]) for _ in range(len(layers))]

        self.bias: list[np.ndarray] = [self.random_bias(i, *bias_interval) for i in range(1, len(layers))]
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
        """
            Verifica se todas as funções de ativação usadas estão no mapa
            
            Parâmetros
            -----------
            activate_function list[str] : Quais funções de ativação serão usadas a cada camada oculta;
            function_dictionary dict[str, tuple[Callable, Callable]] : Mapa de funções de ativação e suas derivadas.
        """
        for function_name in activate_functions:
            if function_name not in function_dictionary:
                raise ValueError(f"ERROR: Função de ativação inválida: {function_name}")

    @staticmethod
    def check_input(x: np.ndarray) -> np.ndarray:
        """
            Verifica se x é valido e arruma suas dimenções 
            
            Parâmetros
            -----------
            x np.ndarray : Array a se verificar.

            Retorna
            -----------
            x com as dimenções atualizadas
        """

        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        elif isinstance(x, list) or isinstance(x, tuple):
            x = np.array(x)

        if x.ndim > 2:
            raise ValueError("Entrada invalida, x deve ser um vetor ou matriz")
        elif x.ndim == 1:
            if len(x) == 1:
                x = x.reshape(-1, 1) 
            else:
                x = x.reshape(1, -1)

        return x

    def random_bias(self, level: int, low: float = 0, high: float = 1) -> np.ndarray:
        """
            Gera um vetor viés uniformemente aleatório entre [low, high]
            
            Parâmetros
            -----------
            level int : Para qual layer será gerado o viés;
            low float : Valor mínimo dos vieses (0 por padrão);
            high float : Valor maximo dos vieses (1 por padrão).

            Retorna
            -----------
            Vetor dos vieses com o mesma largura do layer[nivel]
        """

        array = self._rng.uniform(low, high, (self.layers[level], 1))
        
        return array

    def random_weight(self, level: int, low: float = 0, high: float = 1) -> np.ndarray:
        """
            Gera uma matriz peso uniformemente  aleatória entre [low, high]
            
            Parâmetros
            -----------
            level int : Para qual layer será gerado o peso;
            low float : Valor mínimo dos pesos (0 por padrão);
            high float : Valor maximo dos pesos (1 por padrão).

            Retorna
            -----------
            matriz dos pesos com dimenção da largura do layer[nivel] x layer[nivel-1]
        """

        array = self._rng.uniform(low, high, (self.layers[level], self.layers[level-1]))
        
        return array

    def train(self, X: np.ndarray, y: np.ndarray, learn_rate: float = 0.1, 
              batch: int = 10, epochs: int = 1000, tol = 1e-9) -> None:
        """
            Treina a rede neural a partir dos dados X, y
            
            Parâmetros
            -----------
            X np.ndarray : Matriz P x N dos dados de treino;
            y np.ndarray : Matriz S x N das saidas de treino;
            learn_rate float : Taxa de aprendizado (0.1 por padrão);
            batch int : Quantidade esperada de elementos por balde (10 por padrão);
            epochs int : Quantidade de interações nos dados de treino (1000 por padrão).
        """

        self._loss = {
            'MSE': (MSE, lambda y, y_pred: 2*(y_pred - y)/ y.shape[1]) 
        }

        loss_function: str = 'MSE' #COLOCAR PARA PARAMETROS

        X = self.check_input(X)
        y = self.check_input(y)


        index: np.ndarray = np.arange(X.shape[1])

        loss = self._loss[loss_function][0](y, self.prevision(X))
        for epoch in range(epochs):
            self._rng.shuffle(index)

            for i in range(0, X.shape[1], batch):
                if X.shape[1] - i < 2*batch:
                    B = index[i:]
                else:
                    B = index[i: i+batch]

                self.backpropagation(X[:, B], y[:, B], loss_function = loss_function, learn_rate = learn_rate)

            loss_aux = self._loss[loss_function][0](y, self.prevision(X))

            if abs(loss - loss_aux) < tol:
                break
            else:
                loss = loss_aux

    def backpropagation(self, x: np.ndarray, y: np.ndarray, loss_function: str = 'MSE', learn_rate: float = 0.1) -> None:
        """
            Calcula as decidas de gradiente para cada peso e viés.
            
            Parâmetros
            -----------
            x np.ndarray : Matriz P x N dos dados de treino;
            y np.ndarray : Matriz S x N das saidas de treino;
            loss_function str : Função perda utilizada para calcular o gradiente;
            learn_rate float : Taxa de aprendizado (0.1 por padrão);
        """


        self.prevision(x, save_states=True)
        cost: list[np.ndarray] = [np.array([]) for _ in range(1, len(self.layers))]

        activate_function = self.activate_functions[-1]
        cost[-1] = self._loss[loss_function][1](y, self._state_layers[-1]) * self._function_dictionary[activate_function][1](self._state_linear[-1])

        for level in range(len(self.layers)-3, -1, -1):
            activate_function = self.activate_functions[level]
            cost[level] =  self.weight[level+1].T @ cost[level+1] * self._function_dictionary[activate_function][1](self._state_linear[level])

        for level in range(len(self.layers)-1):
            self.weight[level] -= learn_rate * (cost[level] @ self._state_layers[level].T)
            self.bias[level] -= learn_rate * np.sum(cost[level], axis=1, keepdims=True)


    def prevision(self, x: float | np.ndarray, save_states: bool = False, classify = False) -> np.ndarray:
        """
            Calcula o valor de saida da rede neural com pesos e vieses atuais
            
            Parâmetros
            -----------
            x float | np.ndarray : Vetor ou matriz que queremos prever o resultado;
            save_states bool : Define se queremos salvar os valores de cada layer ou não.

            Retorna
            -----------
            Valor de saida da rede neural com entrada x
        """

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

        if classify:
            return np.argmax(y)

        return y


if __name__ == "__main__":

    print('-'*100)
    print('Teste 1')
    print('-'*100)
    rn1 = NeuralNetwork([2, 4, 5, 2], activate_function="sigmoid", output_activation="sigmoid", bias_interval = (-1, 1), weight_interval = (-1, 1))

    rng = np.random.default_rng(1)
    X = rng.uniform(-4, 4, (2, 1000))
    y = np.zeros(X.shape)

    for i, x in enumerate(X.T):
        if x[0]**2 + x[1]**2 <= 9:
            y[:, i] = [1, 0]
        else:
            y[:, i] = [0, 1]

    rn1.train(X, y, batch=50)

    x1 = np.array([[1], [0]])
    x2 = np.array([[3], [3]])
    x3 = np.array([[5], [5]])

    print(f"\nx_1=({x1[0]}, {x1[1]}) y_1 = {rn1.prevision(x1)}")
    print(f"\ny_2 = {rn1.prevision(x2)}")
    print(f"\ny_3 = {rn1.prevision(x3)}")

    print(f"\nx_1=({x1[0]}, {x1[1]}) y_1 = {rn1.prevision(x1, classify=True)}")
    print(f"\ny_2 = {rn1.prevision(x2, classify=True)}")
    print(f"\ny_3 = {rn1.prevision(x3, classify=True)}")

    print('-'*100)
    print('Teste 2')
    print('-'*100)

    rn2 = NeuralNetwork([1, 4, 5, 1], activate_function="ReLU", output_activation="identity", bias_interval = (-0.1, 0.1), weight_interval = (-0.1, 0.1))

    f = lambda x: np.exp(x)

    X = np.linspace(-2, 3, 10000).reshape(1, -1)
    y = f(X)

    rn2.train(X, y, batch=50, learn_rate=0.001, epochs=10000)

    for x1 in range(20):
        print(f"\nx_1={x1}, y_1={rn2.prevision(x1)}")
    
