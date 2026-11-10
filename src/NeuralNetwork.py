import numpy as np
from . import Functions
from .Metrics import MSE, DMSE 
from typing import Callable



    @staticmethod
    def _check_activate(activate_functions: list[str], function_dictionary: dict[str, tuple[Callable, Callable]]) -> None:
        """
            Verifica se todas as funções de ativação usadas estão no mapa
            
            Parâmetros
            -----------
            activate_function list[str] : Quais funções de ativação serão usadas a cada camada oculta;
            function_dictionary dict[str, tuple[Callable, Callable]] : Mapa de funções de ativação e suas derivadas.
        """
        for function_name in activate_functions:
            if function_name not in function_dictionary:
                raise KeyError(f"ERROR: Função de ativação inválida: {function_name}")

    @staticmethod
    def _check_input(x: np.ndarray) -> np.ndarray:
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
                x = x.reshape(-1, 1)

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

    def fit(self, X: np.ndarray, y: np.ndarray, loss: Callable = MSE, loss_diff: Callable = DMSE, learn_rate: float = 0.1, 
              batch: int = 10, epochs: int = 1000, tol = 1e-9, debug: bool = False) -> None:
        """
            Treina a rede neural a partir dos dados X, y
            
            Parâmetros
            -----------
            X np.ndarray : Matriz P x N dos dados de treino;
            y np.ndarray : Matriz S x N das saidas de treino;
            loss Callable : Função perda para aprendizado;
            loss_diff Callable : Derivada da função perda para backpropagation;
            learn_rate float : Taxa de aprendizado (0.1 por padrão);
            batch int : Quantidade esperada de elementos por balde (10 por padrão);
            epochs int : Quantidade de interações nos dados de treino (1000 por padrão).
        """
        X = self._check_input(X.T)
        y = self._check_input(y.T)


        index: np.ndarray = np.arange(X.shape[1])

        loss_val = loss(self.prevision(X.T), y.T)
        for epoch in range(epochs):
            if debug:
                print(f"Iniciando epoca {epoch}, com erro {loss_val}")
            self._rng.shuffle(index)

            for i in range(0, X.shape[1], batch):
                if X.shape[1] - i < 2*batch:
                    B = index[i:]
                else:
                    B = index[i: i+batch]

                self._backpropagation(X[:, B], y[:, B], loss_diff = loss_diff , learn_rate = learn_rate)

            loss_aux = loss(self.prevision(X.T), y.T)

            if abs(loss_val - loss_aux) / max(abs(loss_val), 1e-9) < tol:
                break
            else:
                loss_val = loss_aux

    def _backpropagation(self, x: np.ndarray, y: np.ndarray, loss_diff = DMSE, learn_rate: float = 0.1) -> None:
        """
            Calcula as decidas de gradiente para cada peso e viés.
            
            Parâmetros
            -----------
            x np.ndarray : Matriz P x N dos dados de treino;
            y np.ndarray : Matriz S x N das saidas de treino;
            learn_rate float : Taxa de aprendizado (0.1 por padrão);
        """


        self.prevision(x.T, save_states=True)
        cost: list[np.ndarray] = [np.array([]) for _ in range(1, len(self.layers))]

        activate_function = self.activate_functions[-1]
        cost[-1] = loss_diff(self._state_layers[-1].T, y.T).T * self._function_dictionary[activate_function][1](self._state_linear[-1])

        for level in range(len(self.layers)-3, -1, -1):
            activate_function = self.activate_functions[level]
            cost[level] =  self.weight[level+1].T @ cost[level+1] * self._function_dictionary[activate_function][1](self._state_linear[level])

        for level in range(len(self.layers)-1):
            self.weight[level] -= learn_rate * (cost[level] @ self._state_layers[level].T)
            self.bias[level] -= learn_rate * np.sum(cost[level], axis=1, keepdims=True)


    def prevision(self, x: float | np.ndarray, save_states: bool = False, classify: bool = False) -> np.ndarray:
        """
            Calcula o valor de saida da rede neural com pesos e vieses atuais
            
            Parâmetros
            -----------
            x float | np.ndarray : Vetor ou matriz que queremos prever o resultado;
            save_states bool : Define se queremos salvar os valores de cada layer ou não;
            classify bool : Define se queremos retornar o vetor de probabilidade ou atribuir o elemento a uma classe.

            Retorna
            -----------
            Valor de saida da rede neural com entrada x
        """

        x = self._check_input(x)
        x = x.T


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

        return y.T

    def save(self, path: str) -> None:
        """
        Salva o estado atual da rede neural quadando layers, activate functions, weight e bias
                    
        Recomenda-se usar a extenção .npz

        Parâmetros
        -----------
        path str : Caminho na qual o arquivo deve ficar salvo
        """
        data = {
            "layers": np.asarray(self.layers),
            "activate_functions": np.asarray(self.activate_functions, dtype=str)
        }

        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            data[f"weight_{i}"] = w
            data[f"bias_{i}"] = b

        np.savez_compressed(path, **data)


def load_neural_network(path: str) -> NeuralNetwork:
    """
    Carrega o estado de uma rede neural, le: layers, activate functions, weight e bias
                        
    Parâmetros
    -----------
    path str : Caminho na qual o arquivo deve ler as informações

    Retorno
    -----------
    Retorna uma NeuralNetwork com layers, activate functions, weight e bias definidos pelo arquivo
    """
    with np.load(path, allow_pickle=False) as data:
        layers = list(data['layers'])
        activate_functions = list(data['activate_functions'])

        weights = [data[f"weight_{i}"].copy() for i in range(len(layers)-1)]
        biases = [data[f"bias_{i}"].copy() for i in range(len(layers)-1)]

        nn = NeuralNetwork(layers, activate_functions[:-1], activate_functions[-1])
        nn.bias = biases
        nn.weight = weights

        return nn


if __name__ == "__main__":

    print('-'*100)
    print('Teste 1')
    print('-'*100)
    rn1 = NeuralNetwork([2, 4, 5, 2], activate_function="sigmoid", output_activation="sigmoid", bias_interval = (-1, 1), weight_interval = (-1, 1))

    rng = np.random.default_rng(1)
    X = rng.uniform(-4, 4, (1000, 2))
    y = np.zeros(X.shape)

    for i, x in enumerate(X):
        if x[0]**2 + x[1]**2 <= 9:
            y[i, :] = [1, 0]
        else:
            y[i, :] = [0, 1]

    rn1.fit(X, y, batch=50)

    x1 = np.array([1, 0])
    x2 = np.array([3, 3])
    x3 = np.array([5, 5])

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

    X = np.linspace(-2, 3, 1000).reshape(-1, 1)
    y = f(X)

    rn2.fit(X, y, batch=50, learn_rate=0.001, epochs=1000)

    for x1 in range(20):
        print(f"\nx_1={x1}, y_1={rn2.prevision(x1)}")
    
