# Módulo Neural Network

Arquivo: `src/NeuralNetwork.py` (apenas NumPy)

Contém **NeuralNetwork** (Rede Neural). `X` é sempre uma matriz `(p_pretitores, n_amostras)`.

## *NeuralNetwork.NeuralNetwork*

```python
NeuralNetwork.NeuralNetwork(layers, activate_function, output_activation = 'identity', function_dictionary = Functions.functions,  bias_interval = (0, 1), weight_interval = (0, 1), seed = 782):
```

Intancia um elemento a classe NeuralNetwork com pesos e vieses aleatórios dentro do intervalo estipulado.

Parâmetros
__________________________

layers list[int] : Quantas layers tem e qual a largura de cada; \
activate_function str | list[str] : Quais funções de ativação serão usadas a cada camada oculta;\
output_activation str | list[str] : Qual a função ativação da camada de saida (identidade por padrão);\
function_dictionary dict[str, tuple[Callable, Callable]] : Mapa de funções de ativação e suas derivadas;\
baias_interval tuple[float, float] : Qual intervalo os vieses serão gerado ((0, 1) por padrão);\
weight_interval tuple[float, float] : Qual intervalo os pesos serão gerado ((0, 1) por padrão);\
seed int : Seed usada para gerar os vieses e pesos

### *NeuralNetwork.NeuralNetwork.random_bias*

```python

NeuralNetwork.NeuralNetwork.random_bias(level, low = 0, high = 1)
```

Gera um vetor viés uniformemente aleatório entre [low, high]

Parâmetros
---
level int : Para qual layer será gerado o viés;\
low float : Valor mínimo dos vieses (0 por padrão);\
high float : Valor maximo dos vieses (1 por padrão).\

Retorna
---
Vetor dos vieses com o mesma largura do layer[nivel]

### *NeuralNetwork.NeuralNetwork.random_weight*

```python
NeuralNetwork.NeuralNetwork.random_weight(level: int, low: float = 0, high: float = 1)
```

Gera uma matriz peso uniformemente  aleatória entre [low, high]

Parâmetros
-----------
level int : Para qual layer será gerado o peso;\
low float : Valor mínimo dos pesos (0 por padrão);\
high float : Valor maximo dos pesos (1 por padrão).

Retorna
-----------
matriz dos pesos com dimenção da largura do layer[nivel] x layer[nivel-1]

### *NeuralNetwork.NeuralNetwork.fit*

```python
NeuralNetwork.NeuralNetwork.fit(X, y: np.ndarray, loss = MSE, loss_diff = DMSE, learn_rate = 0.1, batch = 10, epochs = 1000, tol = 1e-9, debug = False)
```

Treina a rede neural a partir dos dados X, y

Parâmetros
-----------
X np.ndarray : Matriz P x N dos dados de treino;\
y np.ndarray : Matriz S x N das saidas de treino;\
loss Callable : Função perda para aprendizado;\
loss_diff Callable : Derivada da função perda para backpropagation;\
learn_rate float : Taxa de aprendizado (0.1 por padrão);\
batch int : Quantidade esperada de elementos por balde (10 por padrão);\
epochs int : Quantidade de interações nos dados de treino (1000 por padrão)\
debug bool : Quando True mostra em qual epoca está e qual a perda do momento (False por padrão)

### *NeuralNetwork.NeuralNetwork.prevision*

```python
NeuralNetwork.NeuralNetwork.prevision(x, save_states = False, classify = False)
```

Calcula o valor de saida da rede neural com pesos e vieses atuais

Parâmetros
-----------
x float | np.ndarray : Vetor ou matriz que queremos prever o resultado;\
save_states bool : Define se queremos salvar os valores de cada layer ou não;\
classify bool : Define se queremos retornar o vetor de probabilidade ou atribuir o elemento a uma classe.

Retorna
-----------
Valor de saida da rede neural com entrada x
