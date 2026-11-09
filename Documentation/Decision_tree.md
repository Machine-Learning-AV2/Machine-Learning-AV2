# Módulo Árvores de Decisão

Arquivo: `src/arvores_decisao.py` (apenas NumPy)

O módulo implementa dois modelos de árvores de decisão: classificação e regressão. Ambos utilizam divisões binárias recursivas dos dados, escolhendo a característica e o limite que minimizam uma medida de impureza ou erro.

Para os dois modelos, consideramos `X` como a matriz de características dos exemplos e `y` como o vetor de valores associados a cada exemplo. Cada linha de `X` corresponde a um exemplo, e cada coluna representa uma característica.

## *DecisionTreeClassification*

### Parâmetros
__________________________

`max_depth` : Profundidade máxima permitida para a árvore. O valor padrão é 5;

`min_samples` : Número mínimo de exemplos necessário para que um nó possa continuar sendo dividido. O valor padrão é 2.

Inicialmente, a árvore não possui raiz (`root = None`).

### Definição
__________________________

Implementa uma árvore de decisão para problemas de classificação. O algoritmo realiza divisões binárias sucessivas dos dados, buscando separar exemplos pertencentes a classes diferentes.

A qualidade de cada divisão é avaliada pelo índice Gini, sendo escolhida aquela que produz a menor impureza ponderada entre os dois grupos resultantes.

### *DecisionTreeClassification._gini*

#### Parâmetros

`y` : Vetor contendo as classes dos exemplos de um determinado nó.

Retorna um `float`.

#### Definição

Calcula o índice Gini, que mede a impureza de um conjunto de exemplos.

Seja $p_i$ a proporção de exemplos pertencentes à classe $i$, e $k$ o número de classes distintas presentes no conjunto.

$$Gini(y)=1-\sum_{i=1}^{k}p_i^2$$

Quanto menor o índice Gini, mais homogêneo é o conjunto. Quando todos os exemplos pertencem à mesma classe, o índice é zero.

Se o vetor estiver vazio, retorna zero.

```python
y = np.array([0, 0, 0, 1, 1])

_gini(y) # 0.48
```

### *DecisionTreeClassification._best_division*

#### Parâmetros

`X` : Matriz contendo as características dos exemplos;

`y` : Vetor contendo as classes dos exemplos.

Retorna uma tupla `(j, limit)` ou `None`, em que `j` representa o índice da característica escolhida e `limit` representa o limite da divisão.

#### Definição

Busca a melhor divisão possível para os exemplos de um nó, utilizando o índice Gini ponderado como critério.

Inicialmente, calcula a impureza Gini do conjunto original. Em seguida, percorre todas as características e testa diferentes limites de separação.

Para cada característica, são considerados os valores distintos em ordem crescente. Os limites testados correspondem às médias entre valores consecutivos:

$$t_i=\frac{v_i+v_{i+1}}{2}$$

Para cada limite $t$, os exemplos são separados em dois grupos:

$$L=\{x_i\in X:x_{ij}\leq t\}$$

$$R=\{x_i\in X:x_{ij}>t\}$$

A impureza da divisão é calculada pela média ponderada dos índices Gini dos dois grupos:

$$Gini_{\text{divisão}}=
\frac{|L|}{n}Gini(L)+
\frac{|R|}{n}Gini(R)$$

O algoritmo escolhe a divisão que apresentar o menor valor de Gini ponderado, desde que esse valor seja estritamente menor que a impureza original do conjunto.

Caso nenhuma divisão melhore a impureza, retorna `None`.

### *DecisionTreeClassification._build_tree*

#### Parâmetros

`X` : Matriz contendo as características dos exemplos;

`y` : Vetor contendo as classes dos exemplos;

`depth` : Profundidade atual do nó na árvore. Seu valor inicial é zero.

Retorna um dicionário que representa um nó da árvore.

#### Definição

Constrói a árvore de decisão recursivamente, utilizando `_best_division` para determinar as melhores divisões.

Inicialmente, identifica a classe majoritária do conjunto, que será utilizada como previsão caso o nó se torne uma folha.

O algoritmo interrompe a divisão quando alguma das seguintes condições é satisfeita:

- Todos os exemplos pertencem à mesma classe;
- A profundidade máxima da árvore é atingida;
- O número de exemplos é menor que `min_samples`;
- Não existe uma divisão capaz de reduzir a impureza Gini.

Quando uma dessas condições é satisfeita, é criado um nó folha, representado por:

```python
{"folha": True, "valor": class_majority}
```

Caso contrário, são armazenados a característica e o limite escolhidos, e a função é chamada recursivamente para construir as subárvores esquerda e direita.

Os nós internos são representados por:

```python
{
    "folha": False,
    "caracteristica": j,
    "limite": limit,
    "esquerda": ...,
    "direita": ...
}
```

### *DecisionTreeClassification.adjust*

#### Parâmetros

`X` : Matriz bidimensional contendo as características dos exemplos de treinamento;

`y` : Vetor unidimensional contendo as classes correspondentes aos exemplos.

Retorna o próprio objeto da classe `DecisionTreeClassification`.

#### Definição

Realiza o treinamento da árvore de decisão.

Inicialmente, converte `X` para uma matriz NumPy de valores reais e `y` para um vetor NumPy.

Em seguida, verifica se os dados possuem dimensões adequadas, se possuem o mesmo número de exemplos e se não estão vazios. Também verifica se existe pelo menos uma característica.

Caso alguma condição não seja satisfeita, lança um `ValueError`.

Por fim, chama `_build_tree` para construir a árvore e armazena o resultado no atributo `root`.

```python
modelo = DecisionTreeClassification(max_depth=5, min_samples=2)

modelo.adjust(X_train, y_train)
```

### *DecisionTreeClassification._predict_sample*

#### Parâmetros

`x` : Vetor contendo as características de um único exemplo;

`no` : Dicionário que representa o nó atual da árvore.

Retorna o valor da classe prevista para o exemplo.

#### Definição

Realiza a previsão de um único exemplo percorrendo a árvore recursivamente.

Se o nó atual for uma folha, retorna a classe armazenada em `valor`.

Caso contrário, verifica se a característica indicada por `caracteristica` é menor ou igual ao limite armazenado em `limite`.

Se a condição for verdadeira, continua a busca pela subárvore esquerda. Caso contrário, continua pela subárvore direita.

O processo se repete até que uma folha seja encontrada.

### *DecisionTreeClassification.predict*

#### Parâmetros

`X` : Matriz bidimensional contendo as características dos exemplos que serão classificados.

Retorna um `np.ndarray` contendo as classes previstas.

#### Definição

Realiza a previsão das classes para todos os exemplos de uma matriz.

Inicialmente, verifica se a árvore foi treinada, ou seja, se o atributo `root` não está vazio. Também verifica se `X` é bidimensional.

Em seguida, utiliza `_predict_sample` para realizar a previsão individual de cada exemplo, retornando um vetor NumPy com os resultados.

```python
modelo = DecisionTreeClassification()
modelo.adjust(X_train, y_train)

previsoes = modelo.predict(X_test)
```

---

## *DecisionTreeRegression*

### Parâmetros
__________________________

`max_depth` : Profundidade máxima permitida para a árvore. O valor padrão é 5;

`min_samples` : Número mínimo de exemplos necessário para que um nó possa continuar sendo dividido. O valor padrão é 2.

Inicialmente, a árvore não possui raiz (`root = None`).

### Definição
__________________________

Implementa uma árvore de decisão para problemas de regressão. Assim como na classificação, o algoritmo realiza divisões binárias sucessivas dos dados, buscando formar grupos com valores de saída semelhantes.

A qualidade de cada divisão é avaliada pelo erro quadrático médio (MSE), sendo escolhida aquela que produz o menor erro ponderado entre os dois grupos resultantes.

### *DecisionTreeRegression._mse*

#### Parâmetros

`y` : Vetor contendo os valores numéricos associados aos exemplos de um determinado nó.

Retorna um `float`.

#### Definição

Calcula o erro quadrático médio dos valores em relação à sua média.

Seja $\bar y$ a média dos valores do vetor:

$$\bar y=\frac{1}{n}\sum_{i=1}^{n}y_i$$

O MSE é definido por:

$$MSE(y)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\bar y)^2$$

Quanto menor o MSE, menor a dispersão dos valores em torno da média. Quando todos os valores são iguais, o MSE é zero.

Se o vetor estiver vazio, retorna zero.

```python
y = np.array([2, 4, 6])

_mse(y) # 2.666...
```

### *DecisionTreeRegression._best_division*

#### Parâmetros

`X` : Matriz contendo as características dos exemplos;

`y` : Vetor contendo os valores numéricos associados aos exemplos.

Retorna uma tupla `(j, limit)` ou `None`, em que `j` representa o índice da característica escolhida e `limit` representa o limite da divisão.

#### Definição

Busca a melhor divisão possível para os exemplos de um nó, utilizando o MSE ponderado como critério.

O algoritmo percorre todas as características e testa os limites correspondentes às médias entre valores consecutivos distintos de cada característica:

$$t_i=\frac{v_i+v_{i+1}}{2}$$

Para cada limite, os exemplos são separados em dois grupos:

$$L=\{x_i\in X:x_{ij}\leq t\}$$

$$R=\{x_i\in X:x_{ij}>t\}$$

A qualidade da divisão é calculada pela média ponderada dos erros quadráticos médios dos dois grupos:

$$MSE_{\text{divisão}}=
\frac{|L|}{n}MSE(L)+
\frac{|R|}{n}MSE(R)$$

O algoritmo escolhe a divisão que apresentar o menor erro ponderado, desde que esse valor seja estritamente menor que o erro original do conjunto.

Caso nenhuma divisão melhore o erro, retorna `None`.

### *DecisionTreeRegression._build_tree*

#### Parâmetros

`X` : Matriz contendo as características dos exemplos;

`y` : Vetor contendo os valores numéricos associados aos exemplos;

`depth` : Profundidade atual do nó na árvore. Seu valor inicial é zero.

Retorna um dicionário que representa um nó da árvore.

#### Definição

Constrói a árvore de decisão recursivamente, utilizando `_best_division` para determinar as melhores divisões.

Inicialmente, calcula a média dos valores de `y`, que será utilizada como previsão caso o nó se torne uma folha.

O algoritmo interrompe a divisão quando alguma das seguintes condições é satisfeita:

- A profundidade máxima da árvore é atingida;
- O número de exemplos é menor que `min_samples`;
- Todos os valores de `y` são iguais;
- Não existe uma divisão capaz de reduzir o MSE.

Quando uma dessas condições é satisfeita, é criado um nó folha, representado por:

```python
{"folha": True, "valor": mean}
```

Caso contrário, são armazenados a característica e o limite escolhidos, e a função é chamada recursivamente para construir as subárvores esquerda e direita.

Os nós internos possuem a mesma estrutura utilizada na árvore de classificação.

### *DecisionTreeRegression.adjust*

#### Parâmetros

`X` : Matriz bidimensional contendo as características dos exemplos de treinamento;

`y` : Vetor unidimensional contendo os valores numéricos correspondentes aos exemplos.

Retorna o próprio objeto da classe `DecisionTreeRegression`.

#### Definição

Realiza o treinamento da árvore de decisão.

Inicialmente, converte `X` e `y` para arrays NumPy de valores reais.

Em seguida, verifica se os dados possuem dimensões adequadas, se possuem o mesmo número de exemplos e se não estão vazios. Também verifica se existe pelo menos uma característica.

Caso alguma condição não seja satisfeita, lança um `ValueError`.

Por fim, chama `_build_tree` para construir a árvore e armazena o resultado no atributo `root`.

```python
modelo = DecisionTreeRegression(max_depth=5, min_samples=2)

modelo.adjust(X_train, y_train)
```

### *DecisionTreeRegression._predict_sample*

#### Parâmetros

`x` : Vetor contendo as características de um único exemplo;

`no` : Dicionário que representa o nó atual da árvore.

Retorna o valor numérico previsto para o exemplo.

#### Definição

Realiza a previsão de um único exemplo percorrendo a árvore recursivamente.

Se o nó atual for uma folha, retorna a média armazenada em `valor`.

Caso contrário, verifica se a característica indicada por `caracteristica` é menor ou igual ao limite armazenado em `limite`.

Se a condição for verdadeira, continua a busca pela subárvore esquerda. Caso contrário, continua pela subárvore direita.

O processo se repete até que uma folha seja encontrada.

### *DecisionTreeRegression.predict*

#### Parâmetros

`X` : Matriz bidimensional contendo as características dos exemplos que serão utilizados para realizar as previsões.

Retorna um `np.ndarray` contendo os valores numéricos previstos.

#### Definição

Realiza a previsão dos valores para todos os exemplos de uma matriz.

Inicialmente, verifica se a árvore foi treinada, ou seja, se o atributo `root` não está vazio. Também verifica se `X` é bidimensional e se possui pelo menos uma característica.

Em seguida, utiliza `_predict_sample` para realizar a previsão individual de cada exemplo, retornando um vetor NumPy com os resultados.

```python
modelo = DecisionTreeRegression()
modelo.adjust(X_train, y_train)

previsoes = modelo.predict(X_test)
```

---

## Exemplos de utilização

### Classificação

```python
X_train = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [7.0, 1.0],
    [7.5, 2.0]
])

y_train = np.array([0, 0, 1, 1])

X_test = np.array([
    [1.8, 1.5],
    [7.8, 1.5]
])

modelo = DecisionTreeClassification()
modelo.adjust(X_train, y_train)

previsoes = modelo.predict(X_test)
```

### Regressão

```python
X_train = np.array([
    [1.0, 1.0],
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0]
])

y_train = np.array([4.0, 5.0, 7.0, 9.0])

X_test = np.array([
    [1.5, 1.0],
    [3.5, 2.5]
])

modelo = DecisionTreeRegression()
modelo.adjust(X_train, y_train)

previsoes = modelo.predict(X_test)
```
