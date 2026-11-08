# MÓDULO DE PRÉ-PROCESSAMENTO

**Arquivo:** `src/preprocessing.py`

Este módulo prepara os dados antes de entregá-los aos modelos da biblioteca. Ele implementa duas funções e quatro classes:

- **Divisão em treino e teste** (`train_test_split`)
- **Validação da matriz de dados** (`validate_matrix`)
- **Padronização por média-variância** (*StandardScaler*)
- **Normalização min-max** (*MinMaxScaler*)
- **Codificação de um vetor de variáveis qualitativa/categóricas** (*CategoricalEncoder*)
- **Codificação de colunas qualitativas/categóricas de uma matriz** (*ColumnCategoricalEncoder*)

## 1. Métodos comuns às classes

Os métodos descritos nesta seção são aplicáveis a todas as classes do módulo `preprocessing.py`. Para simplificar, utilizaremos `transformer` como referência genérica a qualquer uma dessas classes.

**Importante:** os parâmetros da transformação (médias, mínimos, categorias...) devem ser aprendidos **apenas com os dados de treinamento**. Os dados de teste devem ser transformados com esses mesmos parâmetros. Caso contrário, informações do conjunto de teste vazam para o modelo e o erro estimado fica otimista demais ou passamos a medir erro em uma base incomparável com a que foi utilizado no treino.

### `transformer.fit(X) -> transformer`

Aprende os parâmetros da transformação a partir dos dados fornecidos.

**Parâmetro:**

- `X`: matriz de dados de treinamento. Cada linha representa uma observação individual, enquanto cada coluna representa uma variável preditora. No `CategoricalEncoder`, é um vetor com as categorias de uma única variável.

**Retorno:** o próprio objeto (`self`), o que permite encadear chamadas, como em `transformer.fit(X).transform(X)`.

Uma vez ajustado, não é necessário executar novamente esse método para transformar outros dados com os mesmos parâmetros.

### `transformer.transform(X) -> np.ndarray`

Aplica a transformação utilizando os parâmetros aprendidos no `fit()`.

**Parâmetro:**

- `X`: dados de treinamento ou de teste, com o mesmo número de colunas usado no `fit()`.

**Retorno:** um `np.ndarray` com os dados transformados.

**Observações:**

- O objeto deve ter sido previamente ajustado com `transformer.fit()`; caso contrário, é levantado `RuntimeError`.
- Dados com número de colunas diferente do usado no `fit()` levantam `ValueError`.

### `transformer.fit_transform(X) -> np.ndarray`

Executa `fit()` seguido de `transform()` nos mesmos dados. Deve ser usado nos dados de treinamento.

---

## 2. Divisão em treino e teste

**Função:** `train_test_split(X, y, test_size=0.25, seed=None)`

Embaralha as observações e as divide em um conjunto de treinamento e um de teste, mantendo cada linha de `X` pareada com o elemento correspondente de `y`.

### Parâmetros

| Parâmetro | Descrição |
|---|---|
| `X` | Matriz de variáveis independentes, com uma observação por linha. |
| `y` | Vetor de variáveis dependentes, com o mesmo número de observações de `X`. |
| `test_size` | Fração das observações destinada ao conjunto de teste. Deve ser um `float` entre 0 e 1. |
| `seed` | Semente do gerador aleatório. A mesma semente produz sempre a mesma divisão, tornando os resultados reprodutíveis. Quando `None`, a divisão muda a cada execução. |

**Retorno:** a tupla `X_train, X_test, y_train, y_test`.

Uma divisão que deixaria o treinamento ou o teste vazio levanta `ValueError`.

---

## 3. Validação da matriz de dados

**Função:** `validate_matrix(X)`

Função auxiliar utilizada pelas demais classes. Converte `X` para uma matriz `float` (um array numpy bidimensional), no formato (observações, variáveis). Um vetor unidimensional é tratado como uma única coluna ainda em um array numpy bidimensional.

Levanta `TypeError` se `X` não for numérico e `ValueError` se `X` estiver vazio ou contiver `NaN` ou infinito.

---

## 4. Padronização

**Classe:** `StandardScaler()`

Padroniza cada variável preditora pela fórmula `z = (x - média) / desvio padrão`, de modo que cada coluna passe a ter média 0 e desvio padrão 1.

É recomendada para modelos sensíveis à escala das variáveis: KNN e K-means (distâncias), PCA (variância), Ridge e Lasso (a penalidade trata todos os coeficientes igualmente) e redes neurais e regressão logística (a descida de gradiente converge melhor).

Colunas constantes (desvio padrão 0) são transformadas em 0, evitando divisão por zero.

### Parâmetros de inicialização

Não possui parâmetros de inicialização.

### Atributos

| Atributo | Descrição |
|---|---|
| `mean` | Média de cada variável nos dados de treinamento. |
| `std` | Desvio padrão de cada variável nos dados de treinamento (populacional, dividido por `n`). |

---

## 5. Normalização min-max

**Classe:** `MinMaxScaler()`

Normaliza cada variável preditora pela fórmula `x' = (x - mín) / (máx - mín)`, levando cada coluna para o intervalo [0, 1].

Dados novos fora da faixa observada no treinamento podem resultar em valores fora de [0, 1]. Colunas constantes são transformadas em 0.

### Parâmetros de inicialização

Não possui parâmetros de inicialização.

### Atributos

| Atributo | Descrição |
|---|---|
| `data_min` | Mínimo de cada variável nos dados de treinamento. |
| `data_max` | Máximo de cada variável nos dados de treinamento. |

---

## 6. Codificação de uma variável qualitativa

**Classe:** `CategoricalEncoder()`

Transforma um vetor de categorias (por exemplo, cores ou cidades) em uma matriz de variáveis indicadoras (*dummy variables*) com valores 0 e 1, uma coluna por categoria. Isso permite utilizar variáveis qualitativas em modelos que aceitam apenas números.

Categorias presentes nos dados de teste que não apareceram no `fit()` levantam `ValueError`.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `drop_first` | Quando `True`, gera `K-1` colunas para `K` categorias, deixando a primeira categoria como referência (linha toda zero). Deve ser usado na regressão linear com intercepto: com as `K` colunas, a soma delas é sempre 1 e fica colinear com o intercepto. Por padrão, `False`. |

### Atributos

| Atributo | Descrição |
|---|---|
| `categories` | Categorias encontradas no `fit()`, em ordem crescente. A ordem das colunas geradas segue essa ordem. |

---

## 7. Codificação de colunas qualitativas de uma matriz

**Classe:** `ColumnCategoricalEncoder()`

Aplica o `CategoricalEncoder` às colunas escolhidas de uma matriz inteira. Cada coluna escolhida é substituída, na mesma posição, pelas suas colunas indicadoras; as demais colunas são mantidas como estão.

**Importante:** uma matriz que mistura texto e números deve ser criada com `_as_matrix` para ser convertida em um array numpy 2D com `dtype=object`. Caso contrário, o NumPy converte todos os valores em texto se a especificação `dtype=object` não for explícita.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `columns` | Lista com os índices (entre `0` e `m-1`, onde `m` é o número de colunas) das colunas qualitativas a codificar. Quando `None`, nenhuma coluna é codificada. |
| `drop_first` | Repassado a cada `CategoricalEncoder`. Por padrão, `False`. |

### Atributos

| Atributo | Descrição |
|---|---|
| `encoders` | Dicionário `{índice da coluna: instância de CategoricalEncoder ajustado para essa coluna}`. |
| `n_features` | Número de colunas da matriz usada no `fit()`. |

**Observações:**

- O método aplica `validate_matrix()` as colunas da matriz que não foram passadas para ser codificadas como dummy variables.
- Consequentemente, as colunas que não estão em `columns` devem ser numéricas, sem `NaN` ou infinito. Uma coluna de texto que não foi listada levanta `TypeError`.
- Índices fora do intervalo ou repetidos levantam `ValueError`.

---

## 8. Exemplo de utilização

A seguir, apresentamos um exemplo de preparação de dados artificiais com três variáveis preditoras: idade, estado (qualitativa) e renda.

### 8.1. Preparação dos dados

```python
import numpy as np

# Colunas: idade, estado, renda
X = np.array([
    [25, 'SP', 3000.0], [32, 'RJ', 4500.0],
    [47, 'SP', 8000.0], [51, 'MG', 6200.0],
    [23, 'RJ', 2800.0], [38, 'MG', 5100.0],
    [29, 'SP', 3900.0], [44, 'RJ', 7000.0]
], dtype=object)

# Classes correspondentes às observações
y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
```

### 8.2. Divisão, codificação e normalização

```python
# Divisão em treino (6 observações) e teste (2 observações)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=0)

# Codificação da coluna 1 (estado), com 'MG' como categoria de referência
encoder = ColumnCategoricalEncoder(columns=[1], drop_first=True)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Normalização com os mínimos e máximos do treinamento
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_test)
```

### 8.3. Resultado esperado

Considerando os dados artificiais apresentados, o resultado esperado (arredondado para duas casas decimais) é:

```text
[[ 0.85  0.    1.    1.25]
 [-0.08  1.    0.   -0.05]]
```

As colunas são, nesta ordem: idade, indicadora de RJ, indicadora de SP e renda. A primeira linha é uma pessoa de SP e a segunda, do RJ. Os valores fora de [0, 1] aparecem porque a idade e a renda dessas observações de teste estão fora da faixa vista no treinamento.

Depois desse passo, `X_train` e `X_test` podem ser utilizados diretamente pelos modelos da biblioteca, como `model.train(X_train, y_train)` no módulo de classificação.

---

**Nota:** rodar `python preprocessing.py` dentro de `src/` executa os 23 casos de teste do módulo, cada um imprimindo `Passed` ou `Failed`.
