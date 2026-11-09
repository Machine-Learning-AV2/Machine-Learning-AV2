# MÓDULO DE CLASSIFICAÇÃO

**Arquivo:** `src/classification.py`

Este módulo implementa quatro algoritmos de classificação:

- **Regressão Logística** (*Logistic Regression*)
- **KNN** (*K-Nearest Neighbors*)
- **LDA** (*Linear Discriminant Analysis*)
- **QDA** (*Quadratic Discriminant Analysis*)

## 1. Métodos comuns às classes

Os métodos descritos nesta seção são aplicáveis a todas as classes do módulo `classification.py`. Para simplificar, utilizaremos `model` como referência genérica a qualquer modelo de classificação.

**Importante:** todos os dados de entrada devem ser do tipo `np.ndarray`.

### `model.fit(X, Y) -> None`

Treina o modelo utilizando os dados fornecidos.

**Parâmetros:**

- `X`: matriz de preditores observados no conjunto de treinamento. Cada linha representa uma observação individual, enquanto cada coluna representa uma variável preditora.
- `Y`: vetor contendo as classes correspondentes às observações de `X`. Cada elemento deve corresponder à classe da linha de mesmo índice na matriz `X`.

**Retorno:** não possui retorno (`None`).

Uma vez treinado, não é necessário executar novamente esse método para realizar outras previsões com o mesmo modelo.

### `model.predict(X) -> np.ndarray`

Realiza previsões de classificação para novas observações.

**Parâmetro:**

- `X`: matriz contendo os preditores das observações cujas classes desejamos estimar. Cada linha representa uma observação, enquanto cada coluna representa uma variável preditora.

**Retorno:** um `np.ndarray` contendo as classes atribuídas a cada observação.

**Observações:**

- O modelo deve ter sido previamente treinado com `model.fit()`.
- Alguns algoritmos possuem condições específicas para funcionarem satisfatoriamente. Recomenda-se conhecer as particularidades de cada método antes de utilizá-lo.

---

## 2. Regressão Logística

**Classe:** `Logistic_Regression()`

Implementa o algoritmo de Regressão Logística, utilizando descida de gradiente para otimização.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `learning_rate` | Taxa de aprendizado utilizada na descida de gradiente. Valores muito pequenos podem tornar a convergência lenta, enquanto valores muito altos podem causar divergência. |
| `iterations` | Número de iterações executadas pelo algoritmo de descida de gradiente. |
| `intercept` | Define se o modelo utilizará um termo independente (*bias*). Quando `False`, esse termo é desativado. |

---

## 3. KNN — K-Nearest Neighbors

**Classe:** `KNN()`

Implementa o algoritmo dos K vizinhos mais próximos. A classe prevista para uma observação é determinada pela classe majoritária entre seus vizinhos mais próximos.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `K` | Número de vizinhos considerados para cada previsão. A classe majoritária entre esses vizinhos será atribuída à observação. |
| `p` | Expoente utilizado no cálculo da distância de Minkowski (norma Lp). Por padrão, utiliza a distância euclidiana (L2), correspondente a `p=2`. |
| `normalize` | Quando `True`, normaliza os preditores para média 0 e desvio padrão 1, evitando que variáveis com escalas maiores dominem o cálculo das distâncias. |

---

## 4. LDA — Linear Discriminant Analysis

**Classe:** `LDA()`

Implementa a Análise Discriminante Linear (*Linear Discriminant Analysis*).

Por se tratar de um método com configuração fixa nesta implementação, não possui parâmetros de inicialização.

---

## 5. QDA — Quadratic Discriminant Analysis

**Classe:** `QDA()`

Implementa a Análise Discriminante Quadrática (*Quadratic Discriminant Analysis*).

Por se tratar de um método com configuração fixa nesta implementação, não possui parâmetros de inicialização.

---

## 6. Exemplo de utilização

A seguir, apresentamos um exemplo de utilização do módulo com dados artificiais contendo três classes distintas (`0`, `1` e `2`).

### 6.1. Preparação dos dados

```python
import numpy as np

# Dados de treinamento
X_fit = np.array([
    [1.0, 1.0], [1.5, 2.0],
    [2.0, 1.0], [2.5, 2.5],
    [3.0, 1.5], [1.0, 3.0],
    [2.0, 3.0], [3.0, 2.5],
    [3.5, 1.0], [2.5, 3.5],

    [7.0, 1.0], [7.5, 2.0],
    [8.0, 1.0], [8.5, 2.5],
    [9.0, 1.5], [7.0, 3.0],
    [8.0, 3.0], [9.0, 2.5],
    [9.5, 1.0], [8.5, 3.5],

    [4.0, 7.0], [4.5, 8.0],
    [5.0, 7.0], [5.5, 8.5],
    [6.0, 7.5], [4.0, 9.0],
    [5.0, 9.0], [6.0, 8.5],
    [6.5, 7.0], [5.5, 9.5]
])

# Classes correspondentes aos dados de treinamento
y_fit = np.array([0]*10 + [1]*10 + [2]*10)

# Dados de teste
X_test = np.array([
    [1.8, 1.8], [2.8, 2.0],
    [1.5, 2.8], [3.2, 3.0],
    [2.0, 2.5],

    [7.8, 1.8], [8.8, 2.0],
    [7.5, 2.8], [9.2, 3.0],
    [8.0, 2.5],

    [4.8, 7.8], [5.8, 8.0],
    [4.5, 8.8], [6.2, 8.0],
    [5.0, 8.5]
])
```

### 6.2. Treinamento e previsão

```python
# Exemplo utilizando KNN
model = KNN(K=3)

# Treinamento
model.fit(X_fit, y_fit)

# Previsões
print(model.predict(X_test))
```

### 6.3. Resultado esperado

Considerando os dados artificiais apresentados, o resultado esperado é:

```text
[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]
```

Esse resultado representa a classificação das 15 observações do conjunto de teste, na ordem em que foram fornecidas.

---

**Nota:** o exemplo utiliza KNN apenas para ilustrar o fluxo de utilização. Os demais classificadores seguem a mesma interface de treinamento e previsão (`fit()` e `predict()`).
