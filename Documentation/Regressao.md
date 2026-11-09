# MÓDULO DE CLASSIFICAÇÃO

**Arquivo:** `src/Regressao.py`

Este módulo implementa quatro algoritmos de Regressão:

- **Regressão Lasso** (*Lasso Regression*)
- **Regressão Polinomial** (*Polynomial Regression*)
- **Regressão Por Mínimos Quadrados** (*Least Squares Regression*)
- **Regressão Ridge** (*Ridge Regression*)

Observação: Como os algoritmos funcionam está melhor explicado no docstring da classe.

## 1. Métodos comuns às classes

Os métodos descritos nesta seção são aplicáveis a todas as classes do módulo `Regression.py`. Para simplificar, utilizamos a classe `Regressao` como referência genérica a qualquer modelo de Regressão.

**Importante:** todos os dados de entrada devem ser do tipo `np.ndarray`.

### `Regressao.fit(X, Y) -> None`

Treina o modelo utilizando os dados fornecidos.

**Parâmetros:**

- `X`: matriz de preditores observados no conjunto de treinamento. Cada linha representa uma observação individual, enquanto cada coluna representa uma variável preditora.
- `Y`: vetor contendo as classes correspondentes às observações de `X`. Cada elemento deve corresponder à classe da linha de mesmo índice na matriz `X`.

**Retorno:** não possui retorno.

Uma vez treinado, não é necessário executar novamente esse método para realizar outras previsões com o mesmo modelo.

### `Regressao.predict(X) -> np.ndarray`

Realiza previsões de Regressão para novas observações.

**Parâmetro:**

- `X`: matriz contendo os preditores das observações cujas classes desejamos estimar. Cada linha representa uma observação, enquanto cada coluna representa uma variável preditora.

**Retorno:** um `np.ndarray` contendo as classes atribuídas a cada observação.

**Observações:**

- O modelo deve ter sido previamente treinado com `model.fit()`.
- Alguns algoritmos possuem condições específicas para funcionarem satisfatoriamente. Recomenda-se conhecer as particularidades de cada método antes de utilizá-lo.

---

## 2. Regressão Lasso

**Classe:** `Lasso(Regressao)`

Implementa o algoritmo de Regressão Lasso por descida cíclica de coordenadas.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `alpha` | Parâmetro de regularização L1, que controla a penalização aplicada à soma dos valores absolutos dos coeficientes. Regularizações com valores maiores aumenta viés e diminui a variância, portanto é interessantes escolher um alpha levando em conta o trade-off viés-variância. |
| `max_inter` | Número máximo de iterações executadas pelo algoritmo para encontrar a resposta. |
| `tol` | Tolerância usada para verificar convergência. |

---

## 3. Regressão Polinomial

**Classe:** `RegressaoPolinomial()`

Implementa um algoritmo que faz regressão polinomial de uma variável explicativa para um polinômio de grau d.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `grau` | O grau do polinômio que tenta-se encontrar. |

---

## 4. Regressão por Mínimos Quadrados

**Classe:** `MinimosQuadrados()`

Implementa o Método dos Mínimos Quadrados na regressão, ou seja, mínimiza a soma dos quadrados dos resíduos.

Não precisa de nenhum parâmetro para inicialização.

---

## 5. Regressão Ridge

**Classe:** `Ridge()`

Implementa a Regressão Ridge, que ao invês de ter relarização linear(Lasso) tem regularização quadrática.

### Parâmetros de inicialização

| Parâmetro | Descrição |
|---|---|
| `alpha` | Parâmetro de reguralização L2, que controla penalidade com a magnitude do coeficiente. Valores maiores aumentam a regularização. |

---

## 6. Exemplo de utilização

A seguir, apresentamos um exemplo de utilização do módulo com dados artificiais. Serão demonstrados quatro métodos: Mínimos Quadrados Ordinários (MQO), Ridge, Lasso e Regressão Polinomial.

### 6.1. Preparação dos dados

```python
import numpy as np

# Dados de treinamento
X_train = np.array([
    [-3.0], [-2.5], [-2.0], [-1.5],
    [-1.0], [-0.5], [0.0], [0.5],
    [1.0], [1.5], [2.0], [2.5], [3.0]
])

# Variável resposta com relação quadrática
y_train = 2 + 3 * X_train[:, 0] + 0.5 * X_train[:, 0]**2

# Dados de teste
X_test = np.array([
    [-2.25],
    [-1.25],
    [0.25],
    [1.25],
    [2.25]
])
```

### 6.2. Treinamento e previsão

```python
# Mínimos Quadrados Ordinários
modelo_mqo = MinimosQuadrados()
modelo_mqo.fit(X_train, y_train)
pred_mqo = modelo_mqo.predict(X_test)

# Regressão Ridge
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)
pred_ridge = modelo_ridge.predict(X_test)

# Regressão Lasso
modelo_lasso = Lasso(alpha=1.0)
modelo_lasso.fit(X_train, y_train)
pred_lasso = modelo_lasso.predict(X_test)

# Regressão Polinomial
modelo_polinomial = RegressaoPolinomial(grau=2)
modelo_polinomial.fit(X_train, y_train)
pred_polinomial = modelo_polinomial.predict(X_test)

# Exibição das previsões
print("Mínimos Quadrados:", pred_mqo)
print("Ridge:", pred_ridge)
print("Lasso:", pred_lasso)
print("Polinomial:", pred_polinomial)
```

### 6.3. Resultado esperado
As previsões representam os valores estimados pelos diferentes modelos para as observações do conjunto de teste.

Como os dados foram construídos a partir de uma função quadrática de grau 2 deve recuperar exatamente a relação utilizada na geração dos dados, considerando a ausência de regularizações.

Os demais métodos apresentam resultados que dependem de suas respectivas formulações e parâmetros de regularização.

---

