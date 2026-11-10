# Módulo Não Supervisionado

Arquivo: `src/UnsupervisedLearning.py` (apenas NumPy)

Contém **K-means** (agrupamento) e **PCA** (redução de dimensionalidade). `X` é sempre uma matriz `(n_amostras, n_atributos)`. Chamar `predict` ou `transform` antes do `fit` levanta `RuntimeError`.


## Funções auxiliares

- `validate_matrix(X)`: converte `X` para `float` 2D. Levanta `TypeError` se não for 2D ou estiver vazio, e `ValueError` se tiver `NaN` ou infinito.
- `validate_positive_integer(valor, nome)`: garante que `valor` é um inteiro >= 1. Levanta `ValueError` caso contrário.

## `KMeans(k, max_iterations=100, seed=0)`

Agrupa as amostras em `k` clusters (algoritmo de Lloyd): sorteia `k` pontos distintos como centroides, atribui cada amostra ao centroide mais próximo e recalcula cada centroide como a média do seu cluster, até convergir ou atingir `max_iterations`.

| Parâmetro | Descrição |
|---|---|
| `k` | Número de clusters (inteiro >= 1) |
| `max_iterations` | Limite de iterações (inteiro >= 1) |
| `seed` | Semente do gerador aleatório da inicialização |

| Método | O que faz |
|---|---|
| `fit(X)` | Ajusta o modelo e guarda `centroids` `(k, n_atributos)`. Retorna `self`. Erro se `k` > nº de amostras ou nº de pontos distintos |
| `predict(X)` | Retorna o índice do centroide mais próximo de cada amostra, formato `(n_amostras,)` |

Observação: a inicialização usa a semente (`seed`), então o resultado é reproduzível entre execuções com os mesmos dados.

```python
modelo = KMeans(k=2).fit(X)
rotulos = modelo.predict(X)
```

## `PCA()`

Encontra as direções de maior variância: centraliza os dados, calcula a covariância amostral e usa `np.linalg.eigh` para obter autovalores e autovetores, ordenados do maior para o menor.

| Atributo | Descrição |
|---|---|
| `mean` | Média de cada atributo `(n_atributos,)` |
| `components` | Autovetores nas colunas `(n_atributos, n_atributos)` |
| `eigenvalues` | Variância de cada componente, em ordem decrescente |

| Método | O que faz |
|---|---|
| `fit(X)` | Calcula `mean`, `components` e `eigenvalues`. Exige >= 2 amostras. Retorna `self` |
| `transform(X, n_components=None)` | Centraliza e projeta nos primeiros `n_components` (todos se `None`). Retorna `(n_amostras, n_components)` |
| `fit_transform(X, n_components=None)` | `fit` seguido de `transform` |

Observações: o sinal de cada componente é arbitrário, e o PCA não padroniza os dados (use o pré-processamento se os atributos tiverem escalas diferentes).

```python
pca = PCA()
Z = pca.fit_transform(X, n_components=2)
variancia_explicada = pca.eigenvalues / pca.eigenvalues.sum()
```

## Justificativas

- **Interface `fit` / `predict` / `transform`**: separa o aprendizado da aplicação em dados novos.
- **`eigh`**: a covariância é simétrica, então é mais estável que `eig`.
- **Validação das entradas**: erros claros em vez de resultados errados.

## Testes

No final do arquivo, em `if __name__ == "__main__":`. Rodar com `python src/unsupervised_learning.py`. Depois serão movidos para `tests/test_nao_supervisionado.py`.

- **K-means**: dois grupos bem separados devem cair em dois clusters diferentes.
- **PCA**: dados sobre a reta `y = 2x` devem ter o primeiro componente na direção `(1,2)/√5`, e a projeção com 1 componente deve ter formato `(200, 1)`.
