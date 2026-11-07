# Módulo Não Supervisionado

Arquivo: `src/nao_supervisionado.py` (apenas NumPy)


## Funções auxiliares

- `validar_matriz(X)`: converte `X` para `float` 2D. Levanta `TypeError` se não for 2D ou estiver vazio, e `ValueError` se tiver `NaN` ou infinito.
- `inteiro_positivo(valor, nome)`: garante que `valor` é um inteiro >= 1. Levanta `ValueError` caso contrário.

## `Kmeans(k, max_iteracao=100)`

Agrupa as amostras em `k` clusters (algoritmo de Lloyd): sorteia `k` pontos distintos como centroides, atribui cada amostra ao centroide mais próximo e recalcula cada centroide como a média do seu cluster, até convergir ou atingir `max_iteracao`.

| Parâmetro | Descrição |
|---|---|
| `k` | Número de clusters (inteiro >= 1) |
| `max_iteracao` | Limite de iterações (inteiro >= 1) |

| Método | O que faz |
|---|---|
| `fit(X)` | Ajusta o modelo e guarda `centroides` `(k, n_atributos)`. Retorna `self`. Erro se `k` > nº de amostras ou nº de pontos distintos |
| `estimar(X)` | Retorna o índice do centroide mais próximo de cada amostra, formato `(n_amostras,)` |

Observação: a inicialização é aleatória (sem semente), então a numeração dos clusters pode mudar entre execuções.

```python
modelo = Kmeans(k=2).fit(X)
rotulos = modelo.estimar(X)
```

## `Pca()`

Encontra as direções de maior variância: centraliza os dados, calcula a covariância amostral e usa `np.linalg.eigh` para obter autovalores e autovetores, ordenados do maior para o menor.

| Atributo | Descrição |
|---|---|
| `media` | Média de cada atributo `(n_atributos,)` |
| `componentes` | Autovetores nas colunas `(n_atributos, n_atributos)` |
| `autovalores` | Variância de cada componente, em ordem decrescente |

| Método | O que faz |
|---|---|
| `fit(X)` | Calcula `media`, `componentes` e `autovalores`. Exige >= 2 amostras. Retorna `self` |
| `transformar(X, n_componentes=None)` | Centraliza e projeta nos primeiros `n_componentes` (todos se `None`). Retorna `(n_amostras, n_componentes)` |
| `fit_transform(X, n_componentes=None)` | `fit` seguido de `transformar` |

Observações: o sinal de cada componente é arbitrário, e o PCA não padroniza os dados (use o pré-processamento se os atributos tiverem escalas diferentes).

```python
pca = Pca()
Z = pca.fit_transform(X, n_componentes=2)
variancia_explicada = pca.autovalores / pca.autovalores.sum()
```

## Justificativas

- **Interface `fit` / `estimar` / `transformar`**: separa o aprendizado da aplicação em dados novos.
- **`eigh`**: a covariância é simétrica, então é mais estável que `eig`.
- **Validação das entradas**: erros claros em vez de resultados errados.

## Testes

No final do arquivo, em `if __name__ == "__main__":`. Rodar com `python src/nao_supervisionado.py`. Depois serão movidos para `tests/test_nao_supervisionado.py`.

- **K-means**: dois grupos bem separados devem cair em dois clusters diferentes.
- **PCA**: dados sobre a reta `y = 2x` devem ter o primeiro componente na direção `(1,2)/√5`, e a projeção com 1 componente deve ter formato `(200, 1)`.