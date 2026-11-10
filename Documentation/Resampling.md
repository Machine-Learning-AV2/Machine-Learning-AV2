# Módulo De Reamostragem e Seleção

Arquivo: `src/Resampling.py` (apenas NumPy)

Contém **K-fold e LOOCV** (Validação cruzada), **Bootstrap** (Reamostragem) e **Busca por hiperparâmetros** (grid search). `X` é sempre uma matriz `(n_amostras, n_atributos)`, caso os dados tenham apenas um preditor, deve ser um vetor coluna. 

## `KFold(model,X : np.ndarray, Y : np.ndarray, k: int = 5, hip: float = None)`

Divide os dados em `k` partições, treina o modelo em `k-1` partições e testa na partição restante, repetindo para cada uma. Retorna a média da estatística de interesse.

| Parâmetro | Descrição |
|---|---|
| `model` | Modelo a ser validado (função dessa biblioteca ou com mesmo padrão `model(hip), model.fit(X,Y) e model.predict(X)`) |
| `X` | Dados dos preditores |
| `Y` | Dados de saída |
| `k` | Número de folds (default: 5) |
| `hip` | Hiperparâmetro do modelo. Aceita apenas um valor como argumento fixo (default: None) |


| Método | O que faz |
|---|---|
| `folds()` | Aplica o Kfold. Aplica a mesma permutação aleatória em X e Y e divide essa permutação em k partições, treinando o modelo em k-1 partições e testando na partição seguinte, k vezes |


Observação: A permutação é feita antes da divisão em partições e é aleatória.

```python
kfold = KFold(ConstantModel, X, y, k=5)
mse = kfold.folds()
```

## `LOOCV(model, X : np.ndarray, Y : np.ndarray,hip = None)`

Faz o mesmo que KFold, mas com `k = n_amostras`. Cada amostra é usada como teste uma vez.

| Atributo | Descrição |
|---|---|
| `model` | Modelo a ser validado (função dessa biblioteca ou com mesmo padrão `model(hip), model.fit(X,Y) e model.predict(X)`) |
| `X` | Dados dos preditores |
| `Y` | Dados de saída |
| `hip` | Hiperparâmetro do modelo. Aceita apenas um valor como argumento fixo (default: None) |

| Método | O que faz |
|---|---|
| `LOO(X)` | Aplica o LOOCV. Treina o modelo em todos os dados exceto um, e testa no dado restante, repetindo para cada amostra |


Observações: Diferente do Kfold, não há permutação aleatória.

```python
loocv = LOOCV(ConstantModel, X, y)
mse = loocv.LOO()
```


## `Bootstrap(model, X : np.ndarray, Y : np.ndarray, n: int = 1000, stat = MSE, hip = None)`

Faz `n` reamostragens com reposição dos dados, treina o modelo em cada reamostragem e calcula a estatística de interesse. Retorna a média e o desvio padrão da estatística.

| Atributo | Descrição |
|---|---|
| `model` | Modelo a ser validado (função dessa biblioteca ou com mesmo padrão `model(hip), model.fit(X,Y) e model.predict(X)`) |
| `X` | Dados dos preditores |
| `Y` | Dados de saída |
| `n` | Número de reamostragens (default: 1000) |
| `stat` | Estatística a ser calculada (default: MSE) |
| `hip` | Hiperparâmetro do modelo. Aceita apenas um valor como argumento fixo (default: None) |

| Método | O que faz |
|---|---|
| `build_array()` | Realiza o bootstrap e retorna um array com a estatística de interesse de cada reamostragem |
| `mean()` | Calcula a média do array obtido |
| `std_dev()` | Calcula o desvio padrão do array obtido |
| `conf_int(alpha=0.05)` | Calcula o intervalo de confiança do array obtido |


Observações: A reamostragem é feita com aleatoriedade, podendo resultar em amostras repetidas.

```python
bootstrap = Bootstrap(ConstantModel, X, y)
mse = bootstrap.mean()
intervalo = bootstrap.conf_int(alpha=0.05)
std_dev = bootstrap.std_dev()
mean = bootstrap.mean()
```

## `BuscaHiperparametro(model, X : np.ndarray, Y : np.ndarray, hip_range: list, n_folds: int = 5, stat = MSE)`

Aplica Kfold ou LOOCV para cada valor de hiperparâmetro em `hip_range`, e retorna o valor que minimiza a estatística de interesse.

| Atributo | Descrição |
|---|---|
| `model` | Modelo a ser validado (função dessa biblioteca ou com mesmo padrão `model(hip), model.fit(X,Y) e model.predict(X)`) |
| `X` | Dados dos preditores |
| `Y` | Dados de saída |
| `hip_range` | Lista de valores de hiperparâmetros a serem testados |
| `n_folds` | Número de folds para a validação cruzada (default: 5) |
| `stat` | Estatística a ser calculada (default: MSE) |

| Método | O que faz |
|---|---|
| `cv_score(hipi)` | Calcula a acurácia do modelo para um determinado hiperparâmetro |
| `find()` | Encontra o hiperparâmetro que minimiza a estatística de interesse |

Observações: Diferente do Kfold, não há permutação aleatória.

```python
busca = BuscaHiperparametro(ConstantModel, X, y, hip_range=[0.1, 1, 10, 100])
best_hip = busca.find()
```
