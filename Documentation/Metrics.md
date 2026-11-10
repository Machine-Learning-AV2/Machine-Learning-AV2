# Módulo Metricas

Arquivo: `src/Metrics.py` (apenas NumPy)

Para todas as funções deste módulo, vamos considerar `x` como o vetor de valores previstos pelo método e `y` é o vetor de valores reais observados. Ambos devem ter as mesmas dimensões.

## *metricas.loss*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados;

`p` : Parâmetro que define a função loss;

`l` : Matriz de pesos para erros de classificação via função indicadora;

`subs` : Valor de peso padrão para casos não listados na matriz l.

Retorna um `float`

### Definição
__________________________
`metricas.loss` é a principal função do módulo, de modo que quase todas as demais funções recorrem a ela para realizar seus processamentos internos.

Seja $\hat{l}$ um função perda arbitrária:

$$loss(x,y) = \sum_{i=1}^n\hat{l}(y_i,x_i)$$

Até o momento, 3 classes de funções perda $\hat{l}$ foram implementadas. Qual função perda vai ser aplicada depende do parâmetro `p`.

Se `p` for um valor numérico maior ou igual a 1, a função perda $\hat{l}$ é a perda baseada na norma $L_p$.

$$\hat{l}(y_i,x_i) = |y_i-x_i|^p$$

``` python
x = np.array([1, 1])
y = np.array([2, 3])
rss = loss(x, y, 2) 
```

Se `p` for `'ind'`, a função perda utiliza a função indicadora. E o parâmetro `subs` define o valor de $k$.

$$
\hat{l}(y_i,x_i) = \begin{cases} 
0 & \text{se } y_i=x_i \\ 
k & \text{se } y_i\neq x_i 
\end{cases}
$$

Neste caso, também possuimos o parâmetro `l`. Este parâmetro deve ser uma matriz $L_{\alpha\times3}$ numpy com 3 colunas. Ele estabelece um peso para cada erro.

$$
\hat{l}(y_i,x_i) = \begin{cases}
0 & \text{se } y_i=x_i \\
l_{j,3} & \text{se } y_i=l_{j,2} \text{ e } x_i=l_{j,1} \\ 
k & \text{c.c.}
\end{cases}
$$

``` python
x = np.array([1, 2, 3])
y = np.array([2, 3, 2])
l = np.array([[2, 3, 2],[1, 2, 1]])
# x=2 e y=3 tem peso 2
# x=1 e y=2 tem peso 1
loss(x, y, 'ind', l, 3) # Os demais tem peso 3
```

Se `p` for igual a `'log'`, a função perda é a logarítmica.

$$\hat{l}=-y_i\log(x_i)$$

``` python
x = np.array([0.3, 0.4, 0.9])
y = np.array([1, 0, 1])
loss(x, y, 'log')
```

## *metricas.SAE*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula a soma dos erros absolutos

$$\text{SAE}(x, y) = \sum_{i=1}^n|y_i-x_i|$$

``` python
SAE(x, y) == loss(x, y, 1)
```

## *metricas.RSS*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula a soma do quadrado dos resíduos

$$\text{RSS}(x, y) = \sum_{i=1}^n(y_i-x_i)^2$$

``` python
RSS(x, y) == loss(x, y, 2)
```

## *metricas.ME*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula o erro máximo

$$\text{ME}(x, y) = max\{|y_i-x_i|\}_{i=1}^n$$

``` python
ME(x, y) == loss(x, y, float('inf'))
```

## *metricas.empirical_risk*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados;

`p` : Parâmetro que define a função loss;

`l` : Matriz de pesos para erros de classificação via função indicadora;

`subs` : Valor de peso padrão para casos não listados na matriz l.

Retorna um `float`

### Definição
__________________________

Calcula o risco empírico $R$ utilizando alguma função perda implementada em `metricas.loss`.

$$R(x, y) = \frac{1}{n}\sum_{i=1}^n\hat{l}(y_i,x_i)$$

``` python
empirical_risk(x, y, p, l, subs) == loss(x, y, p, l, subs)/n
```

## *metricas.MAE*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Erro absoluto médio. Calcula o risco usando a soma dos erros absolutos

$$\text{MAE}(x, y) = \frac{1}{n}\sum_{i=1}^n|y_i-x_i| = \frac{\text{SAE}}{n}$$

``` python
MAE(x, y) == empirical_risk(x, y, 1)
```

## *metricas.MSE*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Erro quadrático médio. Calcula o risco usando a soma do quadrado dos resíduos

$$\text{MSE}(x,y) = \frac{1}{n}\sum_{i=1}^n(y_i-x_i)^2 = \frac{\text{RSS}}{n}$$

``` python
MSE(x, y) == empirical_risk(x, y, 2)
```

## *metricas.RMSE*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula o a raiz do erro quadrático médio (MSE)

$$\text{RMSE}(x, y) = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-x_i)^2} = \sqrt{\text{MSE}}$$

``` python
RMSE(x, y) == empirical_risk(x, y, 2)**(1/2)
```

## *metricas.TSS*

### Parâmetros
__________________________

`x` : Vetor numérico.

Retorna um `float`

### Definição
__________________________

Calcula a soma dos quadrados total.

$$\text{TSS}(x) = \sum_{i=1}^n(x_i-\bar{x})^2 \qquad \bar{x}=\frac{1}{n}\sum_{i=1}^nx_i$$

``` python
x = np.array([1, 3, 2, 2]) # x.mean() == 2
TSS(x)
```

## *metricas.R2*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo.;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula o erro $R^2$.

$$\text{R}^2(x, y) = 1 - \frac{\text{RSS}(x, y)}{\text{TSS}(y)}$$

``` python
R2(x, y) == 1 - RSS(x, y)/TSS(y)
```

## *metricas.dif_empirical_risk*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados;

`p` : Parâmetro que define a função loss;

`tol` : Parâmetro de tolerância.

Retorna um `np.nparray`

### Definição
__________________________

Calcula a derivada do risco empírico $R$ utilizando alguma função perda implementada em `metricas.loss`.

$$R(x, y) = \frac{1}{n}\sum_{i=1}^n\hat{l}(y_i,x_i)$$

Se `p` for um valor numérico, temos a função perda baseada em uma norma $L_p$.

$$\frac{\partial \text{R}(x,y,p)}{\partial x} = \frac{-p\cdot sgn(y-x)|y-x|^{p-1}}{n}$$

Quando `p` for `log`, temos a perda logarítmica.

$$\frac{\partial \text{R}(x,y,\text{log})}{\partial x} = \frac{-y}{nx}$$

``` python
x = np.array([1, 4, 3, 2])
y = np.array([3, 1, 4, 1])
dif_empirical_risk(x, y, 4)
dif_empirical_risk(x, y, 'log', 1e-12)
```

## *metricas.DMAE*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula a derivada do erro absoluto médio.

$$\text{DMAE}(x,y) = \frac{-sign(y-x)}{n}$$

``` python
DMAE(x, y) == dif_empirical_risk(x, y, 1)
```

## *metricas.DMSE*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Calcula a derivada do erro quadrático médio.

$$\text{DMSE}(x,y) = \frac{-2sign(y-x)|y-x|}{n}$$

``` python
DMSE(x, y) == dif_empirical_risk(x, y, 2)
```

## *metricas.norm_lp*

### Parâmetros
__________________________

`x` : Vetor numérico;

`p` : Parâmetro que define a norma.

Retorna um `float`

### Definição
__________________________

Calcula a norma $L_p$ de um vetor, com $p\geq 1$.

$$\|x\|_p = \left(\sum_{i=1}^n|x_i|^p\right)^\frac{1}{p}$$

Quando `p` é infinito a norma retorna o maior valor absoluto.

$$\|x\|_\infty = max\{|x_i|\}_{i=1}^n$$

``` python
x = np.array([2, 4, 1, 5])
norm_lp(x, 1)
norm_lp(x, float('inf'))
```

## *metricas.dist*

__________________________

`x` : Vetor numérico;

`y` : Vetor numérico

`p` : Parâmetro que define a norma.

Retorna um `float`

### Definição
__________________________

Calcula a distância entre dois vetores usando a norma $L_p$.

$$\text{dist}(x, y, p) = \left(\sum_{i=1}^n|x_i-y_i|^p\right)^\frac{1}{p}$$

``` python
dist(x, y, p) == norm_lp(y-x, p)
```

## *metricas.erros*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `tuple`

### Definição
__________________________

Essa função calcula o número de verdadeiros positivos, verdadeiros negativos, e falsos positivos (Tipo I) e falsos negativos (Tipo II).

$$\text{VP}: y_i = 1 \text{ e } x_i = 1\\
\text{VN}: y_i = 0 \text{ e } x_i = 0\\
\text{FP}: y_i = 0 \text{ e } x_i = 1\\
\text{FN}: y_i = 1 \text{ e } x_i = 0$$

``` python
x = np.array([1, 1, 0, 1, 0])
y = np.array([1, 0, 1, 1, 0])
vp, vn, fp, fn = erros(x, y) # 2, 1, 1, 1
```

## *metricas.acuracia*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Essa função calcula a acurácia da sua previsão.

$$\text{Acurácia}(x,y) = \frac{\text{VP}+\text{VN}}{\text{VP}+\text{VN}+\text{FP}+\text{FN}}$$

``` python
x = np.array([1, 1, 0, 1, 0])
y = np.array([1, 0, 1, 1, 0])
print(acuracia(x, y)) # 3/5
```

## *metricas.precisao*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Essa função calcula a previsão da sua previsão.

$$\text{Precisão}(x,y) = \frac{\text{VP}}{\text{VP}+\text{FP}}$$

``` python
x = np.array([1, 1, 0, 1, 0])
y = np.array([1, 0, 1, 1, 0])
print(precisao(x, y)) # 2/3
```

## *metricas.recall*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Essa função calcula o recall da sua previsão.

$$\text{Recall}(x,y) = \frac{\text{VP}}{\text{VP}+\text{FN}}$$

``` python
x = np.array([1, 1, 0, 1, 0])
y = np.array([1, 0, 1, 1, 0])
print(recall(x, y)) # 2/3
```

## *metricas.F1*

### Parâmetros
__________________________

`x` : Vetor que contém os valores previstos pelo seu modelo;

`y` : Vetor que contém os valores observados nos seus dados.

Retorna um `float`

### Definição
__________________________

Essa função calcula o $F_1$-score da sua previsão.

$$\text{F}_1(x,y) = 2\cdot\frac{\text{Precisão}(x,y)\cdot\text{Recall}(x,y)}{\text{Precisão}(x,y) + \text{Recall}(x,y)}$$

``` python
x = np.array([1, 1, 0, 1, 0])
y = np.array([1, 0, 1, 1, 0])
print(F1(x, y)) # 2/3
```
