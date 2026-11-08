# Módulo contendo as funções usadas para calcular as métricas
# Responsável: Pedro Paulo Gomes Paiva

import numpy as np


# Aplica o somatório em uma função loss
def loss(x: np.ndarray, y:np.ndarray, p:float | str, l:np.ndarray = np.array([]), subs:float = 1) -> float:

    """Retorna o valor da soma de uma função loss para um vetor de valores previstos e observados
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados;
    p : Parâmetro que define a função loss;
    l : Matriz de pesos para erros de classificação via função indicadora;
    subs : Valor de peso padrão para casos não listados na matriz l.

    Funções loss implementadas
    ----------------------------
    p (numérico) = |y - x|^p;  
    p ('ind') = k se (y != x);
    p ('log') -sum(y*log(x)).

    Matriz de pesos
    ----------------
    l deve ser uma matriz com 3 colunas, de modo que para cada linha:
    O primeiro elemento deve ser o valor para x;
    O segundo elemento deve ser o valor para y;
    O terceiro elemento deve ser o valor k retornado pela função nesse caso.

    Exemplos
    ---------
    >>> x = np.array([1, 2, 4])
    >>> y = np.array([1, 3, 5])
    >>> loss(x, y, 2) # Retorna (|1-1|^2 + |3-2|^2 + |5-4|^2) = 2
    >>> l = np.array([2, 3, 2]) # O peso em classificar x = 2 enquanto y = 3 é 2
    >>> loss(x, y, 'ind', l, 3) # Classificar errado nos demais casos tem peso 3"""
    
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.shape != y.shape:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    if l.size > 0:
        for i in l:
            size = l.shape
            if size[1] != 3:
                raise TypeError('As restrições devem conter o valor previsto, o observado e o peso da penalidade (apenas)')
    if not isinstance(subs, (int, float)):
        raise TypeError('O valor de substituição padrão deve ser numérico')
    
    # losss para problemas de regressão baseadas nas normas L_p
    if isinstance(p, (int, float)):
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise TypeError('Os vetores devem conter dados numéricos para a regressão')
        elif p < 1:
            raise ValueError('O valor do parâmetro deve ser maior ou igual a 1')
        elif p < float('inf'):
            return float(np.sum(np.abs(y-x)**p))
        else:
            return float(np.abs(y-x).max())

    # loss com a função indicadora
    elif p == 'ind':
        soma = 0
        l_x = [i[0] for i in l]
        for i in range(x.size):
            if x[i] != y[i]:
                if x[i] in l_x:
                    indxs = [indx for indx, x_ in enumerate(l_x) if x_ == x[i]]
                    for j in indxs:
                        if l[j][1] == y[i]:
                            soma += float(l[j][2])
                            break
                else:
                    soma += subs
        return float(soma)
    
    # loss com função logaritmica
    elif p == 'log':
        return -float(np.sum(y*np.log(x)))
    else:
        raise TypeError('Método não implementado')


# (Sum of Absolute Errors - Soma dos Erros Absolutos - loss L_1)
def SAE(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a soma de |y_i-x_i| para todo y_i em y e x_i em x

    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return loss(x, y, 1)


# (Residual Sum of Squares - Soma do Quadrado dos Resíduos - loss L_2)
def RSS(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a soma de |y_i-x_i|^2 para todo y_i em y e x_i em x
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return loss(x, y, 2)


# (Maximun Error - Erro Máximo - loss L_inf)
def ME(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna o max{|y_i-x_i|} para todo y_i em y e x_i em x

    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return loss(x, y, float('inf'))


# Risco Empírico - Média da função loss
def empirical_risk(x: np.ndarray, y:np.ndarray, p:float | str, l:np.ndarray = np.array([]), subs:float = 1) -> float:

    """Retorna o valor do risco empírico de uma previsão segundo a função loss desejada
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados;
    p : Parâmetro que define a função loss;
    l : Matriz de pesos para erros de classificação via função indicadora;
    subs : Valor de peso padrão para casos não listados na matriz l.

    Funções loss implementadas
    ----------------------------
    p (numérico) = |y - x|^p;  
    p ('ind') = k se (y != x).

    Matriz de pesos
    ----------------
    l deve ser uma matriz com 3 colunas, de modo que para cada linha:
    O primeiro elemento deve ser o valor para x;
    O segundo elemento deve ser o valor para y;
    O terceiro elemento deve ser o valor k retornado pela função nesse caso.

    Exemplos
    ---------
    >>> x = np.array([1, 2, 4])
    >>> y = np.array([1, 3, 5])
    >>> empirical_risk(x, y, 2) # Retorna (|1-1|^2 + |3-2|^2 + |5-4|^2)/3 ≃ 0.666...
    >>> empirical_risk(x, y, 'ind', subs = 3) # Retorna (0 + 3 + 3)/3 = 2"""

    if isinstance(p, (int, float)):
        if p < float('inf'):
            n = x.size
            return loss(x, y, p, l, subs) / n
        else:
            raise ValueError('O parâmetro deve ser um valor finito')
    elif p == 'ind':
        n = x.size
        return loss(x, y, p, l, subs) / n
    elif p == 'log':
        n, *m = x.shape
        return loss(x, y, p) / n
    else:
        raise TypeError('Método não implementado')

def classification_error(x: np.ndarray, y: np.ndarray) -> float:
    return empirical_risk(x, y, 'ind')

# (Mean Absolute Error - Erro Absoluto Médio)
def MAE(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a soma de |y_i-x_i|/n para todo y_i em y e x_i em x

    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return empirical_risk(x, y, 1)


# (Mean Squared Error - Erro Quadrático Médio)
def MSE(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a soma de (|y_i-x_i|^2)/n para todo y_i em y e x_i em x
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return empirical_risk(x, y, 2)


# (Root Mean Squared Error - Raiz do Erro Quadrático Médio)
def RMSE(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a raiz da soma de |y_i-x_i|^2 para todo y_i em y e x_i em x
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return MSE(x, y)**(1/2)


# (Total Sum of Squares) - Soma dos Quadrados Total
def TSS(x:np.ndarray) -> float:

    """Retorna a soma de |x_i - (média de x)|^2 para todo x_i em x
    
    Parâmetros
    -----------
    x : Vetor que contém valores numéricos."""

    if not np.issubdtype(x.dtype, np.number):
        raise TypeError('O vetor deve conter dados numéricos')
    return float(np.sum((x-x.mean())**2))


# (R^2) - R-Quadrado
def R2(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna o valor R^2 para a previsão do modelo
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    RSS_ = RSS(x, y)
    tss_ = TSS(y)
    return 1 - (RSS_/tss_)


# Norma Lp de um vetor
def norm_lp(x:np.ndarray, p:float) -> float:

    """Retorna a norma L_p de um vetor
    
    Parâmetros
    -----------
    x : Vetor que contém valores numéricos;
    p : Parâmetro que define a norma."""

    if not isinstance(x, np.ndarray):
        raise TypeError('O vetor deve ser um array numpy')
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError('O vetor deve conter dados numéricos para a norma')
    if isinstance(p, (float, int)):
        if p < 1:
            raise ValueError('O valor do parâmetro deve ser maior ou igual a 1')
        if p < float('inf'):
            return float(np.sum(np.abs(x)**p)**(1/p))
        else:
            return float(np.abs(x).max())
    else:
        raise TypeError('O parâmetro deve ser um dado numérico (>=1)')


# Distância entre dois vetores baseada na norma Lp
def dist(x:np.ndarray, y:np.ndarray, p:float) -> float:

    """Retorna a distância entre dois vetores usando a norma L_p
    
    Parâmetros
    -----------
    x : Vetor que contém valores numéricos;
    y : Vetor que contém valores numéricos;
    p : Parâmetro que define a norma."""

    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise TypeError('Os vetores devem conter dados numéricos para a norma')
    return norm_lp(y-x, p)


# Derivada do risco empírico para calcular a descida de gradiente
def dif_empirical_risk(x: np.ndarray, y:np.ndarray, p:float, tol:float = 1e-9) -> np.ndarray:

    """Retorna o valor da deerivada () do risco empírico de uma previsão segundo a função loss desejada
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados;
    p : Parâmetro que define a função loss;
    tol : Tolerância para o float.

    Funções loss implementadas
    ----------------------------
    p (numérico) = |y - x|^p;
    p (log) = -sum(y*log(x)).

    Exemplos
    ---------
    >>> x = np.array([1, 2, 4])
    >>> y = np.array([1, 3, 5])
    >>> dif_empirical_risk(x, y, 2) # Retorna np.array([0, -2/3, -2/3])"""

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.shape != y.shape:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')

    if isinstance(p, (int, float)):
        if p < float('inf'):
            if 1 <= p:
                n, *m = x.shape

                # Caso de descontinuidade
                if p==1:
                    return np.where(np.abs(y - x) < tol, 0.0, (-np.sign(y - x)) / n)

                # Derivada df/dx = -p*sign(y-x)*|y-x|**(p-1)/n
                return (-p * abs(y-x)**(p-1) * np.sign(y-x)) / n
            else:
                raise ValueError('O parâmetro deve ser maior ou igual a 1')
        else:
            raise ValueError('O parâmetro deve ser um valor finito')
    elif p == 'log':
        n, *m = x.shape
        return -y/(x*n)
    else:
        raise TypeError('Método não implementado')


# Derivada do MAE
def DMAE(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a derivada da soma de |y_i-x_i|/n para todo y_i em y e x_i em x

    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return dif_empirical_risk(x, y, 1)


# Derivada do MSE
def DMSE(x:np.ndarray, y:np.ndarray) -> float:

    """Retorna a derivada da soma de (|y_i-x_i|^2)/n para todo y_i em y e x_i em x
    
    Parâmetros
    -----------
    x : Vetor que contém os valores previstos pelo seu modelo;
    y : Vetor que contém os valores observados nos seus dados."""

    return dif_empirical_risk(x, y, 2)
