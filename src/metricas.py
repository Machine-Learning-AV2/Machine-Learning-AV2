# Módulo contendo as funções usadas para calcular as métricas
# Responsável: Pedro Paulo Gomes Paiva

import numpy as np

# Aplica o somatório em uma função perda
def perda(x: np.ndarray, y:np.ndarray, p: float | int) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    
    # Perdas para problemas de regressão baseadas nas normas L_p
    if isinstance(p, (int, float)):
        if p < 1:
            raise ValueError('O valor do parâmetro deve ser maior ou igual a 1')
        elif p < float('inf'):
            return float(np.sum(np.abs(x-y)**p))
        else:
            return float(np.abs(x-y).max())
    else:
        raise TypeError('Ainda não implementado')

# (Sum of Absolute Errors - Soma dos Erros Absolutos - Perda L1)
def sae(x:np.ndarray, y:np.ndarray) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    
    return perda(x, y, 1)

# (Residual Sum of Squares - Soma do Quadrado dos Resíduos - Perda L2)
def rss(x:np.ndarray, y:np.ndarray) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    
    return perda(x, y, 2)

# (Maximun Error - Erro Máximo - Perda L_inf)
def me(x:np.ndarray, y:np.ndarray) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    
    return perda(x, y, float('inf'))

# Risco Empírico - Média da função perda
def risco_empirico(x: np.ndarray, y:np.ndarray, p: float | int) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    
    n = x.size
    if isinstance(p, (int, float)):
        return perda(x, y, p) / n
    else:
        raise TypeError('Ainda não implementado')

# (Mean Absolute Error - Erro Absoluto Médio)
def mae(x:np.ndarray, y:np.ndarray) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')

    return risco_empirico(x, y, 1)

# (Mean Squared Error - Erro Quadrático Médio)
def mse(x:np.ndarray, y:np.ndarray) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')

    return risco_empirico(x, y, 2)
    if x.size != y.size:
        ValueError('Os dois vetores devem ter o mesmo tamanho')

    return risco_empirico(x, y, 2)
