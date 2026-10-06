# Módulo contendo as funções usadas para calcular as métricas
# Responsável: Pedro Paulo Gomes Paiva

import numpy as np

# Aplica o somatório em uma função perda
def perda(x, y, p):
    # Perdas para problemas de regressão baseadas nas normas L_p
    if isinstance(p, (int, float)):
        if p < 1:
            ValueError('O valor do parâmetro deve ser maior ou igual a 1')
        elif p < float('inf'):
            return float(np.sum(np.abs(x-y)**p))
        else:
            return float(np.abs(x).max())

# (Sum of Absolute Errors - Soma dos Erros Absolutos - Perda L1)
def sae(x, y):
    return perda(x, y, 1)

# (Residual Sum of Squares - Soma do Quadrado dos Resíduos - Perda L2)
def rss(x, y):
    return perda(x, y, 2)

# (Maximun Error - Erro Máximo - Perda L_inf)
def me(x, y):
    return perda(x, y, float('inf'))

# Risco Empírico - Média da função perda
def risco_empirico(x, y, p):
    n = x.size
    if isinstance(p, (int, float)):
        return perda(x, y, p) / n

# (Mean Absolute Error - Erro Absoluto Médio)
def mae(x, y):
    return risco_empirico(x, y, 1)

# (Mean Squared Error - Erro Quadrático Médio)
def mse(x, y):
    return risco_empirico(x, y, 2)
