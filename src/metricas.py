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
    return perda(x, y, 1)

# (Residual Sum of Squares - Soma do Quadrado dos Resíduos - Perda L2)
def rss(x:np.ndarray, y:np.ndarray) -> float:
    return perda(x, y, 2)

# (Maximun Error - Erro Máximo - Perda L_inf)
def me(x:np.ndarray, y:np.ndarray) -> float:
    return perda(x, y, float('inf'))

# Risco Empírico - Média da função perda
def risco_empirico(x: np.ndarray, y:np.ndarray, p: float | int) -> float:
    if isinstance(p, (int, float)):
        n = x.size
        return perda(x, y, p) / n
    else:
        raise TypeError('Ainda não implementado')

# (Mean Absolute Error - Erro Absoluto Médio)
def mae(x:np.ndarray, y:np.ndarray) -> float:
    return risco_empirico(x, y, 1)

# (Mean Squared Error - Erro Quadrático Médio)
def mse(x:np.ndarray, y:np.ndarray) -> float:
    return risco_empirico(x, y, 2)

# Testes para as funções criadas até o momento

if __name__ == '__main__':

    # Vetores para teste
    a = np.array([3, 5, 21, 7, 5, 8])
    b = np.array([4, 6, 12, 4, 4, 1])

    tol = 1e-9 # Tolerância do erro

    indx = 1
    # Deve retornar 22
    if abs(perda(a, b, 1) - 22) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 142
    if abs(perda(a, b, 2) - 142) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 76_102
    if abs(perda(a, b, 5) - 76_102) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 9
    if abs(perda(a, b, np.inf) - 9) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 22
    if abs(sae(a, b) - 22) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 142
    if abs(rss(a, b) - 142) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 9
    if abs(me(a, b) - 9) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 22/6 ≃ 3.666666666
    if abs(risco_empirico(a, b, 1) - 3.666666666) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 142/6 ≃ 23.666666666
    if abs(risco_empirico(a, b, 2) - 23.666666666) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 76_102/6 ≃ 12_683.666666666
    if abs(risco_empirico(a, b, 5) - 12_683.666666666) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 9/6 = 1.5
    if abs(risco_empirico(a, b, np.inf) - 1.5) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 22/6 ≃ 3.666666666
    if abs(mae(a, b) - 3.666666666) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 142/6 ≃ 23.666666666
    if abs(mse(a, b) - 23.666666666) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')
