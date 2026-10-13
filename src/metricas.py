# Módulo contendo as funções usadas para calcular as métricas
# Responsável: Pedro Paulo Gomes Paiva

import numpy as np

# Aplica o somatório em uma função perda
def perda(x: np.ndarray, y:np.ndarray, p: float | int | str) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    
    # Perdas para problemas de regressão baseadas nas normas L_p
    if isinstance(p, (int, float)):
        if not np.issubdtype(x.dtype, np.number) and not np.issubdtype(y.dtype, np.number):
            raise TypeError('Os vetores devem conter dados numéricos para a regressão')
        elif p < 1:
            raise ValueError('O valor do parâmetro deve ser maior ou igual a 1')
        elif p < float('inf'):
            return float(np.sum(np.abs(x-y)**p))
        else:
            return float(np.abs(x-y).max())
    elif p == 'ind':
        soma = 0
        for indx in range(x.size):
            if x[indx] != y[indx]:
                soma += 1
        return soma
    else:
        raise TypeError('Método não implementado')

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
def risco_empirico(x: np.ndarray, y:np.ndarray, p: float | int | str) -> float:
    if isinstance(p, (int, float)) or p == 'ind':
        n = x.size
        return perda(x, y, p) / n
    else:
        raise TypeError('Método não implementado')

# (Mean Absolute Error - Erro Absoluto Médio)
def mae(x:np.ndarray, y:np.ndarray) -> float:
    return risco_empirico(x, y, 1)

# (Mean Squared Error - Erro Quadrático Médio)
def mse(x:np.ndarray, y:np.ndarray) -> float:
    return risco_empirico(x, y, 2)

# (Root Mean Squared Error - Raiz do Erro Quadrático Médio)
def rmse(x:np.ndarray, y:np.ndarray) -> float:
    return mse(x, y)**(1/2)

# (Total Sum of Squares) - Soma dos Quadrados Total
def tss(x:np.ndarray) -> float:
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError('O vetor deve conter dados numéricos para a regressão')
    return float(np.sum((x-x.mean())**2))

# (R^2) - R-Quadrado
def R2(x:np.ndarray, y:np.ndarray) -> float:
    rss_ = rss(x, y)
    tss_ = tss(y)
    return 1 - (rss_/tss_)


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

    indx += 1
    # Deve retornar (142/6)**(1/2) ≃ 4.864839839
    if abs(rmse(a, b) - 4.864839839) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 212.833333333
    if abs(tss(a) - 212.833333333) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar -1.062953995
    if abs(R2(a, b) - (-1.062953995)) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    c = np.array(['sim', 'sim', 'não', 'sim', 'não'])
    d = np.array(['sim', 'não', 'sim', 'sim', 'não'])

    indx += 1
    # Deve retornar 2
    if perda(c, d, 'ind') == 2:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 2/5 = 0.4
    if abs(risco_empirico(c, d, 'ind') - 0.4) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')
