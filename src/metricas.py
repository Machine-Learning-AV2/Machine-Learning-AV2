# Módulo contendo as funções usadas para calcular as métricas
# Responsável: Pedro Paulo Gomes Paiva

import numpy as np

# Aplica o somatório em uma função perda
def perda(x: np.ndarray, y:np.ndarray, p:float | str, l:list = [], subs:float = 1) -> float:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Os vetores devem ser arrays numpy')
    if x.size != y.size:
        raise ValueError('Os dois vetores devem ter o mesmo tamanho')
    if l != []:
        for i in l:
            if len(i) != 3:
                raise TypeError('As restrições devem conter o valor previsto, o observado e o peso da penalidade (apenas)')
    if not isinstance(subs, (int, float)):
        raise TypeError('O valor de substituição padrão deve ser numérico')
    
    # Perdas para problemas de regressão baseadas nas normas L_p
    if isinstance(p, (int, float)):
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise TypeError('Os vetores devem conter dados numéricos para a regressão')
        elif p < 1:
            raise ValueError('O valor do parâmetro deve ser maior ou igual a 1')
        elif p < float('inf'):
            return float(np.sum(np.abs(x-y)**p))
        else:
            return float(np.abs(x-y).max())

    # Perda com a função indicadora
    elif p == 'ind':
        soma = 0
        l_x = [i[0] for i in l]
        for i in range(x.size):
            if x[i] != y[i]:
                if x[i] in l_x:
                    indxs = [indx for indx, x_ in enumerate(l_x) if x_ == x[i]]
                    for j in indxs:
                        if l[j][1] == y[i]:
                            soma += l[j][2]
                            break
                else:
                    soma += subs
        return float(soma)
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
def risco_empirico(x: np.ndarray, y:np.ndarray, p:float | str, l:list = [], subs:float = 1) -> float:
    if isinstance(p, (int, float)):
        if p < float('inf'):
            n = x.size
            return perda(x, y, p, l, subs) / n
        else:
            raise ValueError('O parâmetro deve ser um valor finito')
    elif p == 'ind':
        n = x.size
        return perda(x, y, p, l, subs) / n
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
    if abs(perda(c, d, 'ind') - 2) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 2/5 = 0.4
    if abs(risco_empirico(c, d, 'ind') - 0.4) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    l_1 = [['sim', 'não', 2], ['não', 'sim', 1]]

    indx += 1
    # Deve retornar 3
    if abs(perda(c, d, 'ind', l_1) - 3) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')

    indx += 1
    # Deve retornar 3/5 = 0.6
    if abs(risco_empirico(c, d, 'ind', l_1) - 0.6) < tol:
        print(f'{indx} Validada')
    else:
        print(f'{indx} Não Validada')
