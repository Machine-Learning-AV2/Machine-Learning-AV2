import numpy as np 


def validar_matriz(X): #função auxiliar para verificar se os dados da amostra é válido
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0: #número de linha e Núemrod de colunas, dimensão dois porque temos as observações e também temos os atributos de cada observação
        raise TypeError("X deve ser 2D e não vazio: (n_amostras, n_atributos)")
        
    if not np.all(np.isfinite(X)): #np.all verifica todos os valores são verdadeiros, np.infinite retorna veradeiro ou falso para cada valor, vendp se não é Nan ou infinito
        raise ValueError("X contém NaN ou infinito")
    return X


def inteiro_positivo(valor, nome): #função auxiliar para verificar se o valor é válido
    if not isinstance(valor, (int, np.integer)) or isinstance(valor, bool): #verifica se o valor é um inteiro
        raise ValueError(f"{nome} deve ser um inteiro positivo.")

    if valor < 1:
        raise ValueError(f"{nome} deve ser >= 1.") # verifica se o valor é maior ou igual 1

    return int(valor)


class Kmeans:
    def __init__(self, k, max_iteracao=100):
        self.k = inteiro_positivo(k, "k")
        self.max_iteracao = inteiro_positivo(max_iteracao, "max_iteracao") #verificando se o valor é válido
        self.centroides = None # vetor que vai armazemar os centroides (vetor)

    def estimar(self, X):
        if self.centroides is None:
            raise RuntimeError("chame fit antes de estimar")

        X = validar_matriz(X)

        if X.shape[1] != self.centroides.shape[1]: # shape seria: X.shape = (qnt_amostra, qnt_atributos), shape[1] é o número de colunas, ou seja, o número de atributos
            raise ValueError(f"X tem {X.shape[1]} atributos, mas o modelo foi treinado com {self.centroides.shape[1]} atributos")

        distancias = np.sum((X[:, None, :] - self.centroides[None, :, :]) ** 2, axis=2) # (amostra, centroide, coordenada)
        # Com sum(axis=2), somamos as coordenadas: (amostra, centroide), contendo as distâncias euclidianas ao quadrado  
    
        return np.argmin(distancias, axis=1) # Para cada amostra, retorna o índice do centroide mais próximo. Formato: (amostra) contendo o índice do centróide mais próximo.
               
    def fit(self, X):
        X = validar_matriz(X)

        if self.k > len(X):       #verifica se o número de clusters é maior que o número de amostras
            raise ValueError(f"k={self.k} é maior que o número de amostras ({len(X)})")

        distintos = np.unique(X, axis=0) #verifica se o número de clusters é maior que o núemor de amostras distintos
        if self.k > len(distintos):
            raise ValueError(f"há apenas {len(distintos)} pontos distintos em X, menos que k={self.k}")

        rng = np.random.default_rng()                   # gerador de números aleatórios
        indices = rng.choice(len(distintos), self.k, replace=False) # seleciona k índices aleatórios (índices distintos)
        self.centroides = distintos[indices].copy()     #copiando os centroides escolhidos

        for _ in range(self.max_iteracao):
            conjunto = self.estimar(X)          # atribuindo cada amostra ao centroide mais próximo
            novos = self.centroides.copy()      # inicializando os valores dos novos centroides da iteração

            for i in range(self.k):             # para cada centroide calcular seu novo valor como a média dos pontos do seu conjunto
                pontos = X[conjunto == i]       # selecionar todos os X[j] cuja conjunto[j] = i
                if len(pontos) > 0:
                    novos[i] = pontos.mean(axis=0)

            if np.array_equal(self.centroides, novos): # se o valor dos novos centroides não alterou, então convergiu
                break

            self.centroides = novos             #atualizando o valor dos centroides
        return self


class Pca:
    def __init__(self):
        self.media = None                   # média de cada atributo original
        self.componentes = None             # autovetores ordenados
        self.autovalores = None             # variâncias associados aos autovetores (as direções)

    def fit(self, X):
        X = validar_matriz(X)
        if len(X) < 2:                      # se a dimensão for <=1 não temos dimensão para reduzir, calcular covariância usa n - 1.
            raise ValueError("PCA precisa de pelo menos 2 amostras")

        self.media = X.mean(axis=0)                 #média dos vetores colunas (amostra)
        X_centralizado = X - self.media   #centralizando os dados

        cov = X_centralizado.T @ X_centralizado / (len(X) - 1)  #calculando a matriz de covariância amostral
        valores, vetores = np.linalg.eigh(cov)                                      #calculando os auto valores e os autovetores (padronizados)

        indices = np.argsort(valores)[::-1]                     #pegando os índices dos autovalores em ordem decrescente
        self.autovalores = np.maximum(valores[indices], 0.0)    #substituindo os autovalroes negativos por 0, que podem surgir a instabilidade numérica do ponto flutuante (variância negativo não exite)
        self.componentes = vetores[:, indices]                  #reordenando as colunas (veotres colunas)

        return self

    def transformar(self, X, n_componentes=None):
        if self.media is None:
            raise RuntimeError("chame fit antes de transformar")

        X = validar_matriz(X)
        if X.shape[1] != len(self.media):       # compatibilidade entre a dimensão dos dados e a média
            raise ValueError(
                f"X tem {X.shape[1]} atributos, esperado {len(self.media)}"
            )

        n = self.componentes.shape[1] if n_componentes is None else inteiro_positivo(
            n_componentes, "n_componentes"      #usar todos os componentes se None, se não verificar se o valor dado é válido
        )
        if n > self.componentes.shape[1]:       #se o número o n for maior que o número de atributos
            raise ValueError(f"n_componentes={n} excede o número de atributos : {self.componentes.shape[1]}")

        return (X - self.media) @ self.componentes[:, :n]

    def fit_transform(self, X, n_componentes=None):
        return self.fit(X).transformar(X, n_componentes)

# ====================================TESTES=======================================================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # KMEANS
    # dois grupos bem separados: um perto de (0,0) e outro perto de (10,10)
    X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(10, 0.5, (50, 2))])

    modelo = Kmeans(k=2).fit(X)
    rotulos = modelo.estimar(X)

    assert len(set(rotulos[:50])) == 1   # primeiro grupo todo no mesmo cluster
    assert len(set(rotulos[50:])) == 1   # segundo grupo todo no mesmo cluster
    assert rotulos[0] != rotulos[50]     # clusters diferentes
    print("Kmeans OK")

    # PCA
    # dados quase sobre a reta y = 2x
    t = rng.normal(size=200) 
    X = np.column_stack([t, 2 * t]) + 0.01 * rng.normal(size=(200, 2))

    pca = Pca()
    Z = pca.fit_transform(X, n_componentes=1)

    esperado = np.array([1, 2]) / np.sqrt(5)                           # direção da reta
    assert np.isclose(abs(pca.componentes[:, 0] @ esperado), 1, atol=1e-3)  # abs: sinal é arbitrário
    assert Z.shape == (200, 1)
    print("PCA OK")