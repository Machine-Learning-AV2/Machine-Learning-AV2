import numpy as np 


def validade_matrix(X): #função auxiliar para verificar se os dados da amostra é válido
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0: #número de linha e Núemrod de colunas, dimensão dois porque temos as observações e também temos os atributos de cada observação
        raise TypeError("X deve ser 2D e não vazio: (n_amostras, n_atributos)")
        
    if not np.all(np.isfinite(X)): #np.all verifica todos os valores são verdadeiros, np.infinite retorna veradeiro ou falso para cada valor, vendp se não é Nan ou infinito
        raise ValueError("X contém NaN ou infinito")
    return X


def validate_positive_integer(valor, nome): #função auxiliar para verificar se o valor é válido
    if not isinstance(valor, (int, np.integer)) or isinstance(valor, bool): #verifica se o valor é um inteiro
        raise ValueError(f"{nome} deve ser um inteiro positivo.")

    if valor < 1:
        raise ValueError(f"{nome} deve ser >= 1.") # verifica se o valor é maior ou igual 1

    return int(valor)


class KMeans:
    """
    `Kmeans(k, max_iteracao=100)`

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
    """
    def __init__(self, k, max_iterations=100):
        self.k = validate_positive_integer(k, "k")
        self.max_iterations = validate_positive_integer(max_iterations, "max_iterations") #verificando se o valor é válido
        self.centroids = None # vetor que vai armazemar os centroids (vetor)

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("chame fit antes de estimar")

        X = validate_matrix(X)

        if X.shape[1] != self.centroids.shape[1]: # shape seria: X.shape = (qnt_amostra, qnt_atributos), shape[1] é o número de colunas, ou seja, o número de atributos
            raise ValueError(f"X tem {X.shape[1]} atributos, mas o modelo foi treinado com {self.centroids.shape[1]} atributos")

        distancias = np.sum((X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2) # (amostra, centroide, coordenada)
        # Com sum(axis=2), somamos as coordenadas: (amostra, centroide), contendo as distâncias euclidianas ao quadrado  
    
        return np.argmin(distancias, axis=1) # Para cada amostra, retorna o índice do centroide mais próximo. Formato: (amostra) contendo o índice do centróide mais próximo.
               
    def fit(self, X):
        X = validate_matrix(X)

        if self.k > len(X):       #verifica se o número de clusters é maior que o número de amostras
            raise ValueError(f"k={self.k} é maior que o número de amostras ({len(X)})")

        distintos = np.unique(X, axis=0) #verifica se o número de clusters é maior que o núemor de amostras distintos
        if self.k > len(distintos):
            raise ValueError(f"há apenas {len(distintos)} pontos distintos em X, menos que k={self.k}")

        rng = np.random.default_rng()                   # gerador de números aleatórios
        indices = rng.choice(len(distintos), self.k, replace=False) # seleciona k índices aleatórios (índices distintos)
        self.centroids = distintos[indices].copy()     #copiando os centroids escolhidos

        for _ in range(self.max_iterations):
            conjunto = self.predict(X)          # atribuindo cada amostra ao centroide mais próximo
            novos = self.centroids.copy()      # inicializando os valores dos novos centroids da iteração

            for i in range(self.k):             # para cada centroide calcular seu novo valor como a média dos pontos do seu conjunto
                pontos = X[conjunto == i]       # selecionar todos os X[j] cuja conjunto[j] = i
                if len(pontos) > 0:
                    novos[i] = pontos.mean(axis=0)

            if np.array_equal(self.centroids, novos): # se o valor dos novos centroids não alterou, então convergiu
                break

            self.centroids = novos             #atualizando o valor dos centroids
        return self


class PCA:
    """
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
    """
    def __init__(self):
        self.mean = None                   # média de cada atributo original
        self.components = None             # autovetores ordenados
        self.eigenvalues = None             # variâncias associados aos autovetores (as direções)

    def fit(self, X):
        X = validate_matrix(X)
        if len(X) < 2:                      # se a dimensão for <=1 não temos dimensão para reduzir, calcular covariância usa n - 1.
            raise ValueError("PCA precisa de pelo menos 2 amostras")

        self.mean = X.mean(axis=0)                 #média dos vetores colunas (amostra)
        X_centralizado = X - self.mean   #centralizando os dados

        cov = X_centralizado.T @ X_centralizado / (len(X) - 1)  #calculando a matriz de covariância amostral
        valores, vetores = np.linalg.eigh(cov)                                      #calculando os auto valores e os autovetores (padronizados)

        indices = np.argsort(valores)[::-1]                     #pegando os índices dos eigenvalues em ordem decrescente
        self.eigenvalues = np.maximum(valores[indices], 0.0)    #substituindo os autovalroes negativos por 0, que podem surgir a instabilidade numérica do ponto flutuante (variância negativo não exite)
        self.components = vetores[:, indices]                  #reordenando as colunas (veotres colunas)

        return self

    def transform(self, X, n_components=None):
        if self.mean is None:
            raise RuntimeError("chame fit antes de transformar")

        X = validate_matrix(X)
        if X.shape[1] != len(self.mean):       # compatibilidade entre a dimensão dos dados e a média
            raise ValueError(
                f"X tem {X.shape[1]} atributos, esperado {len(self.mean)}"
            )

        n = self.components.shape[1] if n_components is None else validate_positive_integer(
            n_components, "n_components"      #usar todos os components se None, se não verificar se o valor dado é válido
        )
        if n > self.components.shape[1]:       #se o número o n for maior que o número de atributos
            raise ValueError(f"n_components={n} excede o número de atributos : {self.components.shape[1]}")

        return (X - self.mean) @ self.components[:, :n]

    def fit_transform(self, X, n_components=None):
        return self.fit(X).transform(X, n_components)

# ====================================TESTES=======================================================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # KMeans
    # dois grupos bem separados: um perto de (0,0) e outro perto de (10,10)
    X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(10, 0.5, (50, 2))])

    modelo = KMeans(k=2).fit(X)
    rotulos = modelo.predict(X)

    assert len(set(rotulos[:50])) == 1   # primeiro grupo todo no mesmo cluster
    assert len(set(rotulos[50:])) == 1   # segundo grupo todo no mesmo cluster
    assert rotulos[0] != rotulos[50]     # clusters diferentes
    print("KMeans OK")

    # PCA
    # dados quase sobre a reta y = 2x
    t = rng.normal(size=200) 
    X = np.column_stack([t, 2 * t]) + 0.01 * rng.normal(size=(200, 2))

    PCA = PCA()
    Z = PCA.fit_transform(X, n_components=1)

    esperado = np.array([1, 2]) / np.sqrt(5)                           # direção da reta
    assert np.isclose(abs(PCA.components[:, 0] @ esperado), 1, atol=1e-3)  # abs: sinal é arbitrário
    assert Z.shape == (200, 1)
    print("PCA OK")