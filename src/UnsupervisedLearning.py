import numpy as np 
from .preprocessing import validate_matrix

def validate_positive_integer(valor: int, nome: str) -> int: #função auxiliar para verificar se o valor é válido
    if not isinstance(valor, (int, np.integer)) or isinstance(valor, bool): #verifica se o valor é um inteiro
        raise ValueError(f"{nome} deve ser um inteiro positivo.")

    if valor < 1:
        raise ValueError(f"{nome} deve ser >= 1.") # verifica se o valor é maior ou igual 1

    return int(valor)


class KMeans:
    """
    Agrupa as amostras em k clusters: sorteia k pontos distintos como centroides, atribui cada amostra ao centroide mais próximo e recalcula cada centroide como a média do seu cluster, até convergir ou atingir max_iteracao.

    Parâmetros de inicialização:
    k: número de clusters (inteiro >= 1).
    max_iteracao: limite de iterações (inteiro >= 1). Por padrão, 100.
    seed: semente do gerador aleatório usado na inicialização (inteiro). Por padrão, 0.

    Atributo (disponível após o fit):
    centroides: centroide de cada cluster, um em cada linha.

    Métodos:
    fit(X): ajusta o modelo e guarda os centroides. Retorna self. Levanta erro se k for maior que o número de amostras ou que o número de pontos distintos.
    estimar(X): retorna o índice do centroide mais próximo de cada amostra.

    Como a inicialização usa a semente, o resultado é reproduzível entre execuções com os mesmos dados.
    """
    def __init__(self, k: int, max_iterations: int = 100, seed: int = 0):
        self.k = validate_positive_integer(k, "k")
        self.max_iterations = validate_positive_integer(max_iterations, "max_iterations") #verificando se o valor é válido
        self.seed = seed # semente do gerador aleatório da inicialização
        self.centroids = None # vetor que vai armazemar os centroids (vetor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("chame fit antes de estimar")

        X = validate_matrix(X)

        if X.shape[1] != self.centroids.shape[1]: # shape seria: X.shape = (qnt_amostra, qnt_atributos), shape[1] é o número de colunas, ou seja, o número de atributos
            raise ValueError(f"X tem {X.shape[1]} atributos, mas o modelo foi treinado com {self.centroids.shape[1]} atributos")

        distancias = np.sum((X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2) # (amostra, centroide, coordenada)
        # Com sum(axis=2), somamos as coordenadas: (amostra, centroide), contendo as distâncias euclidianas ao quadrado  
    
        return np.argmin(distancias, axis=1) # Para cada amostra, retorna o índice do centroide mais próximo. Formato: (amostra) contendo o índice do centróide mais próximo.
               
    def fit(self, X: np.ndarray):
        X = validate_matrix(X)

        if self.k > len(X):       #verifica se o número de clusters é maior que o número de amostras
            raise ValueError(f"k={self.k} é maior que o número de amostras ({len(X)})")

        distintos = np.unique(X, axis=0) #verifica se o número de clusters é maior que o núemor de amostras distintos
        if self.k > len(distintos):
            raise ValueError(f"há apenas {len(distintos)} pontos distintos em X, menos que k={self.k}")

        rng = np.random.default_rng(self.seed)          # gerador de números aleatórios (com semente)
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
    Análise de Componentes Principais (PCA). Encontra as direções de maior variância dos dados: centraliza os dados, calcula a covariância amostral e obtém seus autovetores e autovalores com np.linalg.eigh, ordenados do maior para o menor autovalor.

    Atributos (disponíveis após o fit):
    media: média de cada atributo.
    componentes: autovetores, um em cada coluna.
    autovalores: variância de cada componente.

    Métodos:
    fit(X): calcula os atributos acima. Exige pelo menos 2 amostras e retorna self.
    transformar(X, n_componentes=None): centraliza X e projeta nos primeiros n_componentes (todos, se None).
    fit_transform(X, n_componentes=None): fit seguido de transformar.

    O sinal de cada componente é arbitrário, e os dados não são padronizados (padronize antes se os atributos tiverem escalas diferentes).
    """
    def __init__(self):
        self.mean = None                   # média de cada atributo original, para centralizar, pois calculamos a variancia amostral sobre a média
        self.components = None             # autovetores ordenados
        self.eigenvalues = None             # variâncias associados aos autovetores (as direções)

    def fit(self, X: np.ndarray):
        X = validate_matrix(X)
        if len(X) < 2:                      # se a dimensão for <=1 não temos dimensão para reduzir, calcular covariância usa n - 1.
            raise ValueError("PCA precisa de pelo menos 2 amostras")

        self.mean = X.mean(axis=0)                 #média dos vetores colunas (amostra)
        X_centralizado = X - self.mean   #centralizando os dados

        cov = X_centralizado.T @ X_centralizado / (len(X) - 1)  #calculando a matriz de covariância amostral
        valores, vetores = np.linalg.eigh(cov)                  #calculando os auto valores e os autovetores (padronizados)

        indices = np.argsort(valores)[::-1]                     #pegando os índices dos eigenvalues em ordem decrescente
        self.eigenvalues = np.maximum(valores[indices], 0.0)    #substituindo os autovalroes negativos por 0, que podem surgir a instabilidade numérica do ponto flutuante (variância negativo não exite)
        self.components = vetores[:, indices]                  #reordenando as colunas (veotres colunas)

        return self

    def transform(self, X: np.ndarray, n_components: int = None) -> np.ndarray:
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

    def fit_transform(self, X: np.ndarray, n_components: int = None) -> np.ndarray:
        return self.fit(X).transform(X, n_components)

# ====================================TESTES=======================================================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # KMeans
    # dois grupos bem separados: um perto de (0,0) e outro perto de (10,10)
    X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(10, 0.5, (50, 2))])

    modelo = KMeans(k=2).fit(X)
    rotulos = modelo.predict(X)

    assert len(set(rotulos[:50])) == 1   # primeiro grupo todo no mesmo cluster, verifica se tem o mesmo rótulo
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
    # o produto escalar dever ser cos(0) = 1 para vetores iguais, pois o angulo entre eles é 0
    assert Z.shape == (200, 1)
    print("PCA OK")
