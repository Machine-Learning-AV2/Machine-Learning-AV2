import numpy as np
from .metricas import dist
from .Functions import sigmoid, softmax

# Verifica se os dados de input para treino possuem algum erro
def train_check(X: np.ndarray, Y: np.ndarray) -> None:
  if not isinstance(X, np.ndarray):
      raise TypeError("X must be numpy array")
  if not isinstance(Y, np.ndarray):
      raise TypeError("Y must be numpy array")
  if X.ndim != 2:
      raise ValueError("X must be 2D array")
  if Y.ndim != 1:
      raise ValueError("Y must be 1D array")
  if X.shape[0] != Y.shape[0]:
      raise ValueError("X and Y must have the same lenght")
  if X.shape[0] == 0:
      raise ValueError("Data must be not empty")
  if X.shape[1] == 0:
      raise ValueError("Samples must have at least one feature")
  if not np.issubdtype(X.dtype, np.number):
      raise TypeError("X must contain numeric values")

def predict_element_check(P: np.ndarray, n_features: int) -> None:
  if P.ndim != 1:
      raise ValueError("Input must be 1D")
  if P.shape[-1] != n_features:
      raise ValueError("Test and train must contain same number of features")
  if not np.issubdtype(P.dtype, np.number):
      raise TypeError("Input must contain numerical values")
  
class KNN():
  """
  O modelo de K nearest neighbors, para um K escolhido
  """

  def __init__(self, K: int, p=2, normalize=False):
    """
    Inicia o modelo KNN
      --------------
      K: Inteiro que representa o tamanho da vizinhança observada em cada dado. Dessa vizinhança, a maioria elegerá a classe prevista.
      
      p: A norma Lp (Σ Pi^p)^(1/p). Por padrão, a distância euclidiana dos pontos (L2).
      normalize: Caso queira manter um desvio padrão consistente para todo preditor, torne isso True. Os dados serão normalizados para média 0 e variância 1.
    """
    
    if not isinstance(K, (int, np.integer)):
      raise TypeError("K must be integer number")
    if K <= 0:
      raise ValueError("K must be positive number")

    self.K = K
    self.X = None
    self.Y = None
    self.p = p
    self.normalize = normalize
    self.standard = None
    self.mean = None

  def distance(self, P1: np.ndarray, P2: np.ndarray) -> float:
    return dist(P1, P2, self.p)

  def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Treina o modelo KNN
    -----------
      X: array com todos os preditores dos dados de treino.

      Y: array com todas as classes observadas nos dados de treino. Deve ser compatível com a dimensão de X.
    """
    
    train_check(X, Y)
    if self.normalize:
      self.mean = X.mean(axis=0)
      self.standard = X.std(axis=0)
      self.X = (X - self.mean) / self.standard
    else:
      self.X = X
    self.Y = Y

  def predict_element(self, P: np.ndarray):
    if self.X is None:
      raise TypeError("The model must be trained")

    predict_element_check(P, self.X.shape[1])

    if self.normalize:
      P = (P-self.mean)/self.standard

    neighborhood = []

    for i in range(len(self.X)):
      d = self.distance(P, self.X[i])
      if i < self.K:
        neighborhood.append((d, self.Y[i]))
      else:
        index_max_distance = np.argmax([neighbor[0] for neighbor in neighborhood])
        if d < neighborhood[index_max_distance][0]:
          neighborhood[index_max_distance] = (d, self.Y[i])

    classes = np.array([neighbor[1] for neighbor in neighborhood])
    values, counting = np.unique(classes, return_counts=True)
    return values[np.argmax(counting)]

  def predict(self, L: np.ndarray) -> np.ndarray:
    """
    Retorna a previsão do KNN para os dados novos. O modelo deve ter sido treinado antes das previsões.
      -----------
      L: array de novos dados com seus preditores
    """
    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)


class Logistic_Regression():
  """
  O modelo de Regressão Logística, com intercepto opcional
  """

  def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, intercept=True):
    """
    Inicia o modelo de Regressão Logística
    -----------
      learning_rate: Valor do passo na descida de gradiente. Se muito pequeno, pode demorar a convergir ou não chegar ao ótimo. Se muito alto, pode divergir.
      iterations: Número de passos que o algoritmo da descida de gradiente executa.
      intercept: Pode ser interpretado como um viés natural para cada classe ser mais ou menos provável. Torne False se quiser desativá-lo.
    """
    
    self.beta = None
    self.beta0 = None
    self.values = None
    self.alpha = learning_rate
    self.iter = iterations
    self.intercept = intercept

  def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Treina o modelo de Regressão Logística

    X: array com todos os preditores dos dados de treino.
    Y: array com todas as classes observadas nos dados de treino. Deve ser compatível com a dimensão de X.
    """
    
    train_check(X, Y)

    # Usa a decida de gradiente para estimar beta
    n_samples, n_parameters = X.shape
    self.values = np.unique(Y)
    n_classes = len(self.values)
    self.beta = np.zeros((n_parameters, n_classes))
    self.beta0 = np.zeros(n_classes) if self.intercept else None

    # Cria uma matriz one hot samples x classes. Marca apenas uma coluna por linha (a classs observada)
    indices = np.searchsorted(self.values, Y)
    T = np.eye(n_classes)[indices]

    for _ in range(self.iter):

      Z = X @ self.beta # Soma feita através de broadcast. O resultado é uma matriz de mesmo shape que T
      if self.intercept:
        Z += self.beta0
      P = np.array([softmax(z) for z in Z]) # Apenas traduz score para probabilidade em cada linha

      # Fórmula dos gradientes. O termo P-T pode ser interpretado como a distância na previsão
      grad_beta = X.T @ (P - T) / n_samples
      grad_beta0 = np.mean(P - T, axis=0)

      # Atualização dos coeficientes. o processo deve repetir a partir daqui.
      self.beta -= self.alpha * grad_beta
      if self.intercept:
        self.beta0 -= self.alpha * grad_beta0

  def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
    Z = X @ self.beta
    if self.intercept:
      Z += self.beta0
    return softmax(Z)

  def predict_element(self, P: np.ndarray):
    predict_element_check(P, self.beta.shape[0])
    ind = np.argmax(self.predict_probabilities(P))
    return self.values[ind]

  def predict(self, L: np.ndarray) -> np.array:
    """
    Retorna a previsão da Regressão Logística para os dados novos. O modelo deve ter sido treinado antes das previsões.

    L: array de novos dados com seus preditores
    """
    
    if self.beta is None:
      raise TypeError("The model must be trained")
    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)


class LDA():
  """
  O modelo de Análise de Discriminante Linear
  """
  
  def __init__(self):
    """
    Inicia o modelo de QDA. Não há parâmetros de input para essa classe.
    """
    self.log_pi = None
    self.means = None
    self.sigma_inv = None
    self.values = None

  def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Treina o modelo LDA

    X: array com todos os preditores dos dados de treino.
    Y: array com todas as classes observadas nos dados de treino. Deve ser compatível com a dimensão de X.
    """
    
    train_check(X, Y)

    # Toma a frequência de cada classe como a probabilidade desta ocorrer
    values, counting = np.unique(Y, return_counts=True)
    log_pi = np.log(counting/len(Y))

    # Toma a média das features em cada classe
    means = []
    for k in values:
      Xk = X[Y==k]
      mean = np.mean(Xk, axis=0)
      means.append(mean)
    means = np.array(means)

    # Cria a matriz de covariância sigma a partir dos dados
    if len(Y) <= len(values):
      raise ValueError("There's no suficcient information to training LDA (need more samples than classes)")
    sigma_dim = np.shape(X)[1]
    sigma = np.zeros((sigma_dim, sigma_dim))

    for i, k in enumerate(values):
      Xk = X[Y == k]
      X_centered = Xk - means[i]
      sigma += X_centered.T @ X_centered
    sigma /= len(Y) - len(values)

    self.log_pi = log_pi
    self.means = means
    self.sigma_inv = np.linalg.inv(sigma)
    self.values = values

  def aux(self, X, sigma_inv, mean, log_freq):
    # Essa fórmula é a que deve ser maximizada pelo LDA
    return (X.T - 1/2 * mean.T) @ sigma_inv @ mean + log_freq

  def predict_element(self, P: np.ndarray):
    predict_element_check(P, self.means.shape[1])

    # Busca a classe que maximiza aux para esses dados novos
    vector_aux = np.zeros(len(self.values))

    for i in range(len(self.values)):
      vector_aux[i] = self.aux(P, self.sigma_inv, self.means[i], self.log_pi[i])

    return self.values[np.argmax(vector_aux)]

  def predict(self, L: np.ndarray) -> np.array:
    """
    Retorna a previsão do LDA para os dados novos. O modelo deve ter sido treinado antes das previsões.

    L: array de novos dados com seus preditores
    """
    
    if self.means is None:
      raise TypeError("The model must be trained")

    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)


class QDA():
  """
  O modelo de Análise de Discriminante Quadrática
  """
  def __init__(self):
    """
    Inicia o modelo de QDA. Não há parâmetros de input para essa classe.
    """
    self.log_pi = None
    self.means = None
    self.sigmas = None
    self.sigmas_inv = None
    self.values = None

  def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Treina o modelo QDA

    X: array com todos os preditores dos dados de treino.
    Y: array com todas as classes observadas nos dados de treino. Deve ser compatível com a dimensão de X.
    """
    
    train_check(X, Y)

    # Toma a frequência de cada classe como a probabilidade desta ocorrer
    values, counting = np.unique(Y, return_counts=True)
    if np.any(counting < 2):
      raise ValueError("All the classes must have at least 2 samples")
    log_pi = np.log(counting/len(Y))

    # Toma a média das features em cada classe
    means = []
    for k in values:
      Xk = X[Y==k]
      mean = np.mean(Xk, axis=0)
      means.append(mean)
    means = np.array(means)

    # Cria as matrizes de covariância para cada classe
    sigmas = []
    for i, k in enumerate(values):
      Xk = X[Y == k]
      X_centered = Xk - means[i]
      sigma_k = X_centered.T @ X_centered / (len(Xk)-1)
      sigmas.append(sigma_k)

    sigmas_inv = []
    for sigma in sigmas:
      try:
        sigma_inv = np.linalg.inv(sigma)
        sigmas_inv.append(sigma_inv)
      except:
        raise ValueError("Some class didn't generate an invertible covariance matrix")


    self.log_pi = log_pi
    self.means = means
    self.sigmas = np.array(sigmas)
    self.sigmas_inv = np.array(sigmas_inv)
    self.values = values
    # Necessário verificar possibilidade de erro na inversão

  def aux(self, X, sigma, sigma_inv, mean, log_freq):
    # Essa fórmula é a que deve ser maximizada pelo QDA
    return -1/2 * (X - mean).T @ sigma_inv @ (X - mean) + log_freq - 1/2 * np.log(np.linalg.det(sigma))

  def predict_element(self, P: np.ndarray):
    predict_element_check(P, self.means.shape[1])

    # Busca a classe que maximiza aux para esses dados novos
    vector_aux = np.zeros(len(self.values))

    for i in range(len(self.values)):
      vector_aux[i] = self.aux(P, self.sigmas[i], self.sigmas_inv[i], self.means[i], self.log_pi[i])

    return self.values[np.argmax(vector_aux)]
    
  def predict(self, L: np.ndarray) -> np.array:
    """
    Retorna a previsão do QDA para os dados novos. O modelo deve ter sido treinado antes das previsões.

    L: array de novos dados com seus preditores
    """
    if self.means is None:
      raise TypeError("The model must be trained")

    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)


class SVM():
  """
  O modelo de Support Vector Machine linear com margem flexível, treinado por descida do subgradiente no primal.
  Com mais de duas classes, usa a estratégia um-contra-um (one-vs-one).
  """
  def __init__(self, C: float, learning_rate: float = 0.01, max_iterations: int = 1000):
    """
    Inicia o modelo SVM

    C: Peso da penalidade por violar a margem.
    learning_rate: Valor do passo na descida do subgradiente.
    max_iterations: Número de passos (cada um usa todos os dados de treino) que o algoritmo executa.
    """

    if C <= 0:
      raise ValueError("C must be positive number")
    if learning_rate <= 0:
      raise ValueError("learning_rate must be positive number")
    if not isinstance(max_iterations, (int, np.integer)):
      raise TypeError("max_iterations must be integer number")
    if max_iterations <= 0:
      raise ValueError("max_iterations must be positive number")

    self.C = C
    self.alpha = learning_rate
    self.max_iterations = max_iterations
    self.values = None
    self.n_features = None
    self.machines = None  # Lista de (i, j, w, b): SVM binário entre as classes values[i] (-1) e values[j] (+1)

  def train_binary(self, X: np.ndarray, y: np.ndarray):
    # Minimiza J(w,b) = 1/2 ||w||^2 + C/N Σ max(0, 1 - y_i (w^T x_i + b)). Os rótulos y são -1 ou +1.
    N, n_parameters = X.shape
    w = np.zeros(n_parameters)
    b = 0.0

    for _ in range(self.max_iterations):
      # Margens e conjunto M dos exemplos que violam a margem: y_i (w^T x_i + b) < 1
      # influencia que hinge loss penaliza pontos que não respeitam a margem funcional de 1.
      margins = y * (X @ w + b)
      M = margins < 1

      # Subgradientes de J
      grad_w = w - (self.C / N) * (y[M] @ X[M])
      grad_b = -(self.C / N) * np.sum(y[M])

      w = w - self.alpha * grad_w
      b = b - self.alpha * grad_b

    return w, b

  def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Treina o modelo SVM

    X: array com todos os preditores dos dados de treino.
    Y: array com todas as classes observadas nos dados de treino. Deve ser compatível com a dimensão de X. Precisa conter pelo menos 2 classes.
    """

    train_check(X, Y)

    values = np.unique(Y)

    X = X.astype(float)

    # Um SVM binário para cada par de classes (i < j). A classe values[i] vira -1 e values[j] vira +1.
    machines = []
    for i in range(len(values)):
      for j in range(i + 1, len(values)):
        pair = (Y == values[i]) | (Y == values[j]) #pega apneas as amostras das duas classes atuais
        y = np.where(Y[pair] == values[j], 1.0, -1.0) #colocando rótulos 1 e -1
        w, b = self.train_binary(X[pair], y)
        machines.append((i, j, w, b))

    self.values = values
    self.n_features = X.shape[1]
    self.machines = machines

  def predict_element(self, P: np.ndarray):
    if self.machines is None:
      raise TypeError("The model must be trained")

    predict_element_check(P, self.n_features)

    return self.predict(P.reshape(1, -1))[0]

  def predict(self, L: np.ndarray) -> np.ndarray:
    """
    Retorna a previsão do SVM para os dados novos. O modelo deve ter sido treinado antes das previsões.
    Cada SVM binário vota em uma de suas duas classes, e vence a classe com mais votos (em empate, a menor na ordem crescente das classes).

    L: array de novos dados com seus preditores
    """

    if self.machines is None:
      raise TypeError("The model must be trained")

    for P in L:
      predict_element_check(P, self.n_features)

    votes = np.zeros((len(L), len(self.values)), dtype=int)
    for i, j, w, b in self.machines:
      # w^T x + b >= 0 vota em values[j]; caso contrário, em values[i]
      winners = np.where(L @ w + b >= 0, j, i)
      votes[np.arange(len(L)), winners] += 1 #arange cria um vetor com indices de L

    return self.values[np.argmax(votes, axis=1)]


# -----TESTES-----
if __name__ == '__main__':
  print("Sigmoide(0) = ", sigmoid(0))
  print("Sigmoide(1) = ", sigmoid(1))
  print("Sigmoide(-1) = ", sigmoid(-1))

  n = np.array([1, 1, 1])
  print("Softmax(1, 1, 1) = ", softmax(n))
  n = np.array([2, 1, 1])
  print("Softmax(2, 1, 1) = ", softmax(n))
  n = np.array([1, 2, 2])
  print("Softmax(1, 2, 2) = ", softmax(n))
  n = np.array([1, 3, 1])
  print("Softmax(1, 3, 1) = ", softmax(n))
  n = np.array([1.5, 1, 0, -1, -1.5])
  print("Softmax(1.5, 1, 0, -1, -1.5) = ", softmax(n))
  n = np.array([5])
  print("Softmax(5) = ", softmax(n))
  n = np.array([1, 1.01])
  print("Softmax(1, 1.01) = ", softmax(n))
  print()

  modelRL = Logistic_Regression()
  modelRLNI = Logistic_Regression(intercept = False)
  modelRLHL = Logistic_Regression(learning_rate = 1)
  modelRLFI = Logistic_Regression(iterations = 2)
  modelKNN5 = KNN(5)
  modelKNN1 = KNN(1)
  modelKNN10 = KNN(10)
  modelLDA = LDA()
  modelQDA = QDA()
  modelSVM = SVM(1)
  modelSVMHC = SVM(10)
  modelSVMFI = SVM(1, max_iterations=2)

  # Para os testes abaixo, o LDA e o QDA foram testados separadamente nos itens em que se aplicam
  def test(X, Y, X_teste):
    modelRL.fit(X, Y)
    modelRLNI.fit(X, Y)
    modelRLHL.fit(X, Y)
    modelRLFI.fit(X, Y)
    modelKNN5.fit(X, Y)
    modelKNN1.fit(X, Y)
    modelKNN10.fit(X, Y)
    modelSVM.fit(X, Y)
    modelSVMHC.fit(X, Y)
    modelSVMFI.fit(X, Y)

    print("Regressão Logística: ", modelRL.predict(X_teste))
    print("Regressão Logística sem intercepto: ", modelRLNI.predict(X_teste))
    print("Regressão Logística com alto learning rate: ", modelRLHL.predict(X_teste))
    print("Regressão Logística com poucas iterações: ", modelRLFI.predict(X_teste))
    print("KNN com K=5: ", modelKNN5.predict(X_teste))
    print("KNN com K=1: ", modelKNN1.predict(X_teste))
    print("KNN com K=10: ", modelKNN10.predict(X_teste))
    print("SVM: ", modelSVM.predict(X_teste))
    print("SVM com C=10: ", modelSVMHC.predict(X_teste))
    print("SVM com poucas iterações: ", modelSVMFI.predict(X_teste))
    print("Regressão Logística: ", modelRL.predict(X_teste))
    print("Regressão Logística sem intercepto: ", modelRLNI.predict(X_teste))
    print("Regressão Logística com alto learning rate: ", modelRLHL.predict(X_teste))
    print("Regressão Logística com poucas iterações: ", modelRLFI.predict(X_teste))
    print("KNN com K=5: ", modelKNN5.predict(X_teste))
    print("KNN com K=1: ", modelKNN1.predict(X_teste))
    print("KNN com K=10: ", modelKNN10.predict(X_teste))
    print("SVM: ", modelSVM.predict(X_teste))
    print("SVM com C=10: ", modelSVMHC.predict(X_teste))
    print("SVM com poucas iterações: ", modelSVMFI.predict(X_teste))

  # Vários pontos
  X = np.array([[0, 0], [1, -0.5], [2, 2], [3, 5], [-3, 10], [4, 2.5], [1.5, 3.5], [-1, 1.5], [-3, 2.5], [-2.5, -0.5], [-0.5, -1], [-2, 7], [-2.5, 6]])
  Y = np.array(["Red", "Red", "Blue", "Blue", "Green", "Blue", "Blue", "Red", "Red", "Red", "Red", "Green", "Green"])
  X_teste = np.array([[0, 0], [3, 3], [-2, 0], [-4, 6], [0, -4]])
  test(X, Y, X_teste)
  modelLDA.fit(X, Y)
  modelQDA.fit(X, Y)
  print("LDA: ", modelLDA.predict(X_teste))
  print("QDA: ", modelQDA.predict(X_teste))
  print()

  # Pontos em uma linha reta
  X = np.array([[1, 2], [0.6, 2], [1, 2], [0.7, 2], [-10, 2], [0, 2]])
  Y = np.array(["ONE", "TWO", "ONE", "TWO", "THREE", "THREE"])
  X_teste = np.array([[0.5, 1.9], [1.1, 2.7], [0, 2.1], [-1, 2.3], [0.3, 1.6]])
  test(X, Y, X_teste)
  print()
  # Pontos em uma linha reta
  X = np.array([[1, 2], [0.6, 2], [1, 2], [0.7, 2], [-10, 2], [0, 2]])
  Y = np.array(["ONE", "TWO", "ONE", "TWO", "THREE", "THREE"])
  X_teste = np.array([[0.5, 1.9], [1.1, 2.7], [0, 2.1], [-1, 2.3], [0.3, 1.6]])
  test(X, Y, X_teste)
  print()

  # Pontos em uma linha quase reta
  X = np.array([[1, 2], [0.6, 2.1], [1, 2.1], [0.7, 2], [-10, 1.9], [0, 2.2]])
  Y = np.array(["ONE", "TWO", "ONE", "TWO", "THREE", "THREE"])
  X_teste = np.array([[0.5, 1.9], [1.1, 2.7], [0, 2.1], [-1, 2.3], [0.3, 1.6]])
  test(X, Y, X_teste)
  modelLDA.fit(X, Y)
  print("LDA: ", modelLDA.predict(X_teste))
  print()

  # Um único ponto
  X = np.array([[8, -7]])
  Y = np.array(["Only"])
  X_teste = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
  test(X, Y, X_teste)
  print()
  # Um único ponto
  X = np.array([[8, -7]])
  Y = np.array(["Only"])
  X_teste = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
  test(X, Y, X_teste)
  print()

  # Prevendo mudança em uma reta
  X = np.array([[-1.51, -1.49], [-1, -1], [1, 1]])
  Y = np.array(["A", "A", "B"])
  X_teste = np.array([[0.9, 0.9], [0.5, 0.5], [0.25, 0.25], [0.1, 0.1], [0, 0], [-0.1, -0.1], [-0.25, -0.25], [-0.5, -0.5], [-0.9, -0.9]])
  test(X, Y, X_teste)
  modelLDA.fit(X, Y)
  print("LDA: ", modelLDA.predict(X_teste))
  print()

  # Pontos iguais em classes diferentes
  X = np.array([[0, 0], [0, 0], [1, 2], [1, 2], [-1, -1], [-1, -1]])
  Y = np.array(["Equal", "Identical", "Equal", "Identical", "Equal", "Identical"])
  X_teste = np.array([[5, -3], [0, 0]])
  test(X, Y, X_teste)
  modelLDA.fit(X, Y)
  print("LDA: ", modelLDA.predict(X_teste))
  print()

  # Teste métrica KNN
  modelKNN2 = KNN(2)
  modelKNN2sigma = KNN(2, normalize=True)
  X = np.array([[10, 10], [11, 11], [8, 9], [0, 0], [1, -1], [-1000, 1]])
  Y = np.array(["Direito", "Direito", "Direito", "Esquerdo", "Esquerdo", "Esquerdo"])
  X_teste = np.array([[11, 1]])
  modelKNN2.fit(X, Y)
  modelKNN2sigma.fit(X, Y)
  print(modelKNN2.predict(X_teste))
  print(modelKNN2sigma.predict(X_teste))