import numpy as np

# Verifica se os dados de input para treino possuem algum erro
def train_check(X, Y):
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
    
def predict_element_check(P, n_features):
  if P.ndim != 1:
      raise ValueError("Input must be 1D")
  if P.shape[-1] != n_features:
      raise ValueError("Test and train must contain same number of features")
  if not np.issubdtype(P.dtype, np.number):
      raise TypeError("Input must contain numerical values")
    
# Converte um valor real em um valor no intervalo (0, 1)
def sigmoid(z: float) -> float:
  return 1/(1+np.exp(-z))

# Versão do sigmoide para mais variáveis (scores). A soma total de probabilities deve ser 1.
def multi_sigmoid(scores: np.array) -> np.array:
  n = len(scores)
  probabilities = np.zeros(n)
  exp = 0
  
  for i in range(n):
    exp += np.exp(scores[i])
  for i in range(n):
    probabilities[i] = np.exp(scores[i]) / (exp)
    
  return probabilities
  
class KNN():
  
  def __init__(self, K = 5):
    if not isinstance(K, (int, np.integer)):
      raise TypeError("K must be integer number")
    if K <= 0:
      raise ValueError("K must be positive number")
    
    self.K = K
    self.X = None
    self.Y = None

  def distance(self, P1, P2):
    pass
    # Aqui, há de se redimensionar os dados e decidir a métrica usada.
  
  def train(self, X, Y):
    train_check(X, Y)
    self.X = X
    self.Y = Y

  def predict_element(self, P):
    if self.X is None:
      raise TypeError("The model must be trained")
    
    neighborhood = []
    
    for i in range(len(self.X)):
      d = self.distance(P, self.X[i])
      if i < self.K:
        neighborhood.append((d, self.Y[i]))
      else:
        index_max_distance = np.argmax([neighbor[0] for neighbor in neighborhood])
        if d < neighborhood[index_max_distance][0]:
          neighborhood[index_max_distance] = (d, self.Y[i])
        # Verificar se não é possível otimizar esse argmax

    classes = np.array([neighbor[1] for neighbor in neighborhood])
    values, counting = np.unique(classes, return_counts=True)
    return values[np.argmax(counting)]

  def predict(self, L):
    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)


class Logistic_Regression():
  
  def __init__(self, learning_rate = 0.01, iterations = 1000, intercept=True):
    self.beta = None
    self.beta0 = None
    self.values = None
    self.alpha = learning_rate
    self.iter = iterations
    self.intercept = intercept

  def train(self, X, Y):
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
      P = np.array([multi_sigmoid(z) for z in Z]) # Apenas traduz score para probabilidade em cada linha

      # Fórmula dos gradientes. O termo P-T pode ser interpretado como a distância na previsão
      grad_beta = X.T @ (P - T) / n_samples
      grad_beta0 = np.mean(P - T, axis=0)

      # Atualização dos coeficientes. o processo deve repetir a partir daqui.
      self.beta -= self.alpha * grad_beta
      if self.intercept:
        self.beta0 -= self.alpha * grad_beta0

  def predict_probabilities(self, X):
    Z = X @ self.beta
    if self.intercept:
      Z += self.beta0
    return multi_sigmoid(Z)

  def predict_element(self, P):
    predict_element_check(P, self.beta.shape[0])
    ind = np.argmax(self.predict_probabilities(P))
    return self.values[ind]
    
  def predict(self, L):
    if self.beta is None:
      raise TypeError("The model must be trained")
    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)
    

class LDA():
  def __init__(self):
    self.log_pi = None
    self.means = None
    self.sigma_inv = None
    self.values = None

  def train(self, X, Y):
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
    # Necessário verificar possibilidade de erro na inversão

  def aux(self, X, sigma_inv, mean, log_freq):
    # Essa fórmula é a que deve ser maximizada pelo LDA
    return (X.T - 1/2 * mean.T) @ sigma_inv @ mean + log_freq

  def predict_element(self, P):
    predict_element_check(P, self.means.shape[1])
    
    # Busca a classe que maximiza aux para esses dados novos
    vector_aux = np.zeros(len(self.values))
    
    for i in range(len(self.values)):
      vector_aux[i] = self.aux(P, self.sigma_inv, self.means[i], self.log_pi[i])
      
    return self.values[np.argmax(vector_aux)]

  def predict(self, L):
    if self.means is None:
      raise TypeError("The model must be trained")
      
    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)
      

class QDA():
  def __init__(self):
    self.log_pi = None
    self.means = None
    self.sigmas = None
    self.sigmas_inv = None
    self.values = None

  def train(self, X, Y):
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

    # Cria as matrizes de covariância para cada classe
    sigmas = []

    for i, k in enumerate(values):
      Xk = X[Y == k]
      X_centered = Xk - means[i]
      sigma_k = X_centered.T @ X_centered / (len(Xk)-1)
      sigmas.append(sigma_k)
    sigmas_inv = [np.linalg.inv(sigma) for sigma in sigmas]
    

    self.log_pi = log_pi
    self.means = means
    self.sigmas = np.array(sigmas)
    self.sigmas_inv = np.array(sigmas_inv)
    self.values = values
    # Necessário verificar possibilidade de erro na inversão

  def aux(self, X, sigma, sigma_inv, mean, log_freq):
    # Essa fórmula é a que deve ser maximizada pelo QDA
    return -1/2 * (X - mean).T @ sigma_inv @ (X - mean) + log_freq - 1/2 * np.log(np.linalg.det(sigma))

  def predict_element(self, P):
    predict_element_check(P, self.means.shape[1])
    
    # Busca a classe que maximiza aux para esses dados novos
    vector_aux = np.zeros(len(self.values))
    
    for i in range(len(self.values)):
      vector_aux[i] = self.aux(P, self.sigmas[i], self.sigmas_inv[i], self.means[i], self.log_pi[i])
      
    return self.values[np.argmax(vector_aux)]

  def predict(self, L):
    if self.means is None:
      raise TypeError("The model must be trained")
      
    predictions = []
    for P in L:
      predictions.append(self.predict_element(P))
    return np.array(predictions)


# -----TESTES-----
print(sigmoid(0))
print(sigmoid(1))
print(sigmoid(-1))

n = np.array([1, 1, 1])
print(multi_sigmoid(n))

n = np.array([2, 1, 1])
print(multi_sigmoid(n))

n = np.array([1, 2, 2])
print(multi_sigmoid(n))

n = np.array([1, 3, 1])
print(multi_sigmoid(n))

n = np.array([1.5, 1, 0, -1, -1.5])
print(multi_sigmoid(n))

n = np.array([5])
print(multi_sigmoid(n))

n = np.array([1, 1.01])
print(multi_sigmoid(n))

X = np.array([[0, 0], [1, -0.5], [2, 2], [3, 5], [4, 2.5], [1.5, 3.5], [-1, 1.5], [-3, 2.5], [-2.5, -0.5], [-0.5, -1], [-2, 7]])
Y = np.array(["Red", "Red", "Blue", "Blue", "Blue", "Blue", "Red", "Red", "Red", "Red", "Green"])
X_teste = np.array([[0, 0], [3, 3], [-2, 0], [-4, 6], [0, -4]])

model = Logistic_Regression()
model.train(X, Y)
print(model.predict(X_teste))

model1 = KNN()
model2 = KNN(K=1)
model3 = KNN(K=10)
model1.train(X, Y)
model2.train(X, Y)
model3.train(X, Y)
print(model1.predict(X_teste))
print(model2.predict(X_teste))
print(model3.predict(X_teste))

model = LDA()
model.train(X, Y)
print(model.predict(X_teste))

mode1 = QDA()
model.train(X, Y)
print(model.predict(X_teste)) # Erro de covariância
