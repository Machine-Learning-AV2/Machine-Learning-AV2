import numpy as np

# Converte um valor real em um valor no intervalo (0, 1)
def sigmoid(z: float) -> float:
  return 1/(1+np.exp(-z))

# Versão do sigmoide para mais variáveis (scores). A soma total de probabilities deve ser 1.
def multi_sigmoid(scores: np.array) -> np.array:
  n = len(scores)
  probabilities = np.zeros(n)
  exp = 0
  
  for i in range(n-1):
    exp += np.exp(scores[i])
  for i in range(n-1):
    probabilities[i] = np.exp(scores[i]) / (1+exp)
    
  probabilities[-1] = 1-np.sum(probabilities)
  return probabilities
  
class KNN():
  
  def __init__(self, K = 5):
    self.K = K
    self.X = None
    self.Y = None

  def distance(self):
    pass
    # Aqui, há de se redimensionar os dados e decidir a métrica usada.
  
  def train(self, X, Y):
    self.X = X
    self.Y = Y

  def predict_element(self, P):
    neighborhood = []
    
    for i in range(len(self.X)):
      d = distance(P, self.X[i])
      if i < k:
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
  
  def __init__(self):
    pass

  def train(self, X, Y):
    pass

  def predict_probabilities(self, X):
    pass
    
  def predict(self, X):
    pass


class LDA():
  def __init__(self):
    pass

class QDA():
  def __init__(self):
    pass
    
# -----TESTES-----
if __name__ == "__main__":
  pass
