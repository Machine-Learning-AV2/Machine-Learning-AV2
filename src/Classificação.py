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
  def __init__(self):
    pass

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
