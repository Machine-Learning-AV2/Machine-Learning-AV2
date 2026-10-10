import numpy as np

class Regressao:
    def __init__(self):
        self.coeficientes = None

    def fit(self, X, y):
        """Treina o modelo com X e y."""
        pass

    def predict(self, X):
        """Retorna a previsão com base no modelo."""
        if self.coeficientes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")
        pass
