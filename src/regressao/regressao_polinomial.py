import numpy as np
from regressao import Regressao

class RegressaoPolinomial(Regressao):
    def __init__(self, grau=2):
        super().__init__()

        if not isinstance(grau, int) or isinstance(grau, bool) or grau < 1:
            raise ValueError("O grau do polinômio deve ser um inteiro positivo.")

        self.grau = grau

    def _transformar(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_pol = np.column_stack([X**i for i in range(self.grau + 1)])

        return X_pol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("As dimensões de X e y são incompatíveis.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X não pode ser vazio.")

        X_design = self._transformar(X)

        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        self.coeficientes = beta
        
        return self
    
    def predict(self, X):
        if self.coeficientes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")
        
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_design = self._transformar(X)

        return X_design @ self.coeficientes

if __name__ == "__main__":
    X = np.array([1, 2, 3, 4])
    y = np.array([2, 5, 10, 17])

    modelo = RegressaoPolinomial(grau=2)
    modelo.fit(X, y)

    print("Coeficientes:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
