import numpy as np
from regressao import Regressao

class Ridge(Regressao):
    def __init__(self, alpha=1.0):
        super().__init__()

        if alpha < 0:
            raise ValueError("O parâmetro alpha deve ser não-negativo.")
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("As dimensões de X e y são incompatíveis.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X não pode ser vazio.")

        X_media = X.mean(axis=0)
        y_media = y.mean()

        X_c = X - X_media
        y_c = y - y_media

        p = X_c.shape[1]

        beta = np.linalg.solve(X_c.T @ X_c + X.shape[0] * self.alpha * np.eye(p), X_c.T @ y_c)

        intercepto = y_media - X_media @ beta
        self.coeficientes = np.concatenate(([intercepto], beta))
        
        return self
    
    def predict(self, X):
        if self.coeficientes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")
        
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != len(self.coeficientes) - 1:
            raise ValueError("Número incorreto de features.")

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        return X_design @ self.coeficientes

if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 6, 9])

    modelo = Ridge(alpha=0.5)
    modelo.fit(X, y)

    print("Coeficientes:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
