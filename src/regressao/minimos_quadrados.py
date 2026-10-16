from regressao import Regressao
import numpy as np

class MinimosQuadrados(Regressao):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        X_design = np.column_stack([np.ones(X.shape[0]), X]);

        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        self.coeficientes = beta

        return self

    def predict(self, X):
        X = np.asarray(X)

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        return X_design @ self.coeficientes


if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 6, 9])

    modelo = MinimosQuadrados()
    modelo.fit(X, y)

    print("Coeficiente:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])

