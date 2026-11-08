import numpy as np

class Regressao:
    def __init__(self):
        self.coeficientes = None

    def fit(self, X, y):
        """Treina o modelo com X e y."""
        raise NotImplementedError("O método fit deve ser implementado na subclasse.")
        

    def predict(self, X):
        """Retorna a previsão com base no modelo."""
        if self.coeficientes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")

        raise NotImplementedError("O método predict deve ser implementado na subclasse.")

class Lasso(Regressao):
    def __init__(self, alpha, max_iter=1000, tol=1e-6):
        super().__init__()
        if alpha < 0:
            raise ValueError("O parâmetro alpha deve ser não-negativo.")
        if max_iter <= 0:
            raise ValueError("O parâmetro max_iter deve ser positivo.")
        if tol <= 0:
            raise ValueError("O parâmetro tol deve ser positivo.")

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
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

        n, p = X_c.shape
        beta = np.zeros(p)

        for _ in range(self.max_iter):
            beta_ant = beta.copy()
            for j in range(p):
                residual = y_c - X_c @ beta + X_c[:, j] * beta[j]
                
                rho = X_c[:, j] @ residual / n
                norma = X_c[:, j] @ X_c[:, j] / n

                if norma == 0:
                    beta[j] = 0
                else:
                    beta[j] = (
                        np.sign(rho) 
                        * max(abs(rho) - self.alpha, 0)
                        / norma
                    )
            if np.max(np.abs(beta - beta_ant)) < self.tol:
                break
        
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
            raise ValueError("Número incorreto de features fornecido para previsão.")
        
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        
        
        return X_design @ self.coeficientes



class RegressaoPolinomial(Regressao):
    def __init__(self, grau: float):
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

class Ridge(Regressao):
    def __init__(self, alpha: int):
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
    ## Ridge
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 6, 9])

    modelo = Ridge(alpha=0.5)
    modelo.fit(X, y)

    print("Regressão Ridge:")
    print("Coeficientes:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
    print()
    
    ## Polinomial
    X = np.array([1, 2, 3, 4])
    y = np.array([2, 5, 10, 17])

    modelo = RegressaoPolinomial(grau=2)
    modelo.fit(X, y)

    print("Regressão Polinomial:")
    print("Coeficientes:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
    print()
    
    ## Lasso 
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 6, 9])

    modelo = Lasso(alpha=0.5)
    modelo.fit(X, y)

    print("Regressão Lasso:")
    print("Coeficiente:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
    print()

    ## Minimos Quadrados
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 6, 9])

    modelo = MinimosQuadrados()
    modelo.fit(X, y)

    print("Minimos Quadrados:")
    print("Coeficiente:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
    print()

