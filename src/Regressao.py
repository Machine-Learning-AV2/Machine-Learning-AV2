import numpy as np
import warnings


class Regressao:
    def __init__(self):
        self.coeficientes = None

    def _validar(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X deve ser um vetor ou uma matriz bidimensional.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X não pode ser vazio.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contém valores não finitos.")

        if y is None:
            return X

        y = np.asarray(y, dtype=float)

        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("As dimensões de X e y são incompatíveis.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contém valores não finitos.")

        return X, y

    def fit(self, X, y):
        """Treina o modelo com X e y."""
        raise NotImplementedError(
            "O método fit deve ser implementado na subclasse."
        )

    def predict(self, X):
        """Retorna a previsão com base no modelo."""
        if self.coeficientes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")

        X = self._validar(X)

        if X.shape[1] != len(self.coeficientes) - 1:
            raise ValueError("Número incorreto de features fornecido para previsão.")

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        return X_design @ self.coeficientes


class Lasso(Regressao):
    """
    Regressão Lasso por descida cíclica de coordenadas.

    Minimiza:

    (1 / (2n)) * ||y_c - X_c @ beta||² + alpha * ||beta||_1

    A convenção de alpha é a mesma utilizada pelo scikit-learn.

    As variáveis são centralizadas internamente, mas não são
    padronizadas. Recomenda-se padronizar X antes do treinamento
    quando as features possuem escalas muito diferentes.
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-9):
        super().__init__()

        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("O parâmetro alpha deve ser não-negativo.")

        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
            raise ValueError("O parâmetro max_iter deve ser positivo.")

        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("O parâmetro tol deve ser positivo.")

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = self._validar(X, y)

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
        else:
            warnings.warn(
                "O Lasso não convergiu dentro de max_iter. "
                "Considere aumentar max_iter ou ajustar tol.",
                RuntimeWarning
            )

        intercepto = y_media - X_media @ beta
        self.coeficientes = np.concatenate(([intercepto], beta))

        return self


class RegressaoPolinomial(Regressao):
    """
    Regressão polinomial de uma variável explicativa.

    Para um polinômio de grau d:

    y = beta_0 + beta_1*x + ... + beta_d*x**d
    """

    def __init__(self, grau=2):
        super().__init__()

        if not isinstance(grau, int) or isinstance(grau, bool) or grau < 1:
            raise ValueError("O grau do polinômio deve ser um inteiro positivo.")

        self.grau = grau

    def _transformar(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(
                "A regressão polinomial aceita apenas uma feature."
            )

        x = X[:, 0]

        X_pol = np.column_stack([x**i for i in range(self.grau + 1)])

        return X_pol

    def fit(self, X, y):
        X, y = self._validar(X, y)

        if X.shape[1] != 1:
            raise ValueError(
                "A regressão polinomial aceita apenas uma feature."
            )

        X_design = self._transformar(X)

        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        self.coeficientes = beta

        return self

    def predict(self, X):
        if self.coeficientes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")

        X = self._validar(X)

        if X.shape[1] != 1:
            raise ValueError(
                "A regressão polinomial aceita apenas uma feature."
            )

        X_design = self._transformar(X)

        return X_design @ self.coeficientes


class MinimosQuadrados(Regressao):
    def fit(self, X, y):
        X, y = self._validar(X, y)

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        self.coeficientes = beta

        return self


class Ridge(Regressao):
    """
    Regressão Ridge.

    Minimiza:

    (1/n) * ||y_c - X_c @ beta||² + alpha * ||beta||²

    O intercepto não é penalizado.

    Em relação ao scikit-learn, que utiliza a função objetivo
    ||y - X @ beta||² + alpha_sklearn * ||beta||²,
    a relação entre os parâmetros é:

    alpha_sklearn = n * alpha

    As variáveis são centralizadas internamente, mas não são
    padronizadas. Recomenda-se padronizar X antes do treinamento
    quando as features possuem escalas muito diferentes.
    """

    def __init__(self, alpha=1.0):
        super().__init__()

        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("O parâmetro alpha deve ser não-negativo.")

        self.alpha = alpha

    def fit(self, X, y):
        X, y = self._validar(X, y)

        X_media = X.mean(axis=0)
        y_media = y.mean()

        X_c = X - X_media
        y_c = y - y_media

        p = X_c.shape[1]

        matriz = (
            X_c.T @ X_c
            + X.shape[0] * self.alpha * np.eye(p)
        )

        beta, *_ = np.linalg.lstsq(
            matriz,
            X_c.T @ y_c,
            rcond=None
        )

        intercepto = y_media - X_media @ beta
        self.coeficientes = np.concatenate(([intercepto], beta))

        return self


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
    print("Coeficientes:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
    print()

    ## Minimos Quadrados
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 6, 9])

    modelo = MinimosQuadrados()
    modelo.fit(X, y)

    print("Minimos Quadrados:")
    print("Coeficientes:", modelo.coeficientes)
    print("Previsão para x=5:", modelo.predict([5])[0])
    print("Previsão para x=10:", modelo.predict([10])[0])
    print()

