import numpy as np


class DecisionTreeClassification:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.n_caracteristicas = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0

        _, counting = np.unique(y, return_counts=True)
        probabilities = counting / len(y)
        return 1.0 - np.sum(probabilities**2)

    def _best_division(self, X, y):
        n_samples, n_predictors = X.shape
        best_gini = self._gini(y)
        best_division = None

        for j in range(n_predictors):
            values = np.unique(X[:, j])
            limits = (values[:-1] + values[1:]) / 2

            for limit in limits:
                mask_left = X[:, j] <= limit
                y_left = y[mask_left]
                y_right = y[~mask_left]

                n_left = len(y_left)
                n_right = len(y_right)

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)

                gini_weighted = (
                    (n_left / n_samples) * gini_left
                    + (n_right / n_samples) * gini_right
                )

                if gini_weighted < best_gini:
                    best_gini = gini_weighted
                    best_division = (j, limit)

        return best_division

    def _build_tree(self, X, y, depth=0):
        classes, counting = np.unique(y, return_counts=True)
        class_majority = classes[np.argmax(counting)]

        if (
            len(classes) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples
        ):
            return {"folha": True, "valor": class_majority}

        division = self._best_division(X, y)

        if division is None:
            return {"folha": True, "valor": class_majority}

        j, limit = division
        mask = X[:, j] <= limit

        return {
            "folha": False,
            "caracteristica": j,
            "limite": limit,
            "esquerda": self._build_tree(
                X[mask], y[mask], depth + 1
            ),
            "direita": self._build_tree(
                X[~mask], y[~mask], depth + 1
            )
        }

    def adjust(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional")

        if y.ndim != 1:
            raise ValueError("y deve ser um vetor unidimensional")

        if len(X) != len(y):
            raise ValueError("X e y devem ter o mesmo número de exemplos")

        if len(y) == 0:
            raise ValueError("Os dados de treinamento não podem estar vazios")

        if X.shape[1] == 0:
            raise ValueError("X deve ter pelo menos uma característica")

        self.root = self._build_tree(X, y)

        return self

    def _predict_sample(self, x, no):
        if no["folha"]:
            return no["valor"]

        j = no["caracteristica"]

        if x[j] <= no["limite"]:
            return self._predict_sample(x, no["esquerda"])
        else:
            return self._predict_sample(x, no["direita"])

    def predict(self, X):
        if self.root is None:
            raise ValueError(
                "A árvore de decisão precisa ser treinada antes de prever"
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional")

        predictions = [
            self._predict_sample(x, self.root)
            for x in X
        ]

        return np.asarray(predictions)


class DecisionTreeRegression:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def _mse(self, y):
        if len(y) == 0:
            return 0.0

        mean = np.mean(y)
        return np.mean((y - mean)**2)

    def _best_division(self, X, y):
        n_samples, n_predictors = X.shape
        best_error = self._mse(y)
        best_division = None

        for j in range(n_predictors):
            values = np.unique(X[:, j])
            limits = (values[:-1] + values[1:]) / 2

            for limit in limits:
                mask = X[:, j] <= limit
                y_left = y[mask]
                y_right = y[~mask]

                n_left = len(y_left)
                n_right = len(y_right)

                error_weighted = (
                    (n_left / n_samples) * self._mse(y_left)
                    + (n_right / n_samples) * self._mse(y_right)
                )

                if error_weighted < best_error:
                    best_error = error_weighted
                    best_division = (j, limit)

        return best_division

    def _build_tree(self, X, y, depth=0):
        mean = float(np.mean(y))

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples
            or np.all(y == y[0])
        ):
            return {"folha": True, "valor": mean}

        division = self._best_division(X, y)

        if division is None:
            return {"folha": True, "valor": mean}

        j, limit = division
        mask = X[:, j] <= limit

        return {
            "folha": False,
            "caracteristica": j,
            "limite": limit,
            "esquerda": self._build_tree(
                X[mask], y[mask], depth + 1
            ),
            "direita": self._build_tree(
                X[~mask], y[~mask], depth + 1
            )
        }

    def adjust(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X deve ser armazenada em uma matriz bidimensional")

        if y.ndim != 1:
            raise ValueError("y deve ser um vetor unidimensional")

        if len(X) != len(y):
            raise ValueError("X e y devem ter o mesmo número de exemplos")

        if len(y) == 0:
            raise ValueError("Os dados de treinamento não podem estar vazios")

        if X.shape[1] == 0:
            raise ValueError("X deve ter pelo menos uma característica")

        self.root = self._build_tree(X, y)

        return self

    def _predict_sample(self, x, no):
        if no["folha"]:
            return no["valor"]

        j = no["caracteristica"]

        if x[j] <= no["limite"]:
            return self._predict_sample(x, no["esquerda"])

        return self._predict_sample(x, no["direita"])

    def predict(self, X):
        if self.root is None:
            raise ValueError(
                "A árvore precisa ser treinada antes de prever"
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional")

        if X.shape[1] == 0:
            raise ValueError("X deve ter pelo menos uma característica")

        predictions = [
            self._predict_sample(x, self.root)
            for x in X
        ]

        return np.asarray(predictions)



if __name__ == "__main__":

    # Teste de classificação
    X_train = np.array([
        [1.0, 1.0, 1.5],
        [1.5, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [7.0, 1.0, 7.0],
        [7.5, 2.0, 8.0],
        [8.0, 1.0, 7.5],
        [4.0, 7.0, 4.0],
        [4.5, 8.0, 5.0],
        [5.0, 7.0, 4.5]
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    X_test = np.array([
        [1.8, 1.5, 1.5],
        [7.8, 1.5, 7.5],
        [4.5, 7.5, 4.5]
    ])

    y_test = np.array([0, 1, 2])

    modelo = DecisionTreeClassification()
    modelo.adjust(X_train, y_train)

    previsoes = modelo.predict(X_test)

    print("Classificação")
    print("Previsões:", previsoes)
    print("Classes reais:", y_test)
    print("Acurácia:", np.mean(previsoes == y_test))


    # Teste de regressão
    X_train_reg = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 2.0],
        [1.0, 2.0, 2.0],
        [3.0, 2.0, 1.0],
        [2.0, 3.0, 1.0],
        [3.0, 3.0, 2.0]
    ])

    y_train_reg = np.array([4.0, 5.0, 5.0, 7.0, 9.0, 10.0])

    X_test_reg = np.array([
        [1.5, 1.5, 1.5],
        [2.5, 2.0, 1.5]
    ])

    y_test_reg = np.array([4.5, 7.0])

    modelo_reg = DecisionTreeRegression()
    modelo_reg.adjust(X_train_reg, y_train_reg)

    previsoes_reg = modelo_reg.predict(X_test_reg)

    print("\nRegressão")
    print("Previsões:", previsoes_reg)
    print("Valores reais:", y_test_reg)
    print("Erro absoluto médio:",
          np.mean(np.abs(previsoes_reg - y_test_reg)))
