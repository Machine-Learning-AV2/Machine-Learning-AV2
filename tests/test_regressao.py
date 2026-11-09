import numpy as np
import pytest

from src.Regressao import (
    MinimosQuadrados,
    Ridge,
    Lasso,
    RegressaoPolinomial
)


def test_minimos_quadrados():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = 2 + 3 * X[:, 0]

    modelo = MinimosQuadrados()
    modelo.fit(X, y)

    assert np.allclose(modelo.coeficientes, [2, 3])


def test_ridge_alpha_zero():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = 2 + 3 * X[:, 0]

    ols = MinimosQuadrados()
    ridge = Ridge(alpha=0)

    ols.fit(X, y)
    ridge.fit(X, y)

    assert np.allclose(ols.coeficientes, ridge.coeficientes)


def test_lasso_alpha_zero():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = 2 + 3 * X[:, 0]

    ols = MinimosQuadrados()
    lasso = Lasso(alpha=0, max_iter=10000)

    ols.fit(X, y)
    lasso.fit(X, y)

    assert np.allclose(ols.coeficientes, lasso.coeficientes, atol=1e-4)


def test_lasso_alpha_alto():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)

    X_c = X - np.mean(X, axis=0)
    y_c = y - np.mean(y)

    alpha_max = np.max(np.abs(X_c.T @ y_c)) / len(y)

    modelo = Lasso(alpha=alpha_max + 1)
    modelo.fit(X, y)

    assert np.all(modelo.coeficientes[1:] == 0)


def test_ridge_coeficientes_diminuem():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 5, 8], dtype=float)

    ridge_1 = Ridge(alpha=0.1)
    ridge_2 = Ridge(alpha=10)

    ridge_1.fit(X, y)
    ridge_2.fit(X, y)

    assert np.linalg.norm(ridge_2.coeficientes[1:]) < np.linalg.norm(
        ridge_1.coeficientes[1:]
    )


def test_regressao_polinomial():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = 1 + X[:, 0] ** 2

    modelo = RegressaoPolinomial(grau=2)
    modelo.fit(X, y)

    assert np.allclose(modelo.predict(X), y)


def test_predict_antes_fit():
    modelo = MinimosQuadrados()

    with pytest.raises(RuntimeError):
        modelo.predict(np.array([[1], [2]]))
