import numpy as np
import pytest

from src.arvores_decisao import (
    DecisionTreeClassification,
    DecisionTreeRegression
)

# Tolerância
tol = 1e-9

# Dados para classificação
X_class = np.array([
    [1],
    [2],
    [8],
    [9]
])

y_class = np.array([0, 0, 1, 1])

# Dados para regressão
X_reg = np.array([
    [1],
    [2],
    [8],
    [9]
])

y_reg = np.array([2, 2, 8, 8])


def test_classificacao():

    arvore = DecisionTreeClassification(max_depth=3)
    arvore.adjust(X_class, y_class)

    previsoes = arvore.predict(X_class)

    assert np.array_equal(previsoes, y_class)


def test_classificacao_uma_classe():

    y = np.array([1, 1, 1])

    arvore = DecisionTreeClassification()
    arvore.adjust(np.array([[1], [2], [3]]), y)

    assert np.array_equal(arvore.predict(np.array([[1], [2], [3]])), y)


def test_regressao():

    arvore = DecisionTreeRegression(max_depth=3)
    arvore.adjust(X_reg, y_reg)

    previsoes = arvore.predict(X_reg)

    assert np.allclose(previsoes, y_reg, atol=tol)


def test_regressao_profundidade_zero():

    arvore = DecisionTreeRegression(max_depth=0)
    arvore.adjust(X_reg, y_reg)

    previsoes = arvore.predict(X_reg)

    assert np.allclose(previsoes, np.array([5, 5, 5, 5]), atol=tol)


def test_classificacao_dados_vazios():

    arvore = DecisionTreeClassification()

    with pytest.raises(ValueError):
        arvore.adjust(np.empty((0, 1)), np.array([]))
```
