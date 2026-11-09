import numpy as np
import pytest

from src.arvores_decisao import (
    DecisionTreeClassification,
    DecisionTreeRegression
)


def test_classificacao_simples():
    X = np.array([
        [1],
        [2],
        [8],
        [9]
    ])

    y = np.array([0, 0, 1, 1])

    arvore = DecisionTreeClassification(max_depth=3)
    arvore.adjust(X, y)

    previsoes = arvore.predict(X)

    assert np.array_equal(previsoes, y)


def test_classificacao_uma_classe():
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([1, 1, 1])

    arvore = DecisionTreeClassification()
    arvore.adjust(X, y)

    previsoes = arvore.predict(X)

    assert np.array_equal(previsoes, y)


def test_regressao_simples():
    X = np.array([
        [1],
        [2],
        [8],
        [9]
    ])

    y = np.array([2, 2, 8, 8])

    arvore = DecisionTreeRegression(max_depth=3)
    arvore.adjust(X, y)

    previsoes = arvore.predict(X)

    assert np.allclose(previsoes, y)


def test_regressao_folha():
    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([2, 4, 6])

    arvore = DecisionTreeRegression(max_depth=0)
    arvore.adjust(X, y)

    previsoes = arvore.predict(X)

    assert np.allclose(previsoes, np.array([4, 4, 4]))
