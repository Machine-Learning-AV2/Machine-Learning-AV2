import pytest
from src.unsupervised_learning import *
import numpy as np

def test_kmeans():
    rng = np.random.default_rng(0)

    # dois grupos bem separados: um perto de (0,0) e outro perto de (10,10)
    X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(10, 0.5, (50, 2))])

    modelo = KMeans(k=2).fit(X)
    rotulos = modelo.predict(X)

    assert len(set(rotulos[:50])) == 1
    assert len(set(rotulos[50:])) == 1
    assert rotulos[0] != rotulos[50]

    print("KMeans OK")


def test_pca():
    rng = np.random.default_rng(0)

    # dados quase sobre a reta y = 2x
    t = rng.normal(size=200)
    X = np.column_stack([t, 2 * t]) + 0.01 * rng.normal(size=(200, 2))

    modelo = PCA()
    Z = modelo.fit_transform(X, n_components=1)

    esperado = np.array([1, 2]) / np.sqrt(5)

    assert np.isclose(abs(modelo.components[:, 0] @ esperado), 1, atol=1e-3)
    assert Z.shape == (200, 1)

    print("PCA OK")