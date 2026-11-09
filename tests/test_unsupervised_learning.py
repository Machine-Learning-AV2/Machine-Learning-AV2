import pytest
from src.preprocessing import validate_matrix
import numpy as np

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # KMeans
    # dois grupos bem separados: um perto de (0,0) e outro perto de (10,10)
    X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(10, 0.5, (50, 2))])

    modelo = KMeans(k=2).fit(X)
    rotulos = modelo.predict(X)

    assert len(set(rotulos[:50])) == 1   # primeiro grupo todo no mesmo cluster, verifica se tem o mesmo rótulo
    assert len(set(rotulos[50:])) == 1   # segundo grupo todo no mesmo cluster
    assert rotulos[0] != rotulos[50]     # clusters diferentes
    print("KMeans OK")

    # PCA
    # dados quase sobre a reta y = 2x
    t = rng.normal(size=200) 
    X = np.column_stack([t, 2 * t]) + 0.01 * rng.normal(size=(200, 2))

    PCA = PCA()
    Z = PCA.fit_transform(X, n_components=1)

    esperado = np.array([1, 2]) / np.sqrt(5)                           # direção da reta
    assert np.isclose(abs(PCA.components[:, 0] @ esperado), 1, atol=1e-3)  # abs: sinal é arbitrário
    # o produto escalar dever ser cos(0) = 1 para vetores iguais, pois o angulo entre eles é 0
    assert Z.shape == (200, 1)
    print("PCA OK")