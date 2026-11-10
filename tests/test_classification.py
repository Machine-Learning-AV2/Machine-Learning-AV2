# Testes de sanidade para classificação

import numpy as np
from src.Classification import *
from src.Functions import *

tol = 10**(-9)

# Sigmoide e softmax

def test_sigmoid():
    assert abs(sigmoid(0) - 0.5) < tol
    assert abs(sigmoid(1) - 0.7310585786) < tol
    assert abs(sigmoid(2) + sigmoid(-2) - 1) < tol


def test_softmax():
    # Entradas iguais
    s = softmax(np.array([1, 1, 1]))
    assert abs(s[2] - 1/3) < tol

    # Maior entrada deve gerar maior probabilidade
    s = softmax(np.array([2, 1, 1]))
    assert s[0] > s[1]

    # Vetor unitário
    s = softmax(np.array([5]))
    assert abs(s[0] - 1) < tol

    # Entradas próximas
    s = softmax(np.array([1, 1.01]))
    assert (s[1] - s[0]) > tol
    assert (s[1] - s[0]) < 0.01


# Modelos

X_train = np.array([
    [1.0, 1.0], [1.5, 2.0],
    [2.0, 1.0], [2.5, 2.5],
    [3.0, 1.5], [1.0, 3.0],
    [2.0, 3.0], [3.0, 2.5],
    [3.5, 1.0], [2.5, 3.5],

    [7.0, 1.0], [7.5, 2.0],
    [8.0, 1.0], [8.5, 2.5],
    [9.0, 1.5], [7.0, 3.0],
    [8.0, 3.0], [9.0, 2.5],
    [9.5, 1.0], [8.5, 3.5],

    [4.0, 7.0], [4.5, 8.0],
    [5.0, 7.0], [5.5, 8.5],
    [6.0, 7.5], [4.0, 9.0],
    [5.0, 9.0], [6.0, 8.5],
    [6.5, 7.0], [5.5, 9.5]])

y_train = np.array([0]*10 + [1]*10 + [2]*10)

X_test = np.array([
    [1.8, 1.8], [2.8, 2.0],
    [1.5, 2.8], [3.2, 3.0],
    [2.0, 2.5],

    [7.8, 1.8], [8.8, 2.0],
    [7.5, 2.8], [9.2, 3.0],
    [8.0, 2.5],

    [4.8, 7.8], [5.8, 8.0],
    [4.5, 8.8], [6.2, 8.0],
    [5.0, 8.5]])

y_test = np.array([0]*5 + [1]*5 + [2]*5)


def test_logistic_regression():
    model = Logistic_Regression()
    model.fit(X_train, y_train)

    assert np.all(model.predict(X_test) == y_test)


def test_knn():
    model = KNN()
    model.fit(X_train, y_train)

    assert np.all(model.predict(X_test) == y_test)


def test_lda():
    model = LDA()
    model.fit(X_train, y_train)

    assert np.all(model.predict(X_test) == y_test)


def test_qda():
    model = QDA()
    model.fit(X_train, y_train)

    assert np.all(model.predict(X_test) == y_test)


def test_svm():
    model = SVM()
    model.fit(X_train, y_train)

    assert np.all(model.predict(X_test) == y_test)


# Testes extras

def test_knn_different_values_of_k():
    X = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [1.0, 1.0]])

    y = np.array([0, 0, 0, 1])

    modelK1 = KNN(K=1)
    modelK3 = KNN(K=3)

    modelK1.fit(X, y)
    modelK3.fit(X, y)

    X_test = np.array([[1.0, 0.9]])

    a = modelK1.predict(X_test)
    b = modelK3.predict(X_test)

    assert a[0] != b[0]
