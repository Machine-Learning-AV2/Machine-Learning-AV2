import pytest
import numpy as np
from metricas import *

# Tolerância
tol = 1e-9


a = np.array([3, 5, 21, 7, 5, 8])
b = np.array([4, 6, 12, 4, 4, 1])

c = np.array(['sim', 'sim', 'não', 'sim', 'não'])
d = np.array(['sim', 'não', 'sim', 'sim', 'não'])
l_1 = np.array([['sim', 'não', 2], ['não', 'sim', 1]])


def test_loss():

    assert loss(a, b, 1) == pytest.approx(22, abs=tol)
    assert loss(a, b, 2) == pytest.approx(142, abs=tol)
    assert loss(a, b, 5) == pytest.approx(76_102, abs=tol)
    assert loss(a, b, np.inf) == pytest.approx(9, abs=tol)

    assert loss(c, d, 'ind') == pytest.approx(2, abs=tol)
    assert loss(c, d, 'ind', l_1) == pytest.approx(3, abs=tol)

def test_empirical_risk():

    assert empirical_risk(a, b, 1) == pytest.approx(3.666666666, abs=tol)
    assert empirical_risk(a, b, 2) == pytest.approx(23.666666666, abs=tol)
    assert empirical_risk(a, b, 5) == pytest.approx(12_683.666666666, abs=tol)

    assert empirical_risk(c, d, 'ind') == pytest.approx(0.4, abs=tol)
    assert empirical_risk(c, d, 'ind', l_1) == pytest.approx(0.6, abs=tol)

def test_sae():
    assert SAE(a, b) == pytest.approx(22, abs=tol)

def test_rss():
    assert RSS(a, b) == pytest.approx(142, abs=tol)

def test_me():
    assert ME(a, b) == pytest.approx(9, abs=tol)

def test_mae():
    assert MAE(a, b) == pytest.approx(3.666666666, abs=tol)

def test_mse():
    assert MSE(a, b) == pytest.approx(23.666666666, abs=tol)

def test_rmse():
    assert RMSE(a, b) == pytest.approx(4.864839839, abs=tol)

def test_tss():
    assert TSS(a) == pytest.approx(212.833333333, abs=tol)

def test_r2():
    assert R2(a, b) == pytest.approx(-1.062953995, abs=tol)

def test_norm_lp():
    assert norm_lp(a, 1) == pytest.approx(49, abs=tol)
    assert norm_lp(b, 2) == pytest.approx(15.132745950, abs=tol)

def test_dist():
    assert dist(a, b, 2) == pytest.approx(11.916375287, abs=tol)
    assert dist(a, b, np.inf) == pytest.approx(9, abs=tol)