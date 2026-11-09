import numpy as np
from src.metricas import *
import pytest

# Tolerância
tol = 1e-9


a = np.array([3, 5, 21, 7, 5, 8])
b = np.array([4, 6, 12, 4, 4, 1])

c = np.array(['sim', 'sim', 'não', 'sim', 'não'])
#c = np.array([1, 1, 0, 1, 0])
d = np.array(['sim', 'não', 'sim', 'sim', 'não'])
#d = np.array([1, 0, 1, 1, 0])
l_1 = np.array([['sim', 'não', 2], ['não', 'sim', 1]])
#l_1 = np.array([[1, 0, 2], [0, 1, 1]])


e = np.array([[0.5, 0.2, 0.3], [0.1, 0.85, 0.05], [0.6, 0.3, 0.1]])
f = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])


def test_loss():

    assert loss(a, b, 1) == pytest.approx(22, abs=tol)
    assert loss(a, b, 2) == pytest.approx(142, abs=tol)
    assert loss(a, b, 5) == pytest.approx(76_102, abs=tol)
    assert loss(a, b, np.inf) == pytest.approx(9, abs=tol)

    assert loss(c, d, 'ind') == pytest.approx(2, abs=tol)
    assert loss(c, d, 'ind', l_1) == pytest.approx(3, abs=tol)

    assert loss(e, f, 'log') == pytest.approx(1.3664917338, abs=tol)

def test_empirical_risk():

    assert empirical_risk(a, b, 1) == pytest.approx(22/6, abs=tol)
    assert empirical_risk(a, b, 2) == pytest.approx(142/6, abs=tol)
    assert empirical_risk(a, b, 5) == pytest.approx(76_102/6, abs=tol)

    assert empirical_risk(c, d, 'ind') == pytest.approx(0.4, abs=tol)
    assert empirical_risk(c, d, 'ind', l_1) == pytest.approx(0.6, abs=tol)

    assert empirical_risk(e, f, 'log') == pytest.approx(0.4554972446, abs=tol)

def test_sae():
    assert SAE(a, b) == pytest.approx(22, abs=tol)

def test_rss():
    assert RSS(a, b) == pytest.approx(142, abs=tol)

def test_me():
    assert ME(a, b) == pytest.approx(9, abs=tol)

def test_mae():
    assert MAE(a, b) == pytest.approx(22/6, abs=tol)

def test_mse():
    assert MSE(a, b) == pytest.approx(142/6, abs=tol)

def test_rmse():
    assert RMSE(a, b) == pytest.approx((142/6)**(1/2), abs=tol)

def test_tss():
    assert TSS(a) == pytest.approx(1_277/6, abs=tol)

def test_r2():
    assert R2(a, b) == pytest.approx(-1.062953995, abs=tol)

def test_norm_lp():
    assert norm_lp(a, 1) == pytest.approx(49, abs=tol)
    assert norm_lp(b, 2) == pytest.approx(229**(1/2), abs=tol)

def test_dist():
    assert dist(a, b, 2) == pytest.approx(142**(1/2), abs=tol)
    assert dist(a, b, np.inf) == pytest.approx(9, abs=tol)

def test_dif_empircal_risk():
    assert dif_empirical_risk(a, b, 2) == pytest.approx(np.array([-1/3, -1/3, 3, 1, 1/3, 7/3]), abs=tol)
    assert dif_empirical_risk(e, f, 'log') == pytest.approx(-f/(3*e), abs=tol)

def test_DMAE():
    assert DMAE(a, b) == pytest.approx(np.array([-1/6, -1/6, 1/6, 1/6, 1/6, 1/6]), abs=tol)

def test_MSE():
    assert DMSE(a, b) == pytest.approx(np.array([-1/3, -1/3, 3, 1, 1/3, 7/3]), abs=tol)
