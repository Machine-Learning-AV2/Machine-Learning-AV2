import numpy as np
import pytest
from src.preprocessing import *

# tests designed by claude

# Error tolerance
tol = 1e-9

# Global test data
X = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])

def test_standard_scaler():
	scaler = StandardScaler()
	Z = scaler.fit_transform(X)

	# must have mean 0 and std 1 in each column
	assert np.allclose(Z.mean(axis=0), 0, atol=tol)
	assert np.allclose(Z.std(axis=0), 1, atol=tol)

	# Must return (1-2.5)/sqrt(1.25) ≃ -1.341640786
	assert Z[0, 0] == pytest.approx(-1.341640786, abs=tol)
	
	# A constant column becomes 0
	assert np.allclose(StandardScaler().fit_transform(np.array([[5.], [5.], [5.]])), 0)
	
	# transform before fit must raise RuntimeError
	with pytest.raises(RuntimeError):
		StandardScaler().transform(X)
		
	# A different number of columns must raise ValueError
	with pytest.raises(ValueError):
		scaler.transform(np.ones((2, 3)))
		
	# NaN must raise ValueError
	with pytest.raises(ValueError):
		StandardScaler().fit(np.array([[1.], [np.nan]]))


def test_min_max_scaler():
	mm = MinMaxScaler()
	N = mm.fit_transform(X)
	
	# Must return [0, 1/3, 2/3, 1] in both columns
	assert np.allclose(N[:, 0], [0, 1/3, 2/3, 1])
	assert np.allclose(N[:, 1], [0, 1/3, 2/3, 1])
	
	# New data outside the training range may leave [0, 1]: x = 5 -> 4/3
	assert mm.transform(np.array([[5., 10.]]))[0, 0] == pytest.approx(4/3, abs=tol)
	
	# A constant column becomes 0
	assert np.allclose(MinMaxScaler().fit_transform(np.array([[5.], [5.], [5.]])), 0)


def test_categorical_encoder():
	color = np.array(['blue', 'green', 'blue', 'red'])
	ce = CategoricalEncoder()
	
	# Columns: blue, green, red
	assert np.array_equal(ce.fit_transform(color), [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
	
	# With drop_first, 'blue' becomes the row [0, 0]
	assert np.array_equal(CategoricalEncoder(drop_first=True).fit_transform(color), [[0, 0], [1, 0], [0, 0], [0, 1]])
	
	# An unknown category must raise ValueError
	with pytest.raises(ValueError):
		ce.transform(np.array(['purple']))


def test_train_test_split():
	X_big = np.arange(40).reshape(20, 2)
	y_big = np.arange(20)
	X_tr, X_te, y_tr, y_te = train_test_split(X_big, y_big, 0.25, seed=0)
	
	# Must give 15 training and 5 test samples
	assert X_tr.shape == (15, 2)
	assert X_te.shape == (5, 2)
	assert y_tr.size == 15
	assert y_te.size == 5
	
	# The rows stay paired with y and no sample is repeated
	assert np.array_equal(X_tr[:, 0] // 2, y_tr)
	assert np.array_equal(np.sort(np.concatenate([y_tr, y_te])), y_big)
	
	# Same seed -> same split
	assert np.array_equal(train_test_split(X_big, y_big, 0.25, seed=0)[3], y_te)
	
	# An invalid test_size must raise ValueError
	with pytest.raises(ValueError):
		train_test_split(X_big, y_big, 1.5)


def test_column_categorical_encoder():
	X_mix = np.array([[1.5, 'blue', 'yes'], [2.0, 'red', 'no'], [0.5, 'blue', 'no']], dtype=object)
	cce = ColumnCategoricalEncoder(columns=[1, 2])
	
	# Columns: original column 0, blue, red, no, yes
	assert np.allclose(cce.fit_transform(X_mix), [[1.5, 1, 0, 0, 1], [2.0, 0, 1, 1, 0], [0.5, 1, 0, 1, 0]])
	
	# With drop_first: original column 0, red, yes
	assert np.allclose(ColumnCategoricalEncoder([1, 2], drop_first=True).fit_transform(X_mix), [[1.5, 0, 1], [2.0, 1, 0], [0.5, 0, 0]])
	
	# transform on new data uses the categories learned in fit
	assert np.allclose(cce.transform(np.array([[3.0, 'red', 'yes']], dtype=object)), [[3.0, 0, 1, 0, 1]])
	
	# columns=None keeps the matrix as it is
	assert np.allclose(ColumnCategoricalEncoder().fit_transform(X), X)
	
	# A text column not listed in columns must raise TypeError
	with pytest.raises(TypeError):
		ColumnCategoricalEncoder([1]).fit_transform(X_mix)
		
	# An index out of range must raise ValueError
	with pytest.raises(ValueError):
		ColumnCategoricalEncoder([3]).fit(X_mix)
		
	# An unknown category must raise ValueError
	with pytest.raises(ValueError):
		cce.transform(np.array([[3.0, 'purple', 'yes']], dtype=object))