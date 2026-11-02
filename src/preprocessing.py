#   fit(X) - learns the parameters (means, minimums, categories...) using only the training data
#   transform(X) - applies the transformation with the learned parameters
#   fit_transform(X) - fit followed by transform

import numpy as np


# Helper function to validate the data matrix
def validate_matrix(X) -> np.ndarray:

    """Converts X to a 2D float numpy matrix (n_samples, n_features)

    Parameters
    X : Input data."""

    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        raise TypeError('X must contain only numeric data')
    if X.ndim == 1:
        X = X.reshape(-1, 1) # reshape((a,b)) returns a 2d-array with dimensions a x b, where -1 means "whatever is needed to make it work"
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0: # shape[0] rows, shape[1] columns
        raise ValueError('X must be 2D and non-empty: (n_samples, n_features)')
    if not np.all(np.isfinite(X)): # np.isfinite returns a boolean array of the same shape as X, True for finite numbers and False for NaN or infinity. np.all(A) returns True if all elements of the array A are True, otherwise False.
        raise ValueError('X contains NaN or infinity')
    return X


# Split the data into training and test sets
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, seed: int | None = None) -> tuple:

    """Shuffles the data and splits it into training and test sets

    Parameters
    X : Matrix (n_samples, n_features) with the independent variables;
    y : Vector (n_samples,) with the dependent variables;
    test_size : Fraction of the data that goes to the test set;
    seed : Seed of the random generator, for reproducible results.

    Returns
    X_train, X_test, y_train, y_test

    Examples
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.3, seed=0) # 7 training samples and 3 test samples"""

    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    if n != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(test_size, (float)) or not 0 < test_size < 1: # isinstance(X,(type1, type2,...)) returns if X is an instance/object of any of the classes in the tuple.
        raise ValueError('test_size must be between 0 and 1')

    n_test = int(round(n * test_size))
    if n_test == 0 or n_test == n:
        raise ValueError('The split would leave the training or the test set empty')

    # here we could have used np.random.shuffle, but it shuffles in place and we want to keep the original order of X and y. So we use np.random.permutation to get a random permutation of the indices.
    # we could also have used np.random.permutation(n) directly, and permutation(n) would be feed with the global numpy number generator that will be seeded with a random seed. This though would break reproducibility. Tip from claude
    inds = np.random.default_rng(seed).permutation(n) # np.random.default_rng(seed) creates a new random number generator with the given seed (same seed produces the same sequence, for reproducible results). If seed is None, a new random number generator is created with a random seed. permutation(n) returns a random permutation of the integers from 0 to n-1.
    test, train = inds[:n_test], inds[n_test:]
    return X[train], X[test], y[train], y[test]


# Standardization (z-score): mean 0 and standard deviation 1 in each column
class StandardScaler:

    def __init__(self):

        """Standardizes each feature: z = (x - mean) / standard deviation

        Useful for KNN, K-means, PCA, Ridge, Lasso and neural networks, which depend on the scale of the features.
        Constant columns (standard deviation 0) become 0, avoiding division by zero.

        Attributes
        mean : Mean of each feature in the training data;
        std : Standard deviation of each feature in the training data.

        Examples
        standardScaler = StandardScaler()
        X_train_s = standardScaler.fit_transform(X_train)
        X_test_s = standardScaler.transform(X_test) # Uses the training mean and standard deviation"""

        self.mean = None
        self.std = None

    def fit(self, X):
        """Computes the mean and the standard deviation of each column of X. X is meant to be the training data only."""
        X = validate_matrix(X)
        self.mean = X.mean(axis=0) # mean(axis=0) computes the mean of each column (feature) of X, resulting in a 1D array of size equal to the number of features.
        std = X.std(axis=0)
        self.std = np.where(std == 0, 1.0, std) # where(boolean_array, x_array, y_array) builds an array with value x_array[i] where boolean_array[i] is True and y_array[i] where boolean_array[i] is False, to avoid division by zero when standardizing constant columns.
        return self

    def transform(self, X):
        """Returns (X - mean) / std. X here is meant to be the training or test data."""
        if self.mean is None:
            raise RuntimeError('Call fit before transform')
        X = validate_matrix(X)
        if X.shape[1] != self.mean.size:
            raise ValueError('Number of features differs from the one used in fit')
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """fit followed by transform for the training data"""
        return self.fit(X).transform(X)


# Min-max normalization: maps each column to the interval [0, 1]
class MinMaxScaler:

    def __init__(self):

        """Normalizes each feature to the interval [0, 1]: x' = (x - min) / (max - min)

        Constant columns (max = min) become 0.

        Attributes
        data_min : Minimum of each feature in the training data;
        data_max : Maximum of each feature in the training data.

        Examples
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)"""

        self.data_min = None
        self.data_max = None

    def fit(self, X):
        """Computes the minimum and the maximum of each column of X. X is meant to be the training data only."""
        X = validate_matrix(X)
        self.data_min = X.min(axis=0)
        self.data_max = X.max(axis=0)
        return self

    def transform(self, X):
        """Returns (X - min) / (max - min). X is meant to be the training or test data."""
        if self.data_min is None:
            raise RuntimeError('Call fit before transform')
        X = validate_matrix(X)
        if X.shape[1] != self.data_min.size:
            raise ValueError('Number of features differs from the one used in fit')
        data_range = self.data_max - self.data_min
        data_range = np.where(data_range == 0, 1.0, data_range) # Same logic of the StandardScaler, avoids division by zero in constant columns
        return (X - self.data_min) / data_range

    def fit_transform(self, X):
        """fit followed by transform for the training data"""
        return self.fit(X).transform(X)


# One hot encoding: a qualitative variable with K categories -> K columns of 0/1 (dummy variables)
class CategoricalEncoder:

    def __init__(self, drop_first: bool = False):

        """Maps a vector of categories to a matrix of indicator (dummy) variables

        Needed to use encode qualitative features (e.g. color, city) in models that only accept numbers.
        This was meant to be used together with ColumnCategoricalEncoder to pass the columns to encode.

        Parameters
        drop_first : If True, creates K-1 columns, leaving the first category as the baseline
        (avoids perfect collinearity with the intercept in linear regression) (Default: False).

        Attributes
        categories : Categories found in fit, in ascending order.

        Examples
        color = np.array(['blue', 'green', 'blue', 'red'])
        CategoricalEncoder().fit_transform(color)
        # Columns: blue, green, red
        # [[1, 0, 0],
        #  [0, 1, 0],
        #  [1, 0, 0],
        #  [0, 0, 1]]"""

        self.drop_first = drop_first
        self.categories = None

    def fit(self, x):
        """Stores the distinct categories of x. x is meant to be the training data only."""
        x = np.asarray(x)
        if x.ndim != 1 or x.size == 0:
            raise ValueError('x must be a non-empty 1D vector')
        self.categories = np.unique(x) # np.unique returns the sorted unique elements of an array.
        if self.drop_first and self.categories.size < 2:
            raise ValueError('At least 2 categories are needed to drop the first one')
        return self

    def transform(self, x):
        """Returns the (n_samples, n_categories) matrix of 0/1. x is meant to be the training or test data."""
        if self.categories is None:
            raise RuntimeError('Call fit before transform')
        x = np.asarray(x)
        unknown = np.setdiff1d(x, self.categories)
        if unknown.size > 0:
            raise ValueError(f'Categories not seen in fit: {unknown}') # it can happen if the test data has categories that were not present in the training data. In this case, with the current code, we would need to always have all categories represented in training data for the code to work well later. Raising error kind of forces the user to check if the test data contains unknown categories. Another option would be to create a column of zeros for unknown categories, but that would require changing the number of columns in the test set and would conflict with the possibility of dropping the first category. It's simply easier and safer to raise an error, we'll let the user handle it alone.
        matrix = (x.reshape(-1, 1) == self.categories.reshape(1, -1)).astype(float) # (x.reshape(-1, 1) == self.categories.reshape(1, -1)) creates a boolean matrix from the column and line arrays where a comparision between the elements (n,1) (1, m) results in a (n, m) boolean input of the array. This is a feature from numpy.
        return matrix[:, 1:] if self.drop_first else matrix # matrix[:, 1:] returns elements of the array starting from the row 0 and for each row, starting from column 1.

    def fit_transform(self, x):
        """fit followed by transform for the training data"""
        return self.fit(x).transform(x)


# One hot encoding applied to chosen columns of a whole matrix
class ColumnCategoricalEncoder:

    def __init__(self, columns: list | None = None, drop_first: bool = False):

        """Applies CategoricalEncoder to the chosen columns of a matrix and keeps the other columns as numbers

        Each chosen column is replaced, in the same position, by its dummy columns.
        The other columns must be numeric and are kept unchanged.

        Parameters
        columns : List of indices (between 0 and m-1) of the qualitative columns to encode.
                  If None, no column is encoded (Default: None);
        drop_first : Passed to each CategoricalEncoder (Default: False).

        Attributes
        encoders : Dictionary {column index: fitted CategoricalEncoder of that column, the corresponding object self from the CategoricalEncoder class};
        n_features : Number of columns of the matrix used in fit (len(self.columns)).

        Examples
        X = np.array([['blue', 1.5], ['red', 2.0], ['blue', 0.5]], dtype=object)
        ColumnCategoricalEncoder(columns=[0]).fit_transform(X)
        # Columns: blue, red, original column 1
        # [[1, 0, 1.5],
        #  [0, 1, 2.0],
        #  [1, 0, 0.5]]"""

        self.columns = [] if columns is None else list(columns)
        self.drop_first = drop_first
        self.encoders = None
        self.n_features = None

    def _as_matrix(self, X):
        # dtype=object keeps each value as it is (strings stay strings, numbers stay numbers). Without it, numpy would turn every value of a matrix with strings into a string.
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError('X must be 2D and non-empty: (n_samples, n_features)')
        return X

    def fit(self, X):
        """Fits one CategoricalEncoder for each chosen column of X. X is meant to be the training data only."""
        X = self._as_matrix(X)
        m = X.shape[1] # number of columns; (a, b, c,...) for tensor with shape a*b*c*...
        for j in self.columns:
            if not isinstance(j, (int, np.integer)) or isinstance(j, bool) or not 0 <= j < m:
                raise ValueError(f'Column indices must be integers between 0 and {m - 1}')
        if len(set(self.columns)) != len(self.columns): # if the list has repeated indices, we force the user to fix it, because this would be almost surely a typo. They can artifically duplicate columns in the original matrix if they really want duplicated encoders, but I see no reason for that.
            raise ValueError('Column indices must not be repeated')
        self.encoders = {j: CategoricalEncoder(self.drop_first).fit(X[:, j]) for j in self.columns} # X[:, j] is the j-th column, fit comes from the class CategoricalEncoder, it's not the same as the fit method of ColumnCategoricalEncoder. We preferred to keep the classes methods name the same to facilitate usage, even if it makes the code a bit confusing. This comment here is to explicitly state that the method comes from the other class. 
        self.n_features = m
        return self

    def transform(self, X):
        """Returns the matrix with each chosen column replaced by its dummy columns. X is meant to be the training or test data."""
        if self.encoders is None:
            raise RuntimeError('Call fit before transform')
        X = self._as_matrix(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features differs from the one used in fit')
        blocks = []
        for j in range(self.n_features):
            if j in self.encoders:
                blocks.append(self.encoders[j].transform(X[:, j])) # appends a 2D array of shape (n_samples, n_categories) to the list of blocks. transform is from the class CategoricalEncoder, not from ColumnCategoricalEncoder. Same logic as in the fit method.
            else:
                try:
                    blocks.append(validate_matrix(X[:, j])) # a column-array with numerical values of shape (n_samples, 1)
                except TypeError:
                    raise TypeError(f'Column {j} is not numeric: you may add it to columns to encode it')
        return np.column_stack(blocks) # Numpy method that joins the list of 2D arrays into a single 2D array by stacking them horizontally (column-wise).

    def fit_transform(self, X):
        """fit followed by transform for the training data"""
        return self.fit(X).transform(X)


# Tests (by claude) for the functions created

if __name__ == '__main__':

    tol = 1e-9 # Error tolerance
    indx = 0

    def check(condition):
        global indx
        indx += 1
        if condition:
            print(f'{indx} Passed')
        else:
            print(f'{indx} Failed')

    def raises(error, function, *args):
        try:
            function(*args)
        except error:
            return True
        return False

    X = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])

    # ----- StandardScaler -----
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    # Must have mean 0 and standard deviation 1 in each column
    check(np.allclose(Z.mean(axis=0), 0, atol=tol) and np.allclose(Z.std(axis=0), 1, atol=tol))
    # Must return (1-2.5)/sqrt(1.25) ≃ -1.341640786
    check(abs(Z[0, 0] - (-1.341640786)) < tol)
    # A constant column becomes 0
    check(np.allclose(StandardScaler().fit_transform(np.array([[5.], [5.], [5.]])), 0))
    # transform before fit must raise RuntimeError
    check(raises(RuntimeError, StandardScaler().transform, X))
    # A different number of columns must raise ValueError
    check(raises(ValueError, scaler.transform, np.ones((2, 3))))
    # NaN must raise ValueError
    check(raises(ValueError, StandardScaler().fit, np.array([[1.], [np.nan]])))

    # ----- MinMaxScaler -----
    mm = MinMaxScaler()
    N = mm.fit_transform(X)
    # Must return [0, 1/3, 2/3, 1] in both columns
    check(np.allclose(N[:, 0], [0, 1/3, 2/3, 1]) and np.allclose(N[:, 1], [0, 1/3, 2/3, 1]))
    # New data outside the training range may leave [0, 1]: x = 5 -> 4/3
    check(abs(mm.transform(np.array([[5., 10.]]))[0, 0] - 4/3) < tol)
    # A constant column becomes 0
    check(np.allclose(MinMaxScaler().fit_transform(np.array([[5.], [5.], [5.]])), 0))

    # ----- CategoricalEncoder -----
    color = np.array(['blue', 'green', 'blue', 'red'])
    ce = CategoricalEncoder ()
    # Columns: blue, green, red
    check(np.array_equal(ce.fit_transform(color), [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    # With drop_first, 'blue' becomes the row [0, 0]
    check(np.array_equal(CategoricalEncoder(drop_first=True).fit_transform(color), [[0, 0], [1, 0], [0, 0], [0, 1]]))
    # An unknown category must raise ValueError
    check(raises(ValueError, ce.transform, np.array(['purple'])))

    # ----- train_test_split -----
    X_big = np.arange(40).reshape(20, 2)
    y_big = np.arange(20)
    X_tr, X_te, y_tr, y_te = train_test_split(X_big, y_big, 0.25, seed=0)
    # Must give 15 training and 5 test samples
    check(X_tr.shape == (15, 2) and X_te.shape == (5, 2) and y_tr.size == 15 and y_te.size == 5)
    # The rows stay paired with y and no sample is repeated
    check(np.array_equal(X_tr[:, 0] // 2, y_tr) and np.array_equal(np.sort(np.concatenate([y_tr, y_te])), y_big))
    # Same seed -> same split
    check(np.array_equal(train_test_split(X_big, y_big, 0.25, seed=0)[3], y_te))
    # An invalid test_size must raise ValueError
    check(raises(ValueError, train_test_split, X_big, y_big, 1.5))

    # ----- ColumnCategoricalEncoder -----
    X_mix = np.array([[1.5, 'blue', 'yes'], [2.0, 'red', 'no'], [0.5, 'blue', 'no']], dtype=object)
    cce = ColumnCategoricalEncoder(columns=[1, 2])
    # Columns: original column 0, blue, red, no, yes
    check(np.allclose(cce.fit_transform(X_mix), [[1.5, 1, 0, 0, 1], [2.0, 0, 1, 1, 0], [0.5, 1, 0, 1, 0]]))
    # With drop_first: original column 0, red, yes
    check(np.allclose(ColumnCategoricalEncoder([1, 2], drop_first=True).fit_transform(X_mix), [[1.5, 0, 1], [2.0, 1, 0], [0.5, 0, 0]]))
    # transform on new data uses the categories learned in fit
    check(np.allclose(cce.transform(np.array([[3.0, 'red', 'yes']], dtype=object)), [[3.0, 0, 1, 0, 1]]))
    # columns=None keeps the matrix as it is
    check(np.allclose(ColumnCategoricalEncoder().fit_transform(X), X))
    # A text column not listed in columns must raise TypeError
    check(raises(TypeError, ColumnCategoricalEncoder([1]).fit_transform, X_mix))
    # An index out of range must raise ValueError
    check(raises(ValueError, ColumnCategoricalEncoder([3]).fit, X_mix))
    # An unknown category must raise ValueError
    check(raises(ValueError, cce.transform, np.array([[3.0, 'purple', 'yes']], dtype=object)))
