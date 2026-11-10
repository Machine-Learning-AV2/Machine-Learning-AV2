import pytest
import src.NeuralNetwork as NN
import numpy as np


def test_bias_shapes():
    nn = NN.NeuralNetwork([1, 2, 3, 4], 'identity')

    for level in range(len(nn.layers)-1):
        assert nn.bias[level].shape == (nn.layers[level+1], 1)


def test_bias_interval():
    nn = NN.NeuralNetwork([1, 2, 3, 4], 'identity', bias_interval = (-1, 1))

    for level in range(len(nn.layers)-1):
        assert -1 <= np.all(nn.weight[level]) <= 1


def test_weight_shapes():
    nn = NN.NeuralNetwork([1, 2, 3, 4], 'identity')

    for level in range(len(nn.layers)-1):
        assert nn.weight[level].shape == (nn.layers[level+1], nn.layers[level])


def test_weight_interval():
    nn = NN.NeuralNetwork([1, 2, 3, 4], 'identity', weight_interval = (-1, 1))

    for level in range(len(nn.layers)-1):
        assert -1 <= np.all(nn.weight[level]) <= 1


def test_same_seed():
    nn = NN.NeuralNetwork([1, 2, 3, 1], 'sigmoid', seed=45)
    nn_1 = NN.NeuralNetwork([1, 2, 3, 1], 'sigmoid', seed=45)

    assert nn.prevision(0.5) == nn_1.prevision(0.5)


def test_diferent_seed():
    nn = NN.NeuralNetwork([1, 2, 3, 1], 'sigmoid', seed=45)
    nn_2 = NN.NeuralNetwork([1, 2, 3, 1], 'sigmoid', seed=55)

    assert nn.prevision(0.5) != nn_2.prevision(0.5)


def test_activation_function():
    nn = NN.NeuralNetwork([1, 1, 1, 1], 'identity', 'identity', bias_interval=(-10, -10), weight_interval=(1, 1))

    assert nn.activate_functions == ['identity', 'identity', 'identity']


def test_multiple_activation_function():
    nn = NN.NeuralNetwork([1, 1, 1, 1], ['ReLU', 'sigmoid'], 'tanh', bias_interval=(-10, -10), weight_interval=(1, 1))

    assert nn.activate_functions == ['ReLU', 'sigmoid', 'tanh']


def test_invalid_activation_function():
    with pytest.raises(KeyError):
        NN.NeuralNetwork([1, 1, 1, 1], 'identity', 'fake func', bias_interval=(-10, -10), weight_interval=(1, 1))
        NN.NeuralNetwork([1, 1, 1, 1], 'fake func', 'identity', bias_interval=(-10, -10), weight_interval=(1, 1))


def test_invalid_number_activation_function():
    with pytest.raises(ValueError):
        NN.NeuralNetwork([1, 1, 1, 1], ['identity', 'identity', 'identity'], 'fake func', bias_interval=(-10, -10), weight_interval=(1, 1))


def test_prevision_multiple_layer():
    nn = NN.NeuralNetwork([1, 2, 2, 1], 'ReLU', output_activation='ReLU', bias_interval=(1, 1), weight_interval= (2, 2)) #f(x) = 32x + 21

    assert nn.prevision(1) == 53
    assert nn.prevision(0) == 21


def test_prevision_identity():
    nn = NN.NeuralNetwork([1, 1], 'identity', output_activation='identity', bias_interval=(1, 1), weight_interval=(2, 2)) #f(x) = 2x + 1

    assert nn.prevision(1) == 3
    assert nn.prevision(0) == 1
    assert nn.prevision(-1) == -1


def test_prevision_ReLU():
    nn = NN.NeuralNetwork([1, 1], 'ReLU', output_activation='ReLU', bias_interval=(1, 1), weight_interval=(2, 2)) #f(x) = 2x + 1

    assert nn.prevision(1) == 3
    assert nn.prevision(0) == 1
    assert nn.prevision(-1) == 0


def test_prevision_sigmoid():
    nn = NN.NeuralNetwork([1, 1], 'sigmoid', output_activation='sigmoid', bias_interval=(0, 0), weight_interval=(1, 1))

    assert nn.prevision(0) == 0.5


def test_prevision_tanh():
    nn = NN.NeuralNetwork([1, 1], 'tanh', output_activation='tanh', bias_interval=(0, 0), weight_interval=(1, 1))
    assert nn.prevision(0) == 0


def test_save_state_layers():
    nn = NN.NeuralNetwork([1, 2, 2, 1], 'ReLU', output_activation='ReLU', bias_interval=(1, 1), weight_interval= (2, 2)) #f(x) = 32x + 21
    nn.prevision(-1, save_states=True)

    correct_layers = [[-1], [0, 0], [1, 1], [5]]

    for level in range(len(nn.layers)):
        assert np.array_equiv(nn._state_layers[level], np.array(correct_layers[level]).T) 


def test_save_state_linear():
    nn = NN.NeuralNetwork([1, 2, 2, 1], 'ReLU', output_activation='ReLU', bias_interval=(1, 1), weight_interval= (2, 2)) #f(x) = 32x + 21
    nn.prevision(-1, save_states=True)

    correct_linear = [[-1, -1], [1, 1], [5]]

    for level in range(len(nn.layers)-1):

        assert np.array_equiv(nn._state_linear[level], np.array(correct_linear[level]).T) 


def test_example_1():

    rn1 = NN.NeuralNetwork([2, 4, 5, 2], activate_function="sigmoid", output_activation="sigmoid", bias_interval = (-1, 1), weight_interval = (-1, 1))

    rng = np.random.default_rng(1)
    X = rng.uniform(-4, 4, (1000, 2))
    y = np.zeros(X.shape)

    for i, x in enumerate(X):
        if x[0]**2 + x[1]**2 <= 9:
            y[i, :] = [1, 0]
        else:
            y[i, :] = [0, 1]

    rn1.fit(X, y, batch=50)

    x1 = np.array([1, 0])
    x2 = np.array([3, 3])
    x3 = np.array([5, 5])

    rn1.prevision(x1.reshape(1, -1))
    rn1.prevision(x2.reshape(1, -1))
    rn1.prevision(x3.reshape(1, -1))

    rn1.prevision(x1.reshape(1, -1), classify=True)
    rn1.prevision(x2.reshape(1, -1), classify=True)
    rn1.prevision(x3.reshape(1, -1), classify=True)


def test_example_2():
    rn2 = NN.NeuralNetwork([1, 4, 5, 1], activate_function="ReLU", output_activation="identity", bias_interval = (-0.1, 0.1), weight_interval = (-0.1, 0.1))

    f = lambda x: np.exp(x)

    X = np.linspace(-2, 3, 100).reshape(-1, 1)
    y = f(X)

    rn2.fit(X, y, batch=50, learn_rate=0.001, epochs=1000)

    for x1 in range(20):
        rn2.prevision(x1)

def test_save_load():
    import os

    rn = NN.NeuralNetwork([2, 4, 5, 2], activate_function="sigmoid", output_activation="sigmoid", bias_interval = (-1, 1), weight_interval = (-1, 1))
    rn.save('test_save_load.npz')

    rn_copy = NN.load_neural_network('test_save_load.npz')

    os.remove("test_save_load.npz")

    assert rn.layers == rn_copy.layers
    assert rn.activate_functions == rn_copy.activate_functions
    assert all(np.array_equal(a, b) for a, b in zip(rn.bias, rn_copy.bias))
    assert all(np.array_equal(a, b) for a, b in zip(rn.weight, rn_copy.weight))


