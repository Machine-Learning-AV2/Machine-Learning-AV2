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
