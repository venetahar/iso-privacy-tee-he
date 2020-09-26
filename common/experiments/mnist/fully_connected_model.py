import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten


def fully_connected_model(input_shape, dense_units, num_classes, batch_size=None, sequential=False):
    """
    Returns a FullyConnectedModel.

    :param input_shape: The input shape.
    :param dense_units: The dense units.
    :param num_classes: The number of classes.
    :param batch_size:
    :param sequential:
    """
    if sequential:
        return _get_sequential_model(input_shape, dense_units, num_classes, batch_size)
    else:
        return _get_model(input_shape, dense_units, num_classes)


def _get_model(input_shape, dense_units, num_classes):
    input_x = Input(input_shape, name="input")
    flatten_shape = np.prod(input_shape)
    x = tf.reshape(input_x, [-1, flatten_shape])

    for dense_unit in dense_units:
        x = Dense(dense_unit, activation='relu', bias_initializer='glorot_uniform')(x)

    output = Dense(num_classes, name="output", bias_initializer='glorot_uniform')(x)

    return Model(inputs=input_x, outputs=output)


def _get_sequential_model(input_shape, dense_units, num_classes, batch_size):

    layers = [Flatten(input_shape=input_shape, batch_size=batch_size, name="input")]

    print("Batch size " + str(batch_size))
    for dense_unit in dense_units:
        layers.append(Dense(dense_unit, activation='relu', bias_initializer='glorot_uniform'))
    layers.append(Dense(num_classes, name="output", bias_initializer='glorot_uniform'))

    return Sequential(layers)
