from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Input, Flatten
from tensorflow.keras import Model, Sequential
import numpy as np
import tensorflow as tf


def conv_model(input_shape, kernel_size, stride, out_channels, avg_pool_size, dense_units,
               num_classes, batch_size=None, sequential=False):
    """
    Creates a convolutional model.
    :param input_shape: The input shape.
    :param kernel_size: The kernel size.
    :param stride: The stride.
    :param out_channels: The number of out channels.
    :param avg_pool_size: The average pool size.
    :param dense_units: The dense units.
    :param num_classes: The num classes.
    :return: A convolutional model.
    """
    if sequential is True:
        return _get_sequential_model(input_shape, kernel_size, stride, out_channels, avg_pool_size, dense_units,
                                     num_classes, batch_size)
    else:
        return _get_model(input_shape, kernel_size, stride, out_channels, avg_pool_size, dense_units,
                          num_classes)


def _get_model(input_shape, kernel_size, stride, out_channels, avg_pool_size, dense_units, num_classes):
    input_x = Input(input_shape, name="input")
    x = Conv2D(out_channels, kernel_size=kernel_size, activation='relu', strides=stride,
               padding='valid', bias_initializer='glorot_uniform')(input_x)
    x = AveragePooling2D(pool_size=avg_pool_size)(x)
    x_shape = x.get_shape()[1:]
    x = tf.reshape(x, [-1, np.prod(x_shape)])

    for dense_unit in dense_units:
        x = Dense(dense_unit, activation='relu', bias_initializer='glorot_uniform')(x)
    output = Dense(num_classes, name="output", bias_initializer='glorot_uniform')(x)

    return Model(inputs=input_x, outputs=output)


def _get_sequential_model(input_shape, kernel_size, stride, out_channels, avg_pool_size, dense_units,
                          num_classes, batch_size=None):
    layers = [Conv2D(out_channels, kernel_size=kernel_size, activation='relu', strides=stride,
                     padding='valid', bias_initializer='glorot_uniform', batch_input_shape=(batch_size,) + input_shape,
                     name="input"),
              AveragePooling2D(pool_size=avg_pool_size),
              Flatten()]

    for dense_unit in dense_units:
        layers.append(Dense(dense_unit, activation='relu', bias_initializer='glorot_uniform'))
    layers.append(Dense(num_classes, name="output", bias_initializer='glorot_uniform'))

    return Sequential(layers)
