import numpy as np

np.random.seed(0)

import random

random.seed(0)

from common.experiments.malaria.constants import MALARIA_TARGET_DATA_PATH_PREFIX, MALARIA_DATA_PATH, \
    MALARIA_MODEL_PATH, MALARIA_MODEL_NAME
from common.experiments.malaria.malaria_training import train_malaria_model, benchmark_malaria_model
from common.experiments.mnist.constants import MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME, \
    MNIST_CONV_MODEL_NAME, MNIST_TARGET_DATA_PREFIX

import tensorflow as tf

tf.compat.v1.set_random_seed(0)
tf.random.set_random_seed(0)

import os

os.environ['PYTHONHASHSEED'] = str(0)

from common.model_factory import CONV_MODEL_TYPE, FULLY_CONNECTED_MODEL_TYPE
from common.experiments.mnist.mnist_training import train_mnist_model, benchmark_mnist_model


def train_model(model_type, sequential=False):
    """
    Trains a model.
    :param model_type: The model type.
    """
    if model_type == 'mnist_fc':
        train_mnist_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME,
                          MNIST_TARGET_DATA_PREFIX, sequential)
    elif model_type == 'mnist_conv':
        train_mnist_model(CONV_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_CONV_MODEL_NAME,
                          MNIST_TARGET_DATA_PREFIX, sequential)
    elif model_type == 'malaria_conv':
        train_malaria_model(model_path=MALARIA_MODEL_PATH, model_name=MALARIA_MODEL_NAME,
                            source_data_path=MALARIA_DATA_PATH,
                            target_data_path_prefix=MALARIA_TARGET_DATA_PATH_PREFIX, sequential=sequential)


train_model('malaria_conv', True)
# benchmark_mnist_model(MNIST_MODEL_PATH + MNIST_FULLY_CONNECTED_MODEL_NAME + '.pb')
# benchmark_mnist_model(MNIST_MODEL_PATH + MNIST_CONV_MODEL_NAME + '.pb')
# benchmark_malaria_model(MALARIA_MODEL_PATH + MALARIA_MODEL_NAME + '.pb', MALARIA_TARGET_DATA_PATH_PREFIX + 'data.npy')
