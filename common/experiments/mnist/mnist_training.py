from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import numpy as np

from common.model_factory import ModelFactory
from common.model_training import ModelTraining
from common.common_utils import CommonUtils
from common.experiments.mnist.constants import MNIST_NUM_CLASSES, TRAINING_PARAMS, MNIST_WIDTH, MNIST_HEIGHT, \
    NUM_CHANNELS, \
    MNIST_INPUT_SHAPE, MNIST_NORM_MEAN, MNIST_NORM_STD


def train_mnist_model(model_type, model_path, model_name, target_data_path_prefix, sequential=False):
    """
    Trains a MNIST model and saves the model graph.
    :param target_data_path_prefix:
    :param model_name:
    :param model_path: The model path.
    :param model_type: The model type to use.
    :param sequential:
    """
    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, training_labels = preprocess(training_data, training_labels)
    test_data, test_labels = preprocess(test_data, test_labels)

    model = ModelFactory.create_model(model_type, MNIST_INPUT_SHAPE, MNIST_NUM_CLASSES, sequential)
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train(training_data, training_labels)
    model_training.evaluate_plain_text(test_data, test_labels)
    if sequential is True:
        model_name = model_name + '_seq'
    model.save_weights(os.path.join(model_path, 'weights', model_name + '.h5'))
    CommonUtils.save_data(test_data, test_labels, target_data_path_prefix)
    CommonUtils.save_graph(model, model_path, model_name + '.pb')


def preprocess(data, labels):
    """
    Pre-processes the data.
    :param data: The data to pre-process.
    :param labels: The labels.
    :return: Pre-processed data and labels.
    """
    data = data.reshape(data.shape[0], MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)
    data = data.astype('float32')
    data = normalize(data)

    labels = to_categorical(labels, MNIST_NUM_CLASSES)
    return data, labels


def normalize(data):
    """
    Normalizes the data.
    :param data: The data.
    :return: Normalized data.
    """
    return (data / 255.0 - MNIST_NORM_MEAN) / MNIST_NORM_STD


def benchmark_mnist_model(model_path, num_runs=20):
    """
    Benchmarks the malaria model.
    :param model_path: The model path.
    :param num_runs: The number of runs.
    """
    (_, _), (test_data, test_labels) = mnist.load_data()
    test_data, test_labels = preprocess(test_data, test_labels)

    graph_def = CommonUtils.load_model(model_path)
    ModelTraining.benchmark_model(graph_def, num_runs, test_data[0:1])
