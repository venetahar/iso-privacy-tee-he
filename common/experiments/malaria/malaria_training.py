import os

import numpy as np

from common.model_factory import ModelFactory, CONV_POOL_MODEL_TYPE
from common.model_training import ModelTraining
from common.common_utils import CommonUtils
from common.experiments.malaria.constants import MALARIA_NUM_CLASSES, \
    MALARIA_INPUT_SHAPE, TRAINING_PARAMS, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE, TEST_PERCENTAGE, IMG_RESIZE
from common.experiments.malaria.malaria_data_generator import MalariaDataGenerator


def train_malaria_model(model_path, model_name, source_data_path, target_data_path_prefix, sequential=False):
    """
    Trains a Malaria model and saves the model graph.
    """
    malaria_data_generator = MalariaDataGenerator(source_data_path,
                                                  parameters={
                                                      'test_batch_size': TEST_BATCH_SIZE,
                                                      'batch_size': TRAIN_BATCH_SIZE,
                                                      'test_split': TEST_PERCENTAGE,
                                                      'target_size': IMG_RESIZE,
                                                      'training_folder': 'training',
                                                      'testing_folder': 'testing'
                                                  })

    model = ModelFactory.create_model(CONV_POOL_MODEL_TYPE, MALARIA_INPUT_SHAPE, MALARIA_NUM_CLASSES, sequential)
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train_generator(malaria_data_generator.train_data_generator)
    model_training.evaluate_generator(malaria_data_generator.test_data_generator)
    if sequential is True:
        model_name = model_name + '_seq'
    model.save_weights(os.path.join(model_path, 'weights', model_name + '.h5'))
    CommonUtils.sava_data_generator(malaria_data_generator.test_data_generator, target_data_path_prefix)
    CommonUtils.save_graph(model, model_path=model_path, model_name=model_name + '.pb')


def benchmark_malaria_model(model_path, test_data_path, num_runs=20):
    """
    Benchmarks the malaria model.
    :param model_path: The model path.
    :param test_data_path: The test data path.
    :param num_runs: The number of runs.
    """
    test_data = np.load(test_data_path)
    graph_def = CommonUtils.load_model(model_path)
    ModelTraining.benchmark_model(graph_def, num_runs, test_data[0:1])
