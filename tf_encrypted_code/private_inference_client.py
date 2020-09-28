import argparse
import time

import tf_encrypted as tfe
import numpy as np

from common.common_utils import CommonUtils
from common.experiments.malaria.constants import MALARIA_TARGET_DATA_PATH_PREFIX, MALARIA_INPUT_SHAPE, \
    MALARIA_NUM_CLASSES
from common.experiments.mnist.constants import MNIST_INPUT_SHAPE, MNIST_NUM_CLASSES, MNIST_TARGET_DATA_PREFIX


def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
    """
    Calculates the number of correct predictions.
    :param prediction_scores: The prediction scores.
    :param one_hot_labels: The one hot labels.
    :return: the number of correct predictions.
    """
    predictions = prediction_scores.argmax(axis=1)
    labels = np.where(one_hot_labels == 1)[1]
    return np.sum(predictions == labels)


def private_inference_client(test_data, test_labels, input_shape, num_classes, batch_size=50, benchmark=False):
    tfe_config = tfe.RemoteConfig.load("/tmp/tfe.config")
    tfe.set_config(tfe_config)

    tfe.set_protocol(tfe.protocol.SecureNN())
    client = tfe.serving.QueueClient(
        input_shape=(batch_size,) + input_shape, output_shape=(batch_size, num_classes))

    print("Started client")
    sess = tfe.Session(config=tfe_config)

    if benchmark is True:
        benchmark_inference(batch_size, client, sess, test_data)
    else:
        run_private_inference(batch_size, client, sess, test_data, test_labels)


def run_private_inference(batch_size, client, sess, test_data, test_labels):
    index = 0
    num_samples = test_data.shape[0]
    correct_predictions = 0
    while index < num_samples:
        new_index = index + batch_size if index + batch_size < num_samples else num_samples
        predicted_labels = client.run(sess, test_data[index: new_index])
        correct_predictions += calculate_num_correct_predictions(predicted_labels,
                                                                 test_labels[index: new_index])
        index = new_index
    print('Test set: Accuracy: ({:.4f})'.format(correct_predictions / test_labels.shape[0]))


def benchmark_inference(batch_size, client, sess, test_data, n_runs=20):
    run_times = []
    for i in range(0, n_runs):
        print("Run", i)
        start_time = time.time()
        client.run(sess, test_data[:batch_size])
        end_time = time.time()
        run_times.append(end_time - start_time)

    print("============Performance metrics: ============ ")
    print("Average evaluate model time: {}".format(np.mean(run_times)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The experiment name. Can be either: mnist_fc, mnist_conv or malaria_conv')
    parser.add_argument('--batch_size', type=int, help='The batch size to use during testing. '
                                                       'Make sure it`s the same as the one provided to the server. '
                                                       'Due to the way tf_encrypted works with fixing the batch size'
                                                       'it`s best to provide a batch size which exactly divides the num'
                                                       'test samples.')
    parser.add_argument('--benchmark', action="store_true",
                        help='Whether to benchmark the experiment. Default False.')

    config = parser.parse_args()

    if config.experiment_name == 'mnist_fc' or config.experiment_name == 'mnist_conv':
        test_data, test_labels = CommonUtils.load_data(MNIST_TARGET_DATA_PREFIX)
        private_inference_client(test_data, test_labels, MNIST_INPUT_SHAPE, MNIST_NUM_CLASSES,
                                 config.batch_size, config.benchmark)
    elif config.experiment_name == 'malaria_conv':
        test_data, test_labels = CommonUtils.load_data(MALARIA_TARGET_DATA_PATH_PREFIX)
        private_inference_client(test_data, test_labels, MALARIA_INPUT_SHAPE, MALARIA_NUM_CLASSES,
                                 config.batch_size, config.benchmark)
    else:
        print("Please supply a valid experiment type. Can be either: mnist_fc, mnist_conv or malaria_conv ")
