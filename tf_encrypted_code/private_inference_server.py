import argparse
import os

import tf_encrypted as tfe
import tf_encrypted.keras.backend as KE
import tensorflow as tf

from common.experiments.malaria.constants import MALARIA_MODEL_NAME, MALARIA_MODEL_PATH, MALARIA_INPUT_SHAPE, \
    MALARIA_NUM_CLASSES
from common.experiments.mnist.constants import MNIST_INPUT_SHAPE, MNIST_NUM_CLASSES, MNIST_MODEL_PATH, \
    MNIST_FULLY_CONNECTED_MODEL_NAME, MNIST_CONV_MODEL_NAME
from common.model_factory import ModelFactory, FULLY_CONNECTED_MODEL_TYPE, CONV_MODEL_TYPE, CONV_POOL_MODEL_TYPE

request_id = 1


def step_fn():
    global request_id
    print("Served encrypted prediction {i} to client.".format(i=request_id))
    request_id += 1


def load_model_from_weights(model_type, input_shape, num_classes, model_path, model_name, batch_size):
    model = ModelFactory.create_model(model_type, input_shape, num_classes, batch_size, True)
    print(model.summary())
    model.load_weights(os.path.join(model_path, 'weights', model_name + '.h5'))
    return model


def load_config():
    config = tfe.RemoteConfig.load("/tmp/tfe.config")
    tfe.set_config(config)


def serve_model(model_type, input_shape, num_classes, model_path, model_name, batch_size=50):
    model = load_model_from_weights(model_type, input_shape, num_classes, model_path, model_name, batch_size)

    load_config()
    tfe.set_protocol(tfe.protocol.SecureNN())

    tf.reset_default_graph()
    with tfe.protocol.SecureNN():
        tfe_model = tfe.keras.models.clone_model(model)

    server = tfe.serving.QueueServer(
        input_shape=(batch_size,) + input_shape, output_shape=(batch_size, num_classes), computation_fn=tfe_model
    )

    session = KE.get_session()
    server.run(session, step_fn=step_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The experiment name. Can be either: mnist_fc, mnist_conv or malaria_conv')
    parser.add_argument('--batch_size', type=int, help='The batch size to use during testing. '
                                                       'Make sure it`s the same as the one provided to the server. '
                                                       'Due to the way tf_encrypted works with fixing the batch size'
                                                       'it`s best to provide a batch size which exactly divides the num'
                                                       'test samples.')
    config = parser.parse_args()

    if config.experiment_name == 'mnist_fc':
        serve_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_INPUT_SHAPE, MNIST_NUM_CLASSES, MNIST_MODEL_PATH,
                    MNIST_FULLY_CONNECTED_MODEL_NAME + '_seq', config.batch_size)
    elif config.experiment_name == 'mnist_conv':
        serve_model(CONV_MODEL_TYPE, MNIST_INPUT_SHAPE, MNIST_NUM_CLASSES, MNIST_MODEL_PATH,
                    MNIST_CONV_MODEL_NAME + '_seq', config.batch_size)
    elif config.experiment_name == 'malaria_conv':
        serve_model(CONV_POOL_MODEL_TYPE, MALARIA_INPUT_SHAPE, MALARIA_NUM_CLASSES,
                    MALARIA_MODEL_PATH, MALARIA_MODEL_NAME + '_seq', config.batch_size)
    else:
        print("Please supply a valid experiment type. Can be either: mnist_fc, mnist_conv or malaria_conv ")

# load_model_from_weights(CONV_POOL_MODEL_TYPE, MALARIA_INPUT_SHAPE, MALARIA_NUM_CLASSES,
#                         MALARIA_MODEL_PATH, MALARIA_MODEL_NAME, 13)