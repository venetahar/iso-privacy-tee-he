from common.experiments.malaria.conv_pool_model import conv_pool_model
from common.experiments.mnist.conv_model import conv_model
from common.experiments.mnist.fully_connected_model import fully_connected_model

FULLY_CONNECTED_MODEL_TYPE = 'FullyConnected3'
CONV_MODEL_TYPE = 'Conv1'
CONV_POOL_MODEL_TYPE = 'Conv2Pool2'


class ModelFactory:
    """
    A Model Factory.
    """

    @staticmethod
    def create_model(model_type, input_shape, num_classes, batch_size=None, sequential=False):
        """
        Returns a model of the appropriate type.
        :param model_type: The model type.
        :param input_shape: The input shape.
        :param num_classes: The number of classes.
        :param batch_size: Optional param, needed for tf_encrypted.
        :param sequential: Whether to return a model using the sequential api, should be False for he-transformer and
                           tf_trusted and True for tf_encrypted.
        :return: An instantiated model.
        """
        if model_type == FULLY_CONNECTED_MODEL_TYPE:
            model = fully_connected_model(input_shape, dense_units=[128, 128], num_classes=num_classes,
                                          batch_size=batch_size, sequential=sequential)
        elif model_type == CONV_MODEL_TYPE:
            model = conv_model(input_shape, kernel_size=5, stride=2, out_channels=5, avg_pool_size=2,
                               dense_units=[100], num_classes=10, batch_size=batch_size, sequential=sequential)
        elif model_type == CONV_POOL_MODEL_TYPE:
            model = conv_pool_model(input_shape=input_shape, kernel_sizes=[5, 5], channels=[36, 36],
                                    stride=1, avg_pool_sizes=[2, 2], dense_units=[72], num_classes=2,
                                    batch_size=batch_size, sequential=sequential)
        else:
            raise ValueError("Invalid model_type provided. ")

        print(model)
        return model
