import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common.experiments.malaria.constants import MALARIA_NORM_MEAN, MALARIA_NORM_STD


class MalariaDataGenerator:
    """
    Creates a train and test Malaria data generators.
    """

    def __init__(self, data_path, parameters):
        """
        Creates a MalariaDataGenerator.
        :param data_path: The data path containing both training and test data.
        :param parameters: The parameters.
        """

        self.parameters = parameters

        image_data_generator = ImageDataGenerator(
            dtype='float32',
            preprocessing_function=self.normalize
        )

        self.train_data_generator = image_data_generator.flow_from_directory(
            os.path.join(data_path, self.parameters['training_folder']),
            target_size=self.parameters['target_size'],
            batch_size=self.parameters['batch_size'],
            class_mode='categorical'
        )

        self.test_data_generator = image_data_generator.flow_from_directory(
            os.path.join(data_path, self.parameters['testing_folder']),
            target_size=self.parameters['target_size'],
            batch_size=self.parameters['test_batch_size'],
            class_mode='categorical'
        )

    def normalize(self, data):
        """
        Normalizes the data.
        :param data: The data.
        :return: Normalized data.
        """
        return ((data/255.0) - MALARIA_NORM_MEAN) / MALARIA_NORM_STD
