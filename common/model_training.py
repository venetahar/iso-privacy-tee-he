import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from tf_trusted.constants import MODEL_OUTPUT_NAME, MODEL_INPUT_NAME


class ModelTraining:
    """
    Responsible for training a Keras model.
    """

    def __init__(self, model, parameters):
        """
        Creates a ModelTraining.
        :param model: The model used for training.
        :param parameters: The training parameters.
        """
        self.model = model
        self.parameters = parameters
        if self.parameters['optimizer'] == 'Adam':
            optimizer = Adam(self.parameters['learning_rate'])
        else:
            optimizer = SGD(self.parameters['learning_rate'], self.parameters['momentum'], nesterov=True)
        self.model.compile(loss=CategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])

    def train(self, data, labels):
        """
        Trains the model.
        :param data: The data used for training.
        :param labels: The labels.
        """
        self.model.fit(data, labels, batch_size=self.parameters['batch_size'], epochs=self.parameters['num_epochs'],
                       verbose=1, shuffle=True)
        print(self.model.summary())

    def train_generator(self, training_data_generator):
        """
        Performs training using a training data generator.
        :param training_data_generator: The training data generator.
        """
        num_steps = len(training_data_generator)
        self.model.fit_generator(training_data_generator, steps_per_epoch=num_steps,
                                 epochs=self.parameters['num_epochs'], verbose=1, shuffle=True)
        print(self.model.summary())

    def evaluate_plain_text(self, test_data, test_labels):
        """
        Evaluates the model in plain text.
        :param test_data: The test data.
        :param test_labels: The test labels.
        """
        metrics = self.model.evaluate(test_data, test_labels, verbose=0)
        print('Test set: Loss: ({:.4f}%) Accuracy: ({:.4f}%)'.format(metrics[0], metrics[1]))

    def evaluate_generator(self, test_data_generator):
        """
        Evaluates the model using a test data generator.
        :param test_data_generator: The test data generator.
        """
        num_steps = len(test_data_generator)
        metrics = self.model.evaluate_generator(test_data_generator, steps=num_steps)
        print('Test set: Loss: ({:.4f}%) Accuracy: ({:.4f}%)'.format(metrics[0], metrics[1]))

    @staticmethod
    def benchmark_model(graph_def, num_runs, data_instance):
        """
        Measures the runtime of model predicitons.
        :param num_runs: The number of runs.
        :param data_instance: The data instance to use.
        """
        tf.compat.v1.reset_default_graph()
        print("Data instance shape {}".format(data_instance.shape))
        all_metrics = []
        for index in range(0, num_runs):
            with tf.compat.v1.Session() as sess:
                tf.graph_util.import_graph_def(graph_def)
                output_tensor = sess.graph.get_tensor_by_name('import/' + MODEL_OUTPUT_NAME + ':0')
                start_time = time.time()
                sess.run(output_tensor, {'import/' + MODEL_INPUT_NAME + ':0': data_instance})
                end_time = time.time()
                all_metrics.append(end_time - start_time)

        print("============Performance metrics: ============ ")
        print("Average plaintext evaluate model time: {}".format(np.mean(all_metrics)))
