import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class CommonUtils:

    @staticmethod
    def save_graph(model, model_path, model_name='alice_model.pb'):
        """
        Saves the model graph as .pb.
        :param model: The model.
        :param model_path: The model path.
        :param model_name: The model name. By default it's alice_model.pb.
        """
        frozen_graph = CommonUtils.freeze_graph(model=model, session=K.get_session())
        tf.io.write_graph(frozen_graph, model_path, model_name, as_text=False)

    @staticmethod
    def freeze_graph(model, session):
        """
        Freezes the model graph. This code is based on this answer:
        https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
        :param model: The model.
        :param session: The session.
        :return: The frozen graph.
        """
        output_names = [output.op.name for output in model.outputs]
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(variable.op.name for variable in tf.compat.v1.global_variables()).difference([]))
            output_names += [variable.op.name for variable in tf.compat.v1.global_variables()]
            input_graph_def = graph.as_graph_def()
            for node in input_graph_def.node:
                node.device = ""
            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                                  output_names, freeze_var_names)
            frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
            return frozen_graph

    @staticmethod
    def save_data(data, labels, path_prefix):
        """
        Saves the data in an .npy format.
        :param data: The data.
        :param labels: The labels.
        :param path_prefix: The path prefix.
        """
        np.save(path_prefix + 'data', data)
        np.save(path_prefix + 'data_labels', labels)

    @staticmethod
    def load_data(path_prefix):
        data = np.load(path_prefix + 'data.npy')
        data_labels = np.load(path_prefix + 'data_labels.npy')
        return data, data_labels

    @staticmethod
    def sava_data_generator(data_generator, data_path_prefix):
        num_steps = len(data_generator)
        steps_done = 0
        all_data = None
        all_labels = None
        while steps_done < num_steps:
            data, labels = next(data_generator)
            all_data = data if all_data is None else np.concatenate((all_data, data), axis=0)
            all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
            steps_done += 1

        CommonUtils.save_data(all_data, all_labels, data_path_prefix)

    @staticmethod
    def load_model(model_path):
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        return graph_def

    @staticmethod
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
