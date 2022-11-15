import tensorflow as tf
from tensorflow.keras import Model

from utils import create_layer, feature_merging_layers, get_indices_from_batch, obtain_output
import pdb
from tensorflow.keras.layers import Input


class SiameseModel(Model):

    def __init__(self,
                 conf,
                 name="siamese",
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.dataset = conf['dataset']
        self._feature_network = create_layer(conf['feature_network'], group_using_sequential=False)
        self.classification_network = create_layer(conf['classification_network'])
        self.final_layer = create_layer(conf['final_layer'])
        self._feature_combination_method = conf['feature_combination']
        self.feature_fusion_layer = feature_merging_layers[self._feature_combination_method]
        self._normalize_representations = conf.get('normalize_representations', False)

        if conf.get('create_ds_nodes', True):
            self.create_input_and_output_nodes()

        self._from_indices = conf.get('from_indices', False)
        if self._from_indices:
            # here I assume we use indices from the main_features array
            self.all_features = tf.convert_to_tensor(self.dataset.main_features)

        if self._from_indices:
            input_shape = (1,)
        else:
            input_shape = self.dataset.input_shape

        input1 = Input(shape=input_shape, name='seq1')
        input2 = Input(shape=input_shape, name='seq2')
        self.inputs = [input1, input2]
        self.outputs = self(self.inputs)

    def call(self, inputs, training=None, mask=None):
        input1, input2 = inputs
        if self._from_indices:
            input1 = get_indices_from_batch(input1, self.all_features)
            input2 = get_indices_from_batch(input2, self.all_features)

        features1, features2 = self.encoder([input1, input2], training, mask)

        self.merged = self.feature_fusion_layer([features1, features2])

        output = self.final_layer(
            self.classification_network(self.merged, training=training)
        )

        return output

    def predict(self,
                codes,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):

        data = self.dataset.load_main_features(codes)

        if batch_size:

            predictions = []
            for i in range(0, len(codes[0]), batch_size):
                batch = (data[0][i: i + batch_size], data[1][i: i + batch_size])
                predictions_for_batch = self.get_predictions_for_batch(batch)
                predictions.append(predictions_for_batch)

            return tf.concat(predictions, axis=0)

        else:
            return self.get_predictions_for_batch(data)

    def get_predictions_for_batch(self, data):
        features1, features2 = self.encoder(data, training=False)

        merged = self.feature_fusion_layer([features1, features2])

        output = self.final_layer(
            self.classification_network(merged)
        )
        return output

    def encoder(self, inputs, training, mask=None):
        input1, input2 = inputs

        features1 = obtain_output(input1, self._feature_network, training=training)
        features2 = obtain_output(input2, self._feature_network, training=training)
        return features1, features2

    def create_input_and_output_nodes(self):

        training_pairs = self.dataset.x_train
        val_pairs = self.dataset.x_val
        self.dataset.train_dataset = [training_pairs[:, 0], training_pairs[:, 1]]
        self.dataset.target_train = self.dataset.target_train
        self.dataset.validation_dataset = ([val_pairs[:, 0], val_pairs[:, 1]], self.dataset.target_val)
