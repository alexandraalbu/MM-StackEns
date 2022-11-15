import pdb

import tensorflow as tf
import numpy as np


from model.AbstractGraphModel import AbstractGraphModel
from utils import obtain_output, graph_processing_layers, adj_processing_functions, feature_merging_layers, \
    create_layer, get_indices_from_batch


class GnnLinkPredictionModel(AbstractGraphModel):

    def __init__(self,
                 conf,
                 name="gnn_link",
                 **kwargs):
        super().__init__(conf, name=name, **kwargs)

        self._graph_nn = graph_processing_layers[conf['graph_nn']]
        self._graph_nn_params = conf['graph_nn_params']
        self._num_encoder_layers = len(self._graph_nn_params)
        self._adj_matrix_processing_fn = adj_processing_functions[conf['graph_nn']]
        self._feature_merging_method = conf.get('feature_combination')
        self.feature_fusion_layer = feature_merging_layers[self._feature_merging_method]

        # set up data
        features_source = conf.get('nodes_features_source')
        if features_source == 'graph_nodes':
            # the nodes features array, which is computed in GraphDataset
            nodes_features_np = conf['dataset'].nodes_features
        elif features_source == 'main_features':
            # the main features array, which may contain already normalized values
            nodes_features_np = conf['dataset'].main_features
        elif features_source == 'extra_features':
            # the extra features array, which may contain already normalized values
            nodes_features_np = conf['dataset'].extra_features
        else:
            raise ValueError('invalid nodes_features_source')
        self.features_source = features_source
        self.nodes_features = tf.convert_to_tensor(nodes_features_np)

        self.id_to_index = conf['dataset'].id_to_index
        self.adj_matrix_train = self._adj_matrix_processing_fn(conf['dataset'].adj_matrix_train)
        if conf['dataset'].sparse_adj_matrix:
            self.adj_matrix_train = tf.sparse.from_dense(self.adj_matrix_train)
        self.num_nodes = self.adj_matrix_train.shape[0]

        # build network
        self.encoder_layers = [self._graph_nn(**self._graph_nn_params[i]) for i in range(self._num_encoder_layers)]
        self.initial_encoder = create_layer(conf['initial_encoder'])
        self.fc = create_layer(conf['fc_layers'], group_using_sequential=False)
        self.skip_network = create_layer(conf['skip_network']) if 'skip_network' in conf else None

        self._normalize_representations = conf.get('normalize_representations', False)

        self.add_classif_net = conf.get('classification_network', False)
        if self.add_classif_net:
            self.classification_network = create_layer(conf['classification_network'])
        self.final_layer = create_layer(conf['final_layer'])

        input1 = tf.keras.Input(shape=(1,))
        input2 = tf.keras.Input(shape=(1,))
        self.inputs = [input1, input2]
        self.outputs = self(self.inputs)

        # self.summary()

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: pairs of indices; we pass all nodes through the GNN layers, but we
        split in batches only the links (not very efficient)
        :param training:
        :param mask:
        :return:
        """
        features = self.call_custom_graph(inputs, self.nodes_features, self.adj_matrix_train, training, mask)

        return self.final_layer(features, training=training)

    def call_custom_graph(self, inputs, nodes_features, adj_matrix, training=None, mask=None):
        emb1, emb2 = self.encoder_custom_graph(inputs, nodes_features, adj_matrix, training)

        self.merged = self.feature_fusion_layer([emb1, emb2])

        if self.add_classif_net:
            # prediction = obtain_output(self.merged, self.classification_network, training=training)
            prediction = self.classification_network(self.merged, training=training)
        else:
            prediction = self.merged

        return prediction

    def encoder_custom_graph(self, inputs, nodes_features, adj_matrix, training):
        index1, index2 = inputs
        # obtain graph embeddings
        embeddings = self.initial_encoder(nodes_features, training=training)

        for i in range(self._num_encoder_layers):
            embeddings = self.encoder_layers[i]((embeddings, adj_matrix), training=training)

        emb1 = get_indices_from_batch(index1, embeddings)
        emb2 = get_indices_from_batch(index2, embeddings)

        if self.skip_network:
            node1 = get_indices_from_batch(inputs[0], nodes_features)
            node2 = get_indices_from_batch(inputs[1], nodes_features)

            skip1 = self.skip_network(node1, training=training)
            skip2 = self.skip_network(node2, training=training)

            emb1 = emb1 + skip1
            emb2 = emb2 + skip2

        emb1 = obtain_output(emb1, self.fc, training=training)
        emb2 = obtain_output(emb2, self.fc, training=training)

        return emb1, emb2

    def encoder(self, inputs, training):
        return self.encoder_custom_graph(inputs, self.nodes_features, self.adj_matrix_train, training)

    def predict(self, protein_codes, **kwargs):
        only_unseen = kwargs['only_unseen']
        # if true, loop through batches of 50,000 and make predictions for all of them
        predict_in_batches = kwargs.get('predict_in_batches', False)

        if predict_in_batches:
            batch_size = 50000
            predictions = []
            for i in range(0, len(protein_codes[0]), batch_size):
                batch = (protein_codes[0][i: i + batch_size], protein_codes[1][i: i + batch_size])
                predictions_for_batch = self.get_predictions_for_batch(batch, only_unseen)
                predictions.append(predictions_for_batch)
            graph_prediction = np.concatenate(predictions)

        else:
            graph_prediction = self.get_predictions_for_batch(protein_codes, only_unseen)

        return graph_prediction

    def get_predictions_for_batch(self, protein_codes, only_unseen):
        if only_unseen:
            indices_pairs, nodes_features, adj_matrix = self.compute_nodes_features_adj_matrix_only_unseen_nodes(
                protein_codes, self.features_source
            )
        else:
            indices_pairs, nodes_features, adj_matrix = self.compute_nodes_features_adj_matrix_all_nodes(
                protein_codes, self.features_source
            )

        graph_features = self.call_custom_graph(
            [indices_pairs[:, 0], indices_pairs[:, 1]],
            nodes_features,
            self._adj_matrix_processing_fn(adj_matrix),
            training=False
        )

        graph_prediction = self.final_layer(graph_features)
        return graph_prediction
