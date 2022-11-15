from copy import copy

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model


class AbstractGraphModel(Model):

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = conf['dataset']

    def compute_nodes_features_adj_matrix_all_nodes(self, protein_codes, graph_features_source):
        indices_pairs = []
        num_pairs = len(protein_codes[0])

        adj_matrix = self.adj_matrix_train
        if graph_features_source == 'graph_nodes':
            nodes_features = self.dataset.nodes_features
            features_dict = self.dataset.protein_features
        elif graph_features_source == 'main_features':
            nodes_features = self.dataset.main_features
            features_dict = self.dataset.protein_features
        else:
            nodes_features = self.dataset.extra_features
            features_dict = self.dataset.extra_features_dict

        index = adj_matrix.shape[0]
        new_features = []
        initial_num_proteins = index

        # we start with the training proteins and add the others
        id_to_index = copy(self.dataset.id_to_index)

        for i in range(num_pairs):
            p1, p2 = protein_codes[0][i], protein_codes[1][i]

            if p1 in id_to_index.keys():
                # p1 is in the current collection
                index1 = id_to_index[p1]
            else:
                # next available index
                index1 = index
                index += 1

                nf = features_dict[p1]
                new_features.append(nf)

            if p2 in id_to_index.keys():
                index2 = id_to_index[p2]
            else:
                # next available index
                index2 = index
                index += 1

                nf = features_dict[p2]
                new_features.append(nf)

            indices_pairs.append([index1, index2])

        num_new_rows = index - initial_num_proteins

        if num_new_rows > 0:

            if self.dataset.sparse_adj_matrix:
                adj_matrix = tf.sparse.to_dense(adj_matrix)

            # here the matrix is not sparse
            adj_matrix = np.pad(adj_matrix, (0, num_new_rows))

            # make the matrix sparse again
            if self.dataset.sparse_adj_matrix:
                adj_matrix = tf.sparse.from_dense(adj_matrix)

            nodes_features = np.concatenate([nodes_features, np.array(new_features)])

        indices_pairs = np.array(indices_pairs)

        return indices_pairs, tf.convert_to_tensor(nodes_features), adj_matrix

    def compute_nodes_features_adj_matrix_only_unseen_nodes(self, protein_codes, graph_features_source):
        """
        This is used to optimize memory for HeldOut data sets - we know that no protein is in the
        training graph and we build an empty adjacency matrix for these nodes since they will not have any
        known connections
        """
        indices_pairs = []
        num_pairs = len(protein_codes[0])

        if graph_features_source == 'graph_nodes':
            features_dict = self.dataset.protein_features
        elif graph_features_source == 'main_features':
            features_dict = self.dataset.protein_features
        else:
            features_dict = self.dataset.extra_features_dict

        index = 0
        new_features = []
        id_to_index = {}

        for i in range(num_pairs):
            p1, p2 = protein_codes[0][i], protein_codes[1][i]

            if p1 in id_to_index.keys():
                # p1 is in the current collection
                index1 = id_to_index[p1]
            else:
                # next available index
                index1 = index
                index += 1

                nf = features_dict[p1]
                new_features.append(nf)

            if p2 in id_to_index.keys():
                index2 = id_to_index[p2]
            else:
                # next available index
                index2 = index
                index += 1

                nf = features_dict[p2]
                new_features.append(nf)

            indices_pairs.append([index1, index2])

        num_new_rows = index

        if self.dataset.sparse_adj_matrix:
            adj_matrix = tf.SparseTensor(
                indices=np.empty((0, 2), dtype=np.int64), values=[], dense_shape=(num_new_rows, num_new_rows)
            )
        else:
            adj_matrix = tf.zeros((num_new_rows, num_new_rows), dtype=np.float32)

        nodes_features = np.array(new_features)

        indices_pairs = np.array(indices_pairs)

        return indices_pairs, tf.convert_to_tensor(nodes_features), adj_matrix
