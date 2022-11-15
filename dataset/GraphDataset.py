from dataset.FeaturesDataset import FeaturesDataset
import numpy as np
import tensorflow as tf


class GraphDataset(FeaturesDataset):

    def __init__(self, conf_params, split_indices=None, split_files=None):
        super().__init__(conf_params, split_indices, split_files, load_pair_features=False)

        if conf_params['adj_matrix_nodes'] == 'training_nodes':
            # create adjacency matrix using only nodes from the training set and val set
            # the validation nodes are added to the matrix, but their interactions are not
            train_val_interactions = np.concatenate([self.train_interactions, self.val_interactions])
            # sort nodes to be sure the order is the same across different runs (set function is not deterministic)
            # we need the same order when we evaluate again an already trained model
            self.training_nodes = sorted(list(set([p for protein_pair in train_val_interactions
                                                   for p in [protein_pair[self._protein1_col], protein_pair[self._protein2_col]]])))
        elif conf_params['adj_matrix_nodes'] == 'all_nodes_interactions':
            # create adjacency matrix using all nodes in the dataset...(not used anymore)
            self.training_nodes = sorted(list(set([p for protein_pair in self.protein_interactions
                                                   for p in [protein_pair[self._protein1_col], protein_pair[self._protein2_col]]])))

        elif conf_params['adj_matrix_nodes'] == 'all_sequences':
            self.training_nodes = sorted(self.protein_features.keys())
        else:
            raise ValueError('invalid adjacency matrix construction method!')

        # mapping from protein codes (or ids, as they appear in the data set interaction files) to their index in the
        # adj matrix and nodes features array (needed in all graph models)
        self.id_to_index = {self.training_nodes[i]: i for i in range(len(self.training_nodes))}
        self.num_nodes = len(self.training_nodes)
        self.adj_matrix_train = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        self.num_edges = 0
        self.sparse_adj_matrix = conf_params.get('sparse_adj_matrix', False)

        self.build_binary_adj_matrix()

        if conf_params['node_features'] == 'protein_features':
            # use node features
            self.nodes_features = np.array([self.protein_features[p] for p in self.training_nodes])
        elif conf_params['node_features'] == 'onehot':
            # feature-less => use one hot encoding of nodes as features
            self.nodes_features = np.eye(len(self.training_nodes)).astype(np.float32)
        else:
            # we don't use the nodes features from this class, but some other array computed in
            # some other class
            self.nodes_features = None

        self.create_tf_datasets()

    def build_binary_adj_matrix(self):
        # adjacency matrix contains only training interactions
        for row in self.train_interactions:
            label = self._label_map[row[self._label_col]]

            if label > 0:
                index1 = self.id_to_index[row[self._protein1_col]]
                index2 = self.id_to_index[row[self._protein2_col]]
                self.adj_matrix_train[index1][index2] = self.adj_matrix_train[index2][index1] = 1
                self.num_edges += 1

    def create_tf_datasets(self):
        self.train_dataset = tf.data.Dataset.from_tensors(
            (self.nodes_features, self.adj_matrix_train))
        self.validation_dataset = None

    def load_data_from_indices(self, indices):
        """
        For this dataset, the "data" to be loaded is represented by pairs of node indices from the adjacency matrix
        """
        selected_interactions = [self.get_proteins_and_label(p_tuple) for p_tuple in self.protein_interactions[indices]]
        return self.compute_nodes_indices_and_labels(selected_interactions)

    def load_features_from_indices(self, indices):
        return super(GraphDataset, self).load_data_from_indices(indices)

    def load_data_from_file(self, file):
        selected_interactions = self.load_interactions_and_labels_from_file(file)
        return self.compute_nodes_indices_and_labels(selected_interactions)

    def compute_nodes_indices_and_labels(self, selected_interactions):
        # we map the protein codes to their index in the adjacency matrix
        nodes_indices_and_labels = np.array([[self.id_to_index[p1], self.id_to_index[p2], int(label)]
                                             for p1, p2, label in selected_interactions
                                             if p1 in self.training_nodes and p2 in self.training_nodes])

        return (nodes_indices_and_labels[:, 0], nodes_indices_and_labels[:, 1]), nodes_indices_and_labels[:, 2]
