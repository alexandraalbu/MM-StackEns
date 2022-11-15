from dataset.FeaturesDataset import merge_dictionaries, FeaturesDataset
from dataset.GraphDataset import GraphDataset
import numpy as np


class MultiViewDataset(GraphDataset):

    def __init__(self, conf_params, split_indices=None, split_files=None):
        super().__init__(conf_params, split_indices, split_files)

        self.main_feature_scaler = None
        self.extra_features_scaler = None

        if conf_params.get('create_features_array', False):
            self.main_features = None
            self.create_features_array()

        self._normalize_extra_features = conf_params.get('normalize_features')

        if conf_params.get('create_extra_features_array', False):
            self.extra_features_dict = {}
            self.extra_features = None
            self.create_additional_features_array(conf_params['create_extra_features_array'])

    def create_tf_datasets(self):
        """
        Create training and validation data sets:
        triples of (index in adj matrix for protein1, index in adj matrix for protein2, interaction label)
        """
        train_indices, self.y_train = self.get_pair_features(self.train_interactions, self.id_to_index,
                                                             duplicate=self._duplicate_samples)
        val_indices, self.y_val = self.get_pair_features(self.val_interactions, self.id_to_index)

        if self.onehot:
            self.target_train, self.target_val = self.onehot_encode_labels()
        else:
            self.target_train, self.target_val = self.y_train, self.y_val

        self.train_dataset = [train_indices[:, 0], train_indices[:, 1]]
        self.validation_dataset = ([val_indices[:, 0], val_indices[:, 1]], self.target_val)
        assert len(self.target_train) == len(self.train_dataset[0])

    def load_data_from_indices(self, indices):
        """
        Load the pairs of protein codes using the interaction indices (used only in datasets for which we create
        the splits)
        :param indices: interaction indices
        :return: tuple (interactions, labels)
        """
        selected_interactions = self.protein_interactions[indices]
        labels = selected_interactions[:, self._label_col]

        if self._ds_name == 'STRING':
            labels = np.array([self._label_map[v] for v in labels])

        return (selected_interactions[:, self._protein1_col],
                selected_interactions[:, self._protein2_col]), labels.astype(np.int32)

    def load_data_from_file(self, file):
        data, labels = self.load_interactions_and_labels_from_file(
            file,
            split_codes_and_labels=True
        )

        return data, labels

    def load_extra_features(self, codes_pairs):
        """
        Load features for the proteins pairs with codes given by codes_pairs
        All proteins in the dataset are loaded in the extra_features_dict dictionary, not just the training ones
        """
        feats1 = np.array([self.extra_features_dict[p] for p in codes_pairs[0]])
        feats2 = np.array([self.extra_features_dict[p] for p in codes_pairs[1]])
        # we don't need to normalize here because if this option was chosen,
        # the features (stored in the self.extra_features_dict dictionary) have already been normalized
        return [feats1, feats2]

    def load_main_features(self, codes_pairs):
        seqs1 = np.array([self.protein_features[p] for p in codes_pairs[0]])
        seqs2 = np.array([self.protein_features[p] for p in codes_pairs[1]])
        # we don't need to normalize here because if this option was chosen,
        # the features (stored in the self.protein_features dictionary) have already been normalized
        return [seqs1, seqs2]

    def create_features_array(self):
        """
        Create a numpy array with the main protein features loaded in the base class
        The order of the proteins is the same as in the self.training_nodes list (order is important because
        we extract proteins using indices in graph models)
        :return: the features array
        """
        if self._normalize:
            # self.training_nodes may contain train+validation or all proteins,
            # but for normalization we need to use only training proteins
            # we select them here:
            train_codes = sorted(list(set([p for protein_pair in self.train_interactions
                                           for p in
                                           [protein_pair[self._protein1_col], protein_pair[self._protein2_col]]])))
            train_features = np.array([self.protein_features[p] for p in train_codes])
            self.main_feature_scaler = self.create_and_fit_scaler(train_features, self._normalize)

            # normalize all proteins using stats computed using the training proteins
            # we normalize directly the features in self.protein_features because all functions take the data from there
            for p in self.protein_features.keys():
                old = self.protein_features[p]
                self.protein_features[p] = self.main_feature_scaler.transform(old.reshape(1, -1)).squeeze()

        self.main_features = np.array([self.protein_features[p] for p in self.training_nodes])

    def create_additional_features_array(self, features_types):
        self.extra_features_dict = merge_dictionaries(
            [self._load_precomputed_features(feature_type) for feature_type in features_types]
        )

        if self._normalize_extra_features:
            train_codes = sorted(list(set([p for protein_pair in self.train_interactions
                                           for p in
                                           [protein_pair[self._protein1_col], protein_pair[self._protein2_col]]])))
            extra_features_tr = np.array([self.extra_features_dict[p] for p in train_codes])
            self.extra_features_scaler = self.create_and_fit_scaler(
                extra_features_tr, self._normalize_extra_features
            )

            # normalize all proteins using values computed using the training proteins
            for p in self.extra_features_dict.keys():
                old = self.extra_features_dict[p]
                self.extra_features_dict[p] = self.extra_features_scaler.transform(old.reshape(1, -1)).squeeze()

        self.extra_features = np.array([self.extra_features_dict[p] for p in self.training_nodes])

    def create_and_fit_scaler(self, data, norm_type):
        scaler = FeaturesDataset.scalers[norm_type]
        scaler.fit(data)
        return scaler
