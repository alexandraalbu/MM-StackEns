from dataset.Dataset import Dataset
import pickle as pkl
import os
import numpy as np
import pdb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from dataset.normalization import MinMaxStdNormalization, normalize_pair


def _process_df(df):
    # replace "Positive" keyword with 1, "Negative" with 0
    df.replace(to_replace="Positive", value=1, inplace=True)
    df.replace(to_replace="Negative", value=0, inplace=True)

    # rearrange columns to be "protein1", "protein2", "interaction"
    return df.reindex(columns=[1, 2, 0]).to_numpy()


def load_h5(data_dir, feature_type):
    import h5py
    f = h5py.File(data_dir + "/{}/stage_0/reduced_embeddings_file.h5".format(feature_type), 'r')
    df = pd.read_csv(data_dir + "/{}/mapping_file.csv".format(feature_type))
    mapping = dict(zip(df['Unnamed: 0'].astype(str), df['original_id']))
    embeddings = {mapping[hash_val]: emb for hash_val, emb in f.items()}
    return embeddings


def merge_dictionaries(list_of_dictionaries):
    """
    Merge multiple dictionaries by concatenating the arrays corresponding to each key
    Preconditions: all dictionaries must have the same keys and
                the numpy arrays corresponding to a key must be of same shape on the 0 axis
    :param list_of_dictionaries
    :return: a dictionary in which the arrays corresponding to each key are concatenated
    """
    final_dict = {}
    for key in list_of_dictionaries[0].keys():
        final_dict[key] = np.concatenate(list(d[key] for d in list_of_dictionaries))
    return final_dict


def split_in_codes_and_labels(interactions):
    return (interactions[:, 0], interactions[:, 1]), interactions[:, 2].astype(np.int32)


def remove_interactions_of_protein(interactions, protein_code):
    """
    Returns the list of interactions in which the protein with ID "protein_code" appears
    Removes from the list "interactions" all these interactions
    """
    current_protein_interactions = []
    i = 0
    while i < len(interactions):
        triple = interactions[i]
        if protein_code in triple[:-1]:
            current_protein_interactions.append(triple)
            interactions.pop(i)
        else:
            i += 1
    return current_protein_interactions


class FeaturesDataset(Dataset):

    feature_combination_methods = {
        'concatenation': lambda x, y: np.concatenate([x, y], axis=-1),
        'hadamard': lambda x, y: np.multiply(x, y),
        'interaction_matrix': lambda x, y: np.multiply(x, y.T),
        'abs-diff': lambda x, y: np.abs(x - y),
        'abs-diff_hadamard': lambda x, y: np.concatenate([np.abs(x - y), np.multiply(x, y)], axis=-1),
        'tuple': lambda x, y: (x, y),
    }

    scalers = {
        'min_max': MinMaxScaler(),
        'std': StandardScaler(),
        'min_max_std': MinMaxStdNormalization(),
        'robust': RobustScaler(),
    }

    def __init__(self, conf_params, split_indices=None, split_files=None, load_pair_features=True):

        super().__init__(conf_params)

        self.data_dir = self._params['data_dir']
        self._duplicate_samples = self._params.get('double_samples', False)
        self._feature_comb_method = self._params.get('feature_combination', 'tuple')
        self._feature_type = self._params.get('feature_type', 'ct')
        self._max_length = self._params.get('max_length', 2000)
        self._normalize = self._params.get('normalize', False)
        self._normalize_seq = self._params.get('normalize_seq', False)
        self._features_indices = self._params.get('features_indices', None)
        self.onehot = self._params['onehot']
        # for the Yeast and Human data sets of Park and Marcotte, the label is the first column and it needs to be moved
        self.move_columns = conf_params['organism_name'] in ['yeast', 'human']

        self._pair_features_path = self._params.get('pair_features_path', None)
        self._tag = self._params.get("tag", "")
        self._random_val = self._params.get('random_val', True)

        self.protein_interactions = None
        self.protein_features = None
        self.protein_lengths = {}

        self._header = conf_params.get('header', None)
        self._set_dataset_params()

        assert self._feature_comb_method in FeaturesDataset.feature_combination_methods.keys() \
               or self._feature_comb_method is None
        assert self._normalize is False or self._normalize in FeaturesDataset.scalers.keys()

        if split_indices is not None:
            self.train_interactions, self.val_interactions = self.load_interactions_from_indices(split_indices)
        elif split_files is not None:
            self.train_interactions, self.val_interactions = self.load_interactions_from_files(split_files)
        else:
            raise ValueError("Invalid params! You have to provide either split_indices or split_files")

        self.protein_features = self._load_protein_features()

        if load_pair_features:

            self.x_train, self.y_train = self.get_pair_features(self.train_interactions, self.protein_features,
                                                                duplicate=self._duplicate_samples)
            self.x_val, self.y_val = self.get_pair_features(self.val_interactions, self.protein_features,
                                                            duplicate=False)

            if 'load_embeddings' in self._params:
                protein_embeddings = self._load_precomputed_features('elmo_embeddings')
                self.training_embeddings, _ = self.get_pair_features(self.train_interactions, protein_embeddings,
                                                                     duplicate=self._duplicate_samples)
                self.val_embeddings, _ = self.get_pair_features(self.val_interactions, protein_embeddings,
                                                                duplicate=False)

            if self._normalize:
                self.x_train, self.x_val = self._normalize_fct(self.x_train, self.x_val)

            if self.onehot:
                self.target_train, self.target_val = self.onehot_encode_labels()
            else:
                self.target_train, self.target_val = self.y_train, self.y_val

    def onehot_encode_labels(self):
        self.y_train = self.y_train.reshape((-1, 1))
        self.y_val = self.y_val.reshape((-1, 1))
        onehotencoder = OneHotEncoder(sparse=False)
        target_train = onehotencoder.fit_transform(self.y_train)
        target_val = onehotencoder.transform(self.y_val)
        return target_train, target_val

    def _set_dataset_params(self):
        self._ds_name = self.data_dir.split("/")[-1]
        if self._ds_name == 'STRING':
            if self._tag != 'full':
                self._protein1_col = 2
                self._protein2_col = 3
                self._label_col = 4
            else:
                self._protein1_col = 0
                self._protein2_col = 1
                self._label_col = 2

            self.label_names = ['reaction', 'binding', 'ptmod', 'activation', 'inhibition', 'catalysis', 'expression']
            self._label_map = {label: i for i, label in enumerate(self.label_names)}
            self.num_classes = 7
        else:
            self._protein1_col = 0
            self._protein2_col = 1
            self._label_col = 2
            self._label_map = {
                0: 0,
                1: 1
            }
            self.label_names = ["non-interaction", "interaction"]
            self.num_classes = 2

    def get_proteins_and_label(self, protein_tuple):
        return protein_tuple[self._protein1_col], protein_tuple[self._protein2_col], self._label_map[
            protein_tuple[self._label_col]]

    def load_interactions_from_indices(self, train_indices):
        self.protein_interactions = pd.read_csv(self.data_dir + '/protein_interactions{}.tsv'.format(self._tag),
                                                header=self._header, delimiter='\t').to_numpy()

        full_train_interactions = self.protein_interactions[train_indices]

        if self._random_val:
            split_point = int(0.9 * train_indices.shape[0])
            # we assume indices are already shuffled at thus point (we used fixed permutation files)
            train_interactions = full_train_interactions[:split_point]
            val_interactions = full_train_interactions[split_point:]

        else:
            train_interactions, val_interactions = self.create_validation_set_including_unseen_proteins(
                full_train_interactions
            )

        return train_interactions, val_interactions

    def load_interactions_from_files(self, train_file):
        """
        Load train interactions from a file - used for data sets with fixed (given) splits

        :param train_file: path of the file containing the interactions and their labels
        :return: tuple of: train interactions and validation interactions, each of them being an array of shape
        (n_interactions, 3)
        """

        full_train_interactions = self.load_interactions_and_labels_from_file(train_file)

        if self._random_val:
            # we randomly select 90% of the data in the training set and 10% in the validation set
            # (the dataset may be ordered, so we need to shuffle)
            train_interactions, val_interactions = train_test_split(full_train_interactions, train_size=0.9, random_state=0)

        else:
            train_interactions, val_interactions = self.create_validation_set_including_unseen_proteins(
                full_train_interactions
            )

        return train_interactions, val_interactions

    def create_validation_set_including_unseen_proteins(self, interactions):
        interactions = interactions.tolist()
        protein_set = sorted(
            list(set([p for protein_pair in interactions
                      for p in [protein_pair[self._protein1_col], protein_pair[self._protein2_col]]]))
        )

        # we select a validation set of ~10% interactions, out of which half (5%) are formed of
        # unseen proteins and the other half are randomly sampled
        num_unseen = int(0.05 * len(interactions))
        unseen = []

        # in order to create the pairs of unseen proteins, we randomly
        # select multiple proteins, which we exclude from the training set
        potentially_excluded_proteins = np.random.default_rng(0).choice(
            protein_set, size=int(0.05*(len(protein_set))), replace=False
        )

        # we loop through the set of proteins and keep excluding the proteins
        # and all their interactions from the training set until we reach 5% interactions
        # disclaimer: this is not an efficient implementation, but for our use case it's ok
        i = 0
        really_excluded = []
        while len(unseen) < num_unseen:
            prot = potentially_excluded_proteins[i]
            interactions_for_current = remove_interactions_of_protein(interactions, prot)
            # print("Removing {} for protein {}".format(len(interactions_for_current), prot))
            unseen.extend(interactions_for_current)
            really_excluded.append(prot)
            i += 1

        # the rest of the interactions are selected randomly from the remaining interactions
        train_interactions, randomly_chosen = train_test_split(interactions, test_size=num_unseen, random_state=0)

        return train_interactions, unseen + randomly_chosen

    def load_interactions_and_labels_from_file(self, file, split_codes_and_labels=False):
        """
        Load the interactions and labels from a file, in two possible formats
        :param file: the name of the file containing the current file to be read
        :param split_codes_and_labels: boolean
        :return:
        - if split_codes_and_labels is False: array of shape (n_interactions, 3) containing for each interaction:
        protein1_code, protein2_code, label
        - else: the tuple (interactions, labels), where interactions is a tuple containing the array of proteins on
        the first position and the array of proteins on the second position
        """
        df_train = pd.read_csv(self.data_dir + "/interactions/" + file, sep='\t', header=None)

        if self.move_columns:
            # 2012 datasets (created by Park and Marcotte) need to be preprocessed
            # to be in the form (protein1_code, protein2_code, label)
            interactions = _process_df(df_train)
        else:
            # Human 2021 dataset
            interactions = df_train.to_numpy()

        if split_codes_and_labels:
            return split_in_codes_and_labels(interactions)

        return interactions

    def _normalize_fct(self, train_data, val_data):
        self.scaler = FeaturesDataset.scalers[self._normalize]

        if self._feature_comb_method == 'tuple':
            data_to_fit = np.concatenate([train_data[:, 0], train_data[:, 1]], axis=0)
        else:
            data_to_fit = train_data

        self.scaler.fit(data_to_fit)

        return normalize_pair(self.scaler, train_data), normalize_pair(self.scaler, val_data)

    def get_pair_features(self, codes_pairs, protein_features, duplicate=False):
        pair_features = []
        labels = []

        for tuple in codes_pairs:
            features1 = protein_features[tuple[self._protein1_col]]
            features2 = protein_features[tuple[self._protein2_col]]
            label = self._label_map[tuple[self._label_col]]
            labels.append(label)

            pair_features.append(
                FeaturesDataset.feature_combination_methods[self._feature_comb_method](features1, features2))

            if duplicate:
                pair_features.append(
                    FeaturesDataset.feature_combination_methods[self._feature_comb_method](features2, features1))
                labels.append(label)

        return np.stack(pair_features).astype(np.float32), np.array(labels)

    def _load_precomputed_features(self, feature_type):
        if feature_type == 'elmo_embeddings':
            protein_features = self._load_npz_features(feature_type)
        elif feature_type in ['elmo']:
            protein_features = self._load_h5(feature_type)
        else:
            with open(self.data_dir + '/{}_features.pkl'.format(feature_type), "rb") as f:
                protein_features = pkl.load(f)

        return protein_features

    def _load_npz_features(self, feature_type):
        return np.load(self.data_dir + "/{}.npz".format(feature_type))

    def _load_h5(self, feature_type):
        return load_h5(self.data_dir, feature_type)

    def _load_protein_features(self):
        if type(self._feature_type) is list:

            protein_features = merge_dictionaries([self._load_precomputed_features(feature_type)
                                                   for feature_type in self._feature_type])
        else:
            protein_features = self._load_precomputed_features(self._feature_type)

        return protein_features

    def normalize_data(self, data):
        return self.scaler.transform(data)

    def load_data_from_indices(self, indices):
        """
        Load any interactions from the self.protein_interactions array, given the indices (rows) of the interactions
        """

        selected_interactions = self.protein_interactions[indices]

        if self._feature_type == 'precomputed_pair':
            pair_features = np.load(self._pair_features_path)
            features = pair_features[indices]
            labels = selected_interactions[:, 2].astype(np.int)

        else:
            features, labels = self.get_pair_features(selected_interactions, self.protein_features)

        if self._normalize:
            features = normalize_pair(self.scaler, features)

        if self._feature_comb_method == 'tuple':
            features = [features[:, 0], features[:, 1]]

        return features, labels

    def load_data_from_file(self, file):
        selected_interactions = self.load_interactions_and_labels_from_file(file)
        features, labels = self.get_pair_features(selected_interactions, self.protein_features)

        if self._normalize:
            features = normalize_pair(self.scaler, features)

        if self._feature_comb_method == 'tuple':
            features = [features[:, 0], features[:, 1]]

        return features, labels

    @property
    def input_shape(self):
        protein_dict = self.protein_features
        return protein_dict[list(protein_dict.keys())[0]].shape


