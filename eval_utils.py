import os
import pdb
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from utils import load_model, create_experiment_directory, save_conf_file, process_confs, compute_stats_multiple_keys
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, \
    f1_score, precision_recall_curve, auc, average_precision_score


def _generate_shuffle_file(file_name, n_points):
    permutation = np.random.permutation(n_points)
    np.savetxt(file_name, permutation, "%d")


def load_shuffled_indexes(data_dir, tag="", header=None):
    shuffle_file = data_dir + '/shuffled_indices{}.txt'.format(tag)
    # if file with shuffled indexes does not exist, create it
    if not os.path.exists(shuffle_file):
        protein_interactions = pd.read_csv(data_dir + '/protein_interactions{}.tsv'.format(tag),
                                           header=header, delimiter='\t').to_numpy()
        _generate_shuffle_file(shuffle_file, protein_interactions.shape[0])

    indexes = np.loadtxt(shuffle_file, dtype=np.int)
    return indexes


def create_paired_input_and_output_nodes(dataset):
    training_pairs = dataset.x_train
    val_pairs = dataset.x_val

    # set datasets for training
    dataset.train_dataset = [training_pairs[:, 0], training_pairs[:, 1]]
    dataset.target_train = [training_pairs[:, 0], training_pairs[:, 1]]
    dataset.validation_dataset = ([val_pairs[:, 0], val_pairs[:, 1]], [val_pairs[:, 0], val_pairs[:, 1]])


def train_model(dataset, model_conf, training_conf, run_conf, onehot_labels):
    # load model
    model = load_model(model_conf['name'], model_conf['kwargs'], dataset)

    model.compile(**training_conf)

    if run_conf.get('summary', False):
        model.summary()

    if isinstance(dataset.train_dataset, tf.data.Dataset) or isinstance(dataset.train_dataset, tf.keras.utils.Sequence):

        model.fit(dataset.train_dataset,
                  batch_size=dataset.batch_size_train,  # in case dataset is a tf dataset, this is not used
                  epochs=run_conf['epochs'],
                  validation_data=dataset.validation_dataset,
                  callbacks=run_conf['callbacks'])
    else:

        if hasattr(dataset, "compute_sample_weights"):
            sample_weights = dataset.sample_weights
        else:
            sample_weights = None

        model.fit(dataset.train_dataset,
                  y=dataset.target_train if onehot_labels else dataset.y_train,
                  batch_size=dataset.batch_size_train,
                  epochs=run_conf['epochs'],
                  validation_data=dataset.validation_dataset,
                  callbacks=run_conf['callbacks'],
                  sample_weight=sample_weights)
    return model


class_encodings = {
    'positive': 1,  # positive class corresponds to interaction, encoded as 1
    'negative': 0  # negative class corresponds to non-interaction, encoded as 0
}


def create_params_for_class(training_conf, model_conf, run_conf, full_str_conf,
                            fold_no, train):
    experiment_dir = run_conf['dir_name'] + 'Fold_{}/'.format(fold_no)

    if train:
        # create directory for the experiment
        create_experiment_directory(experiment_dir)

        # update conf with path where the model is saved
        updated_full_conf = deepcopy(full_str_conf)
        updated_full_conf['run']['dir_name'] = experiment_dir

        if fold_no == 1 or fold_no == "11" or "0_0" in str(fold_no):
            # save conf file in the experiment's folder
            save_conf_file(run_conf['dir_name'], updated_full_conf)

    r_conf = deepcopy(run_conf)
    r_conf['dir_name'] = experiment_dir

    tr_conf, r_conf = process_confs(training_conf, r_conf)

    m_conf = deepcopy(model_conf)

    return tr_conf, m_conf, r_conf


def compute_performance_metrics(labels, predictions, num_classes):
    if num_classes > 2:
        # for multiclass we compute metrics using predicted labels
        return compute_performance_metrics_multiclass(labels, predictions[0])
    else:
        # extract score of positive class
        return compute_performance_metrics_binary(labels, *predictions)


def aupr_score(test_labels, predicted_scores_positive_class):
    precision_list, recall_list, thresholds = precision_recall_curve(test_labels, predicted_scores_positive_class)
    return auc(recall_list, precision_list)


def compute_performance_metrics_binary(test_labels, predicted_labels, predicted_scores_positive_class):
    # compute metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    # specificity is equivalent to recall when the positive label is 0
    specificity = recall_score(test_labels, predicted_labels, pos_label=0)
    f1 = f1_score(test_labels, predicted_labels)
    roc_auc = roc_auc_score(test_labels, predicted_scores_positive_class)
    # average precision
    avgpr = average_precision_score(test_labels, predicted_scores_positive_class)

    return [accuracy, precision, recall, specificity, f1, roc_auc, avgpr], \
           ['Fold', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1', 'ROC-AUC', 'AUPR(AVG-P)']


def check_onehot(x, t):
    return x[:, 1] > t


def check_binary(x, t):
    return x > t


def compute_best_threshold(thresholds, scores, labels, onehot=False):
    max_f1 = 0
    best_threshold = None

    if onehot:
        condition = check_onehot
    else:
        condition = check_binary

    for t in thresholds:
        predictions = tf.cast(condition(scores, t), tf.int32)

        f1 = f1_score(labels, predictions)
        if f1 > max_f1:
            best_threshold = t
            max_f1 = f1

    return best_threshold


def compute_performance_metrics_multiclass(test_labels, predicted_labels):
    accuracy = accuracy_score(test_labels, predicted_labels)
    micro_f1 = f1_score(test_labels, predicted_labels, average='macro')
    f1s_per_class = f1_score(test_labels, predicted_labels, average=None)
    return [accuracy, micro_f1] + list(f1s_per_class), \
           ['Fold', 'Accuracy', 'macro F1'] + ["F1 class {}".format(i) for i in range(len(f1s_per_class))]


def save_data(samples, indices, dir_name, tag='test'):
    np.save(dir_name + '/{}_data.npy'.format(tag), samples)
    np.savetxt(dir_name + '/{}_indices.txt'.format(tag), indices, fmt="%d")


