import numpy as np
import os
import pdb
import tensorflow as tf
import csv

from sklearn.model_selection import KFold, StratifiedKFold

from dataset.MultiViewDataset import MultiViewDataset
from eval_utils import create_params_for_class, compute_performance_metrics, compute_best_threshold, \
    load_shuffled_indexes
from main_ensemble import SimpleEnsemble, StackedModel
from utils import read_conf, load_model, load_weights, append_row, parse_arguments
from sklearn.linear_model import LogisticRegression


def run_ensemble_k_fold(args):
    # read config file and split in model/training/run configs
    conf_file_path = args.dataset
    dataset_conf_dict, model_conf_dict, training_conf_dict, run_conf_dict, entire_conf = read_conf(conf_file_path)

    shuffled_indexes = load_shuffled_indexes(dataset_conf_dict['data_dir'], tag=dataset_conf_dict.get("tag", ""))

    num_splits = int(args.num_folds)
    kf = KFold(n_splits=num_splits)

    fold_start = int(args.fold_start) - 1
    fold_end = num_splits

    fold_no = fold_start
    splits = list(kf.split(shuffled_indexes))

    header = True
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    for train_shuffle_indices, test_shuffle_indices in splits[fold_start: fold_end]:

        fold_no += 1

        train_indices = shuffled_indexes[train_shuffle_indices]
        test_indices = shuffled_indexes[test_shuffle_indices]

        print("Fold {}".format(fold_no))

        # create dataset
        dataset = MultiViewDataset(dataset_conf_dict, split_indices=train_indices)

        # set folder for saving sequence model
        run_conf_dict['protein_model']['dir_name'] = run_conf_dict['dir_name'] + '/sequence/'

        seq_training_conf, seq_model_conf, seq_run_conf = create_params_for_class(
            training_conf_dict['protein_model'],
            model_conf_dict['protein_model'],
            run_conf_dict['protein_model'],
            entire_conf,
            fold_no, args.train
        )

        sequence_model = load_model(seq_model_conf['name'], seq_model_conf['kwargs'], dataset)
        sequence_model.compile(**seq_training_conf)

        run_conf_dict['graph_model']['dir_name'] = run_conf_dict['dir_name'] + '/graph/'
        graph_training_conf, graph_model_conf, graph_run_conf = create_params_for_class(
            training_conf_dict['graph_model'],
            model_conf_dict['graph_model'],
            run_conf_dict['graph_model'],
            entire_conf,
            fold_no, args.train
        )

        graph_model = load_model(graph_model_conf['name'], graph_model_conf['kwargs'], dataset)
        graph_model.compile(**graph_training_conf)

        if args.train:

            sequence_model.fit(
                dataset.train_dataset,
                y=dataset.target_train,
                batch_size=dataset_conf_dict['batch_size_train'],
                epochs=seq_run_conf['epochs'],
                callbacks=seq_run_conf['callbacks'],
                validation_data=dataset.validation_dataset
            )

            graph_model.fit(
                dataset.train_dataset,
                y=dataset.target_train,
                batch_size=dataset_conf_dict['batch_size_train'],
                epochs=graph_run_conf['epochs'],
                callbacks=graph_run_conf['callbacks'],
                validation_data=dataset.validation_dataset
            )

        else:
            load_weights(seq_run_conf['dir_name'] + '/ckpt', sequence_model)
            load_weights(graph_run_conf['dir_name'] + '/ckpt', graph_model)

        val_codes = [[v[0] for v in dataset.val_interactions],
                     [v[1] for v in dataset.val_interactions]]
        val_labels = np.argmax(dataset.validation_dataset[1], axis=1)

        if model_conf_dict['ensemble_type'] == 'stack':
            ml_clf = LogisticRegression()
            ensemble_model = StackedModel(sequence_model, graph_model, ml_clf)
            ensemble_model.train_ml_model(val_codes, val_labels)

        else:
            ensemble_model = SimpleEnsemble(sequence_model, graph_model)

        predicted_scores_val, seq_scores_val, graph_scores_val = ensemble_model.predict(
            val_codes, only_unseen=False, predict_in_batches=False)

        best_th_ens = compute_best_threshold(thresholds, predicted_scores_val, val_labels, onehot=True)
        best_th_seq = compute_best_threshold(thresholds, seq_scores_val, val_labels, onehot=True)
        best_th_graph = compute_best_threshold(thresholds, graph_scores_val, val_labels, onehot=True)

        # load protein codes and labels
        test_interactions = dataset.protein_interactions[test_indices]
        test_data = [test_interactions[:, 0], test_interactions[:, 1]]
        test_labels = test_interactions[:, 2].astype(np.int32)

        predicted_scores, seq_scores, graph_scores = ensemble_model.predict(
            test_data, only_unseen=False,
            predict_in_batches=False
        )

        # compute results for ensemble model
        predicted_labels_ens = tf.cast(predicted_scores[:, 1] > best_th_ens, tf.int32)

        results, columns = compute_performance_metrics(
            test_labels,
            [predicted_labels_ens, predicted_scores[:, 1]],
            dataset.num_classes
        )

        # results for sequence model
        predicted_labels_seq = tf.cast(seq_scores[:, 1] > best_th_seq, tf.int32)
        results_seq, _ = compute_performance_metrics(
            test_labels,
            [predicted_labels_seq, seq_scores[:, 1]],
            dataset.num_classes
        )

        # results for graph model
        predicted_labels_graph = tf.cast(graph_scores[:, 1] > best_th_graph, tf.int32)
        results_graph, columns = compute_performance_metrics(
            test_labels,
            [predicted_labels_graph, graph_scores[:, 1]],
            dataset.num_classes
        )

        print("Results:", {metric: res for (metric, res) in zip(columns[1:], results)})

        fold_tag = "{}".format(fold_no)
        # save results
        append_row(
            run_conf_dict['dir_name'] + '/results.csv',
            [fold_tag] + results, header, columns
        )

        append_row(
            run_conf_dict['dir_name'] + '/results_seq.csv',
            [fold_tag] + results_seq, header, columns
        )

        append_row(
            run_conf_dict['dir_name'] + '/results_graph.csv',
            [fold_tag] + results_graph, header, columns
        )

        del test_data
        del test_labels

        del dataset.adj_matrix_train
        del dataset.protein_features
        del dataset

        tf.keras.backend.clear_session()

        # in next set of experiments we start from the first fold
        header = False


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    tf.random.set_seed(0)

    args = parse_arguments()

    run_ensemble_k_fold(args)
