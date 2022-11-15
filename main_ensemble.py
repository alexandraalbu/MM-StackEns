import numpy as np
import os
import pdb
import tensorflow as tf

from dataset.MultiViewDataset import MultiViewDataset
from eval_utils import create_params_for_class, compute_performance_metrics, compute_best_threshold
from utils import read_conf, parse_arguments_fixed_folds, DatasetParams, compute_stats_multiple_keys, write_stats, \
    load_model, load_weights, append_row
from sklearn.linear_model import LogisticRegression


class StackedModel:

    def __init__(self, seq_model, graph_model, ml_model):
        self.seq_model = seq_model
        self.graph_model = graph_model
        self.ml_model = ml_model

    def train_ml_model(self, data, labels):
        train_features = np.stack([
            self.seq_model.predict(data)[:, 1],
            self.graph_model.predict(data, only_unseen=False)[:, 1]
        ]).T

        self.ml_model.fit(
            train_features,
            labels
        )

    def predict(self, data, only_unseen, predict_in_batches):

        features, predictions_graph, predictions_seq = self.compute_base_models_scores(data, only_unseen, predict_in_batches)

        return self.ml_model.predict_proba(features), predictions_seq, predictions_graph

    def compute_base_models_scores(self, data, only_unseen, predict_in_batches):
        if predict_in_batches:
            batch_size = 50000
        else:
            batch_size = None

        predictions_seq = self.seq_model.predict(data, batch_size=batch_size)
        predictions_graph = self.graph_model.predict(
            data, only_unseen=only_unseen, predict_in_batches=predict_in_batches
        )

        features = np.stack([
            predictions_seq[:, 1],
            predictions_graph[:, 1]
        ]).T

        return features, predictions_graph, predictions_seq


class SimpleEnsemble:
    def __init__(self, seq_model, graph_model):
        self.seq_model = seq_model
        self.graph_model = graph_model

    def predict(self, data, only_unseen, predict_in_batches):
        predictions_seq = self.seq_model.predict(data)
        predictions_graph = self.graph_model.predict(
            data, only_unseen=only_unseen, predict_in_batches=predict_in_batches
        )
        predictions = (predictions_seq + predictions_graph) / 2

        return predictions, predictions_seq, predictions_graph


def run_ensemble(args):
    # read config file and split in model/training/run configs
    conf_file_path = args.dataset

    dataset_conf_dict, model_conf_dict, training_conf_dict, run_conf_dict, entire_conf = read_conf(conf_file_path)

    organism_name = dataset_conf_dict['organism_name']

    params = DatasetParams(organism_name)

    # used if we continue training
    fold_start = args.fold_start
    start_ex = int(args.start_ex) - 1
    fold_pos = params.folds.index(fold_start)
    header = True
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    for k in range(start_ex, len(params.experiments)):
        experiment_type = params.experiments[k]

        for pos in range(fold_pos, len(params.folds)):
            fold_no = params.folds[pos]
            train_file, test_files = params.get_train_test_files_fct(organism_name, experiment_type, fold_no)

            fold_id = experiment_type + "{}".format(fold_no)

            print("Fold " + fold_id)

            # create dataset
            dataset = MultiViewDataset(dataset_conf_dict, split_files=train_file)

            # set folder for saving sequence model
            run_conf_dict['protein_model']['dir_name'] = run_conf_dict['dir_name'] + '/sequence/'

            seq_training_conf, seq_model_conf, seq_run_conf = create_params_for_class(
                training_conf_dict['protein_model'],
                model_conf_dict['protein_model'],
                run_conf_dict['protein_model'],
                entire_conf,
                fold_id, args.train
            )

            sequence_model = load_model(seq_model_conf['name'], seq_model_conf['kwargs'], dataset)
            sequence_model.compile(**seq_training_conf)

            run_conf_dict['graph_model']['dir_name'] = run_conf_dict['dir_name'] + '/graph/'
            graph_training_conf, graph_model_conf, graph_run_conf = create_params_for_class(
                training_conf_dict['graph_model'],
                model_conf_dict['graph_model'],
                run_conf_dict['graph_model'],
                entire_conf,
                fold_id, args.train
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

            # prepare validation data - will be used in Stack ensemble but also in threshold computation
            val_codes = [[v[0] for v in dataset.val_interactions], [v[1] for v in dataset.val_interactions]]
            val_labels = np.argmax(dataset.validation_dataset[1], axis=1)

            if model_conf_dict['ensemble_type'] == 'stack':
                ml_clf = LogisticRegression()
                ensemble_model = StackedModel(sequence_model, graph_model, ml_clf)
                ensemble_model.train_ml_model(val_codes, val_labels)

            else:
                ensemble_model = SimpleEnsemble(sequence_model, graph_model)

            # for large test sets, we predict in batches since the entire set does not fit into memory
            predict_in_batches = experiment_type in ['Random20', 'HeldOut20']

            predicted_scores_val, seq_scores_val, graph_scores_val = ensemble_model.predict(
                val_codes, only_unseen=False, predict_in_batches=False)

            best_th_ens = compute_best_threshold(thresholds, predicted_scores_val, val_labels, onehot=True)
            best_th_seq = compute_best_threshold(thresholds, seq_scores_val, val_labels, onehot=True)
            best_th_graph = compute_best_threshold(thresholds, graph_scores_val, val_labels, onehot=True)

            for key in params.keys_list_fct(experiment_type):
                print("Test set {}".format(key))
                test_file = test_files[key]

                # load protein codes and labels
                test_data, test_labels = dataset.load_interactions_and_labels_from_file(
                    test_file,
                    split_codes_and_labels=True
                )

                predicted_scores, seq_scores, graph_scores = ensemble_model.predict(
                    test_data, run_conf_dict['graph_model'].get('only_unseen_nodes', False),
                    predict_in_batches=predict_in_batches
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

                fold_tag = key + "_{}".format(fold_id)
                # save results
                append_row(
                    run_conf_dict['dir_name'] + '/results_{}.csv'.format(key),
                    [fold_tag] + results, header, columns
                )

                append_row(
                    run_conf_dict['dir_name'] + '/results_{}_{}.csv'.format("seq", key),
                    [fold_tag] + results_seq, header, columns
                )

                append_row(
                    run_conf_dict['dir_name'] + '/results_{}_{}.csv'.format("graph", key),
                    [fold_tag] + results_graph, header, columns
                )

                del test_data
                del test_labels

            header = False

            del dataset.adj_matrix_train
            del dataset.protein_features
            del dataset

            tf.keras.backend.clear_session()

        # in next set of experiments we start from the first fold
        fold_pos = 0
        header = True


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    tf.random.set_seed(0)

    args = parse_arguments_fixed_folds()

    run_ensemble(args)
