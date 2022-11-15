import os
import argparse
from copy import deepcopy
import importlib
import numpy as np
import pprint
import csv
import pdb
import pandas as pd
import tensorflow as tf
from spektral.layers import GCNConv, GATConv, GINConv

from spektral.utils.convolution import gcn_filter


def load_list_of_modules(string_params, key, root_modules):
    metrics_str = string_params.get(key, [])
    modules = []

    for val in metrics_str:
        # try to load a function/class from one of the modules
        for root_module in root_modules:
            try:
                # we manage to load it, so we break (no need to search the module any more, we'll load it only once)
                module = load_method(root_module, val[0], val[1])
                modules.append(module)
                break
            except (ModuleNotFoundError, AttributeError):
                continue

    return modules


def convert_str_to_py(string_params, list_like_param, modules_list):
    params = {}
    for key, val in string_params.items():
        if key != list_like_param:
            if type(val) is tuple:
                try:
                    params[key] = load_method(tf, val[0], val[1])
                except AttributeError or ModuleNotFoundError:
                    params[key] = load_method('model.losses', val[0], val[1])
            else:
                params[key] = val

    params[list_like_param] = load_list_of_modules(string_params, list_like_param, modules_list)
    return params


def set_directory_for_callbacks(callbacks_list, dir_name):
    new_callbacks_list = []
    for callback_name, callback_params in callbacks_list:
        new_params = deepcopy(callback_params)

        if callback_name == 'keras.callbacks.ModelCheckpoint':
            new_params['filepath'] = dir_name + '/ckpt/' + callback_params['filepath']
        elif callback_name == 'keras.callbacks.TensorBoard':
            new_params['log_dir'] = dir_name + '/logs/'
        elif callback_name == 'keras.callbacks.CSVLogger':
            new_params['filename'] = dir_name + '/' + callback_params['filename']
        elif callback_name in ['LossPlotCallback']:
            new_params['dir_name'] = dir_name

        new_callbacks_list.append((callback_name, new_params))
    return new_callbacks_list


def create_experiment_directory(dir_name):
    # create checkpoint directory
    create_directory(dir_name + '/ckpt/')


def save_conf_file(dir_name, conf):
    # save confing file
    with open(dir_name + "/params.conf", 'w') as f:
        f.write(pprint.pformat(conf))


def read_conf(conf_file):
    conf = eval(open(conf_file, 'r').read())

    return conf['dataset'], conf['model'], conf['training'], conf['run'], deepcopy(conf)


def process_confs(training_conf, run_conf):
    run_conf['callbacks'] = set_directory_for_callbacks(callbacks_list=run_conf['callbacks'],
                                                        dir_name=run_conf['dir_name'])

    training_params = convert_str_to_py(training_conf, 'metrics', [tf])
    run_params = convert_str_to_py(run_conf, 'callbacks', [tf, 'callbacks'])

    return training_params, run_params


# The following function loads functions or classes dynamically from a module
# The implementation is adapted from the original implementation provided by the Argo library: https://github.com/rist-ro/argo
# For more complex structures and processing of config files, please see the Argo library in the link
"""
MIT License

Copyright (c) 2020 deepriemann-poc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def load_method(root_module, method_path, method_kwargs, dict_param=False):
    path_tokens = method_path.split(".")
    method_name = path_tokens[-1]

    path = root_module + '.' + method_path if type(root_module) is str else root_module.__name__

    middle_path = '.'.join(path_tokens[:-1])
    if middle_path:
        path += '.' + middle_path

    # import the module
    child_module = importlib.import_module(path)

    # get the method from imported module
    method = getattr(child_module, method_name)

    if method_kwargs:
        if dict_param:
            return method(method_kwargs)
        else:
            return method(**method_kwargs)

    return method()


def load_model(model_name, model_kwargs, dataset):

    try:
        # try to load a tf model
        model = load_method(tf, model_name, model_kwargs, dict_param=True)
        # create input nodes of dataset manually
        dataset.create_tf_datasets()
    except (ModuleNotFoundError, AttributeError):
        try:
            # try to load a custom model
            model_kwargs['dataset'] = dataset
            model = load_method("model", model_name, model_kwargs, dict_param=True)
        except ModuleNotFoundError:
            raise ValueError("Unknown model! Check the path :P")
    return model


def load_dataset(dataset_name, dataset_kwargs):
    return load_method('dataset', dataset_name, dataset_kwargs, dict_param=True)


def create_layer(layers_tuples, group_using_sequential=True):
    loaded_layers = []
    for layer in layers_tuples:
        try:
            # try to load a usual layer
            loaded_layer = load_method(tf.keras.layers, layer[0], layer[1])
        except AttributeError or ModuleNotFoundError:
            try:
                # try to load a custom layer
                loaded_layer = load_method('model.layers', layer[0], layer[1])
            except ModuleNotFoundError or AttributeError:
                try:
                    # try to load a distribution
                    import tensorflow_probability as tfp
                    loaded_layer = load_method(tfp.distributions, layer[0], layer[1])
                except AttributeError or ModuleNotFoundError:
                    try:
                        # try to load a probabilistic layer
                        import tensorflow_probability as tfp
                        loaded_layer = load_method(tfp.layers, layer[0], layer[1])
                    except AttributeError or ModuleNotFoundError:
                        try:
                            # try to load sth from tf addons
                            import tensorflow_addons as tfa
                            loaded_layer = load_method(tfa, layer[0], layer[1])
                        except AttributeError or ModuleNotFoundError:
                            raise ValueError("Unknown layer! Check the name of path :P")

        loaded_layers.append(loaded_layer)

    if group_using_sequential:
        net = tf.keras.Sequential(loaded_layers)
    else:
        net = loaded_layers

    return net


def create_directory(dir_name):
    # create directory if it does not exist
    os.makedirs(dir_name, exist_ok=True)


def save_results(results, dir_name, columns, header=True):
    # save results to file
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(dir_name + '/results.csv', index=False, mode='a', header=header, float_format='%.4f')


def append_row(file_name, data, header, columns):
    with open(file_name, 'a') as f:
        results_writer = csv.writer(f, delimiter=',')
        if header:
            results_writer.writerow(columns)
        results_writer.writerow(data)


def compute_stats_multiple_keys(results):
    stats = {}
    for key, res in results.items():
        # add means and standard deviations
        results_array = np.array([r[1:] for r in res])
        means = np.mean(results_array, axis=0)
        stds = np.std(results_array, axis=0)
        stats[key] = [['Mean'] + means.tolist(), ['Std'] + stds.tolist()]
    return stats


def compute_means_and_std(results):
    results_array = np.array(results)
    means = np.mean(results_array, axis=0)
    stds = np.std(results_array, axis=0)
    return means, stds


def write_stats(results, dir_name):
    stats = compute_stats_multiple_keys(results)
    for key, _ in results.items():
        with open(dir_name + '/results_{}.csv'.format(key), 'a') as f:
            results_writer = csv.writer(f, delimiter=',')
            results_writer.writerow(stats[key][-2])
            results_writer.writerow(stats[key][-1])


def get_trained_model(dataset, model_conf, ckpt_file, training_conf=None):
    model = load_model(model_conf['name'], model_conf['kwargs'], dataset)

    # compile model
    if training_conf is not None:
        model.compile(**training_conf)

    load_weights(ckpt_file, model)

    return model


def load_weights(ckpt_file, model):
    # load weights
    latest = tf.train.latest_checkpoint(ckpt_file)
    model.load_weights(latest).expect_partial()


def obtain_output(inputs, layers, training=None):
    output = inputs
    for layer in layers:
        output = layer(output, training=training)
    return output


feature_merging_layers = {
    'concatenation': tf.keras.layers.Concatenate(),
    'multiplication': tf.keras.layers.Multiply(),

    'wasserstein_aggregate_mult': tf.keras.layers.Lambda(
        lambda x:
        tf.keras.layers.Concatenate()([
            tf.sqrt(
                tf.reduce_sum(wasserstein_distance(x[0], x[1], sq=False), axis=1, keepdims=True)
            ),
            tf.multiply(x[0][0], x[1][0]),
        ])
    ),

    'wasserstein_aggregate_concat': tf.keras.layers.Lambda(
        lambda x:
        tf.keras.layers.Concatenate()([
            tf.sqrt(
                tf.reduce_sum(wasserstein_distance(x[0], x[1], sq=False), axis=1, keepdims=True)
            ),
            x[0][0],
            x[1][0],
        ])
    ),

}


def wasserstein_distance(distribution1, distribution2, sq=True):
    mean1, std1 = distribution1
    mean2, std2 = distribution2
    wd = (mean1 - mean2) ** 2 + (std1 - std2) ** 2
    if sq:
        wd = tf.math.sqrt(wd)
    return wd


# GNN utils
graph_processing_layers = {
    'gcn': GCNConv,
    'gat': GATConv,
    'gin': GINConv,
}

adj_processing_functions = {
    'gcn': gcn_filter,
    'gat': lambda x: x,
    'gin': lambda x: tf.sparse.from_dense(x)
}


def get_indices_from_batch(indices, batch):
    indices = tf.cast(indices, dtype=tf.int32)
    selection = tf.gather(batch, indices)
    # this is needed when indices are of shape (n, 1); alternatively we could have squeezed indices
    if selection.shape[1] == 1:
        selection = tf.squeeze(selection, axis=1)  # need to squeeze only axis 1 because channel might be of dim 1

    # tensor has an unknown shape because of tf.gather and gives an error in graph mode
    # here we manually set the shape, which is also checked at runtime by tf.ensure_shape
    selection = tf.ensure_shape(selection, [None] + batch.get_shape()[1:])
    return selection


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset')
    parser.add_argument('--num_folds', default=5, help='The number of folds')
    parser.add_argument('--train', action='store_true', help='Whether you want to train the models or to evaluate')
    parser.add_argument('--debug', action='store_true', help='Whether you want to save plots and misclassifications')
    parser.add_argument('--pretrain', action='store_true', help='Whether you want to pretrain the individual models')
    parser.add_argument('--finetune', action='store_true', help='Whether you want to pretrain the individual models')
    parser.add_argument('--fold_start', default=1, help='The starting fold')

    return parser.parse_args()


def parse_arguments_fixed_folds():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset')
    parser.add_argument('--train', action='store_true', help='Whether you want to train the models or to evaluate')
    parser.add_argument('--debug', action='store_true', help='Whether you want to save plots and misclassifications')
    parser.add_argument('--pretrain', action='store_true', help='Whether you want to pretrain the individual models')
    parser.add_argument('--finetune', action='store_true', help='Whether you want to pretrain the individual models')

    parser.add_argument('--start_ex', default=1, help='The starting fold')
    parser.add_argument('--fold_start', default="11", help='The starting fold')

    return parser.parse_args()


def get_train_test_files_names(dataset_name, experiment_type, fold_id):
    train_file = "{}_random_train_{}{}.txt".format(dataset_name, experiment_type, fold_id)

    if experiment_type == "CV_":
        test_files = {"CV": "{}_random_test_CV_{}.txt".format(dataset_name, fold_id)}
    else:
        test_files = {
            exp: "{}_random_test_{}_{}.txt".format(dataset_name, exp, fold_id) for exp in ["C1", "C2", "C3"]
        }
    return train_file, test_files


def get_train_test_files_names_balanced(dataset_name, experiment_type, fold_id):
    train_file = "{}_balanced_train_{}{}.txt".format(dataset_name, experiment_type, fold_id)

    if experiment_type == "CV_":
        test_files = {"CV": "{}_balanced_test_CV_{}.txt".format(dataset_name, fold_id)}
    else:
        test_files = {
            exp: "{}_balanced_test_{}_{}.txt".format(dataset_name, exp, fold_id) for exp in ["C1", "C2", "C3"]
        }
    return train_file, test_files


def get_train_test_files_names_human2021(dataset_name, experiment_type, fold_id):
    train_file = "{}/Train_{}.tsv".format(experiment_type, fold_id)

    if experiment_type == "HeldOut50" or experiment_type == "Random50":
        test_files = {experiment_type: "{}/Test_{}.tsv".format(experiment_type, fold_id)}

    elif experiment_type == "HeldOut20" or experiment_type == "Random20":
        test_files = {
            "{}-10".format(experiment_type): "{}/Test1_{}.tsv".format(experiment_type, fold_id),
            "{}-0.3".format(experiment_type): "{}/Test2_{}.tsv".format(experiment_type, fold_id)
        }

    else:
        raise ValueError('Invalid experiment type')

    return train_file, test_files


class DatasetParams:
    def __init__(self, dataset_type):

        if dataset_type in ['yeast', 'human']:
            self.get_train_test_files_fct = get_train_test_files_names

            self.results = {"C1": [], "C2": [], "C3": [], "CV": []}
            self.results_seq = {"C1": [], "C2": [], "C3": [], "CV": []}
            self.results_graph = {"C1": [], "C2": [], "C3": [], "CV": []}
            self.tprs = {"C1": [], "C2": [], "C3": [], "CV": []}

            self.experiments = ["", "CV_"]
            self.folds = ["{}{}".format(i, j) for i in range(1, 11) for j in range(1, 5)]

            self.keys_list_fct = lambda experiment_type: ["CV"] if experiment_type == "CV_" else ["C1", "C2", "C3"]

        elif dataset_type == 'human2021':
            self.get_train_test_files_fct = get_train_test_files_names_human2021

            self.results = {"HeldOut50": [], "HeldOut20-10": [], "HeldOut20-0.3": []}
            self.results_seq = {"HeldOut50": [], "HeldOut20-10": [], "HeldOut20-0.3": []}
            self.results_graph = {"HeldOut50": [], "HeldOut20-10": [], "HeldOut20-0.3": []}
            self.tprs = {"HeldOut50": [], "HeldOut20-10": [], "HeldOut20-0.3": []}

            self.experiments = ["HeldOut50", "HeldOut20"]
            self.folds = ["{}_{}".format(i, j) for i in range(6) for j in range(i, 6)]

            self.keys_list_fct = lambda experiment_type: ["HeldOut50"] \
                if experiment_type == "HeldOut50" else ["HeldOut20-10", "HeldOut20-0.3"]

        elif dataset_type == 'human2021_random':
            self.get_train_test_files_fct = get_train_test_files_names_human2021

            self.results = {"Random50": [], "Random20-10": [], "Random20-0.3": []}
            self.results_seq = {"Random50": [], "Random20-10": [], "Random20-0.3": []}
            self.results_graph = {"Random50": [], "Random20-10": [], "Random20-0.3": []}
            self.tprs = {"Random50": [], "Random20-10": [], "Random20-0.3": []}

            self.experiments = ["Random50", "Random20"]
            self.folds = ["{}".format(i) for i in range(5)]

            self.keys_list_fct = lambda experiment_type: ["Random50"] \
                if experiment_type == "Random50" else ["Random20-10", "Random20-0.3"]

        else:
            raise ValueError("incorrect dataset type!")
