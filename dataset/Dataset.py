from copy import deepcopy


class Dataset:
    default_params = {
        'batch_size_val': 10000,
        'batch_size_test': 1000,
        'flatten': False
    }

    def __init__(self, conf_params):
        self._params = deepcopy(Dataset.default_params)
        self._params.update(conf_params)

        self.x_train, self.y_train, self.x_val, self.y_val = None, None, None, None
        self.train_dataset, self.validation_dataset = None, None

    def create_tf_datasets(self):
        pass

    def map_dataset(self, dataset):
        return dataset

    @property
    def data_shape(self):
        return self.x_train.shape

    @property
    def batch_size_train(self):
        return self._params['batch_size_train']

    @property
    def batch_size_test(self):
        return self._params['batch_size_test']

    def get_directory_name(self):
        pass
