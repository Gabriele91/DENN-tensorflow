import tensorflow as tf
import numpy as np
from copy import copy
import json
import zipfile
from os import path
from os import makedirs
import time
from collections import namedtuple
from . dataset_loaders import *

__all__ = ['gen_network', 'Dataset', 'create_dataset']

BASEDATASETPATH = './datasets'
Batch = namedtuple('Batch', ['data', 'labels'])


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_dataset(dataset, loader, batch_size, name,
                   seed=None, train_percentage=0.8, validation_percentage=0.1,
                   debug=False):
    makedirs(BASEDATASETPATH, exist_ok=True)

    data, labels = globals().get(loader)(dataset, debug)

    np.random.seed(seed)

    if len(data) == 2 and len(labels) == 2:
        ##
        # Test set is already splitted
        # Train set will split by train_percentage
        partial_data, test_data = data
        partial_labels, test_labels = labels

        train_size = int(len(partial_data) * train_percentage)
        validation_size = len(partial_data) - train_size
        test_size = len(test_data)

        indexes = np.random.permutation(len(partial_data))
        partial_data = partial_data[indexes]

        train_data, validation_data = np.array_split(
            partial_data, [train_size])
        train_labels, validation_labels = np.array_split(
            partial_labels, [train_size])

    else:
        ##
        # Will create all the sets
        train_size = int(len(data) * train_percentage)
        validation_size = int(len(data) * validation_percentage)
        test_size = len(data) - train_size - validation_size

        indexes = np.random.permutation(len(data))
        data = data[indexes]

        train_data, validation_data, test_data = np.split(data, [
            train_size,
            train_size + validation_size
        ])

        train_labels, validation_labels, test_labels = np.split(labels, [
            train_size,
            train_size + validation_size
        ])

    # print(train_size, validation_size, test_size)
    # print(train_data.shape, train_labels.shape)
    # print(validation_data.shape, validation_labels.shape)
    # print(test_data.shape, test_labels.shape)

    train_data = np.split(train_data, batch_size)
    train_labels = np.split(train_labels, batch_size)

    stats = {
        'n_classes': len(train_labels[0]),
        'n_features': len(train_data[0]),
        # Elems
        'n_train_elms': len(train_data),
        'n_validation_elms': len(validation_data),
        'n_test_elms': len(test_data),
        # Shapes
        'train_data_shape': [elm.shape for elm in train_data],
        'train_labels_shape': [elm.shape for elm in train_labels],
        'validation_data_shape': validation_data.shape,
        'validation_labels_shape': validation_labels.shape,
        'test_data_shape': test_data.shape,
        'test_labels_shape': test_labels.shape,
        # Stats
        'seed': seed,
        'train_percentage': train_percentage
    }

    with zipfile.ZipFile(path.join(BASEDATASETPATH, "{}_{}_batches.bzip".format(
        name, batch_size)
    ),
            mode='w', compression=zipfile.ZIP_BZIP2) as zip_file:

        zip_file.writestr(
            'stats.json',
            json.dumps(stats, indent=4)
        )
        for index, t_data in enumerate(train_data):
            zip_file.writestr(
                'train_{}.data'.format(index),
                t_data.tobytes()
            )
            zip_file.writestr(
                'train_{}.labels'.format(index),
                train_labels[index].tobytes()
            )
        zip_file.writestr(
            'validation.data',
            validation_data.tobytes()
        )
        zip_file.writestr(
            'validation.labels',
            validation_labels.tobytes()
        )
        zip_file.writestr(
            'test.data',
            test_data.tobytes()
        )
        zip_file.writestr(
            'test.labels',
            test_labels.tobytes()
        )


class Dataset(object):

    def __init__(self, file_name):
        self.__zip_file = zipfile.ZipFile(
            file_name, mode='r',
            compression=zipfile.ZIP_DEFLATED)

        self.stats = json.loads(self.__zip_file.read(
            "stats.json").decode("utf-8"))

    @property
    def test_data(self):
        data = np.frombuffer(self.__zip_file.read(
            'test.data'), dtype=np.float64)
        data = data.reshape(*self.stats['test_data_shape'])
        return data

    @property
    def test_labels(self):
        data = np.frombuffer(self.__zip_file.read(
            'test.labels'), dtype=np.float64)
        data = data.reshape(*self.stats['test_labels_shape'])
        return data

    @property
    def num_batches(self):
        return len(self.stats['train_data_shape'])

    def __getitem__(self, index):
        train_shape = self.stats['train_data_shape']
        train_lbl_shape = self.stats['train_labels_shape']

        if index > len(train_shape):
            raise Exception("index > of num batches: {} > {}".format(
                index, len(train_shape)
            ))

        with self.__zip_file.open('train_{}.data'.format(index)) as train_stream:
            data = np.frombuffer(
                train_stream.read(),
                dtype=np.float64
            )
            data = data.reshape(*train_shape[index])

        with self.__zip_file.open('train_{}.labels'.format(index)) as train_labels:
            labels = np.frombuffer(
                train_labels.read(),
                dtype=np.float64
            )
            labels = labels.reshape(*train_lbl_shape[index])

        return Batch(data, labels)

    def __del__(self):
        self.__zip_file.close()


def gen_network(levels, options, cur_data, cur_label, test_data, test_labels, rand_pop):
    target_ref = []
    pop_ref = []
    rand_pop_ref = []
    cur_pop_VAL = tf.placeholder(tf.float64, [options.NP])
    weights = []

    last_input_train = cur_data
    last_input_test = test_data

    for num, cur_level in enumerate(levels, 1):
        level, type_ = cur_level

        SIZE_W, SIZE_B = level

        ##
        # DE W -> NN (W, B)
        deW_nnW = np.full(SIZE_W, options.W)
        deW_nnB = np.full(SIZE_B, options.W)

        weights.append(deW_nnW)
        weights.append(deW_nnB)

        ##
        # Random functions
        if rand_pop:
            create_random_population_W = tf.random_uniform(
                [options.NP] + SIZE_W, dtype=tf.float64, seed=1)
            create_random_population_B = tf.random_uniform(
                [options.NP] + SIZE_B, dtype=tf.float64, seed=1)

            rand_pop_ref.append(create_random_population_W)
            rand_pop_ref.append(create_random_population_B)

        ##
        # Placeholder
        target_w = tf.placeholder(tf.float64, SIZE_W)
        target_b = tf.placeholder(tf.float64, SIZE_B)

        target_ref.append(target_w)
        target_ref.append(target_b)

        cur_pop_W = tf.placeholder(tf.float64, [options.NP] + SIZE_W)
        cur_pop_B = tf.placeholder(tf.float64, [options.NP] + SIZE_B)

        pop_ref.append(cur_pop_W)
        pop_ref.append(cur_pop_B)

        if num == len(levels):
            ##
            # NN TRAIN
            y = tf.matmul(last_input_train, target_w) + target_b
            cross_entropy = tf.reduce_mean(
                getattr(tf.nn, type_)(
                    y, cur_label), name="evaluate")

            ##
            # NN TEST
            y_test = tf.matmul(last_input_test, target_w) + target_b
            correct_prediction = tf.equal(
                tf.argmax(y_test, 1),
                tf.argmax(test_labels, 1)
            )
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
        else:
            last_input_train = getattr(tf.nn, type_)(
                tf.matmul(last_input_train, target_w) + target_b)
            last_input_test = getattr(tf.nn, type_)(
                tf.matmul(last_input_test, target_w) + target_b)

    return ENDict([
        ('targets', target_ref),
        ('populations', pop_ref),
        ('rand_pop', rand_pop_ref),
        ('weights', weights),
        ('evaluated', cur_pop_VAL),
        ('y', y),
        ('y_test', y_test),
        ('cross_entropy', cross_entropy),
        ('accuracy', accuracy)
    ])
