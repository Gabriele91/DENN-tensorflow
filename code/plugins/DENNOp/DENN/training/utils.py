import tensorflow as tf
import numpy as np
from random import shuffle
from random import seed as set_rnd_seed
from copy import copy
import json
import zipfile
from os import path
from os import makedirs
import time
from collections import namedtuple

__all__ = ['gen_network', 'Dataset', 'create_dataset']

BASEDATASETPATH = './datasets'
Batch = namedtuple('Batch', ['data', 'labels'])


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_dataset(name, data, label, batch_size, seed=None, train_percentage=0.8):
    makedirs(BASEDATASETPATH, exist_ok=True)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    set_rnd_seed(seed)

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    train_size = int(len(data) * train_percentage)

    indexes = [_ for _ in range(len(data))]

    shuffle(indexes)

    for index in indexes:
        if len(train_data) < train_size:
            train_data.append(copy(data[index]))
            train_labels.append(copy(label[index]))
        else:
            test_data.append(copy(data[index]))
            test_labels.append(copy(label[index]))

    train_data = np.array(train_data, np.float64)
    train_data = np.array_split(train_data, batch_size)
    train_labels = np.array(train_labels, np.float64)
    train_labels = np.array_split(train_labels, batch_size)

    test_data = np.array(test_data, np.float64)
    test_labels = np.array(test_labels, np.float64)

    stats = {
        'n_classes': len(train_labels[0]),
        'n_features': len(train_data[0]),
        'n_train_elms': len(train_data),
        'n_test_elms': len(test_data),
        'train_data_shape': [elm.shape for elm in train_data],
        'train_labels_shape': [elm.shape for elm in train_labels],
        'test_data_shape': test_data.shape,
        'test_labels_shape': test_labels.shape,
        'seed': seed,
        'train_percentage': train_percentage
    }

    with zipfile.ZipFile(path.join(BASEDATASETPATH, "{}_{}_batches.zip".format(
        name, batch_size)
    ),
        mode='w',
            compression=zipfile.ZIP_DEFLATED) as zip_file:

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


def gen_network(levels, options, cur_data, cur_label, test_data, test_labels):
    target_ref = []
    pop_ref = []
    rand_pop_ref = []
    cur_pop_VAL = tf.placeholder(tf.float64, [options.NP])
    weights = []

    last_input_train = cur_data
    last_input_test = test_data

    for num, level in enumerate(levels, 1):
        SIZE_W, SIZE_B = level

        ##
        # DE W -> NN (W, B)
        deW_nnW = np.full(SIZE_W, options.W)
        deW_nnB = np.full(SIZE_B, options.W)

        weights.append(deW_nnW)
        weights.append(deW_nnB)

        ##
        # Random functions
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
                tf.nn.softmax_cross_entropy_with_logits(
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
            last_input_train = tf.nn.relu(
                tf.matmul(last_input_train, target_w) + target_b)
            last_input_test = tf.nn.relu(
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
