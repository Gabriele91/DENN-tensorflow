import tensorflow as tf
import numpy as np
from copy import copy
import json
from os import path
from os import makedirs
from os import SEEK_CUR
import time
import struct
import gzip
import binascii
from collections import namedtuple
from . dataset_loaders import *

__all__ = ['gen_network', 'Dataset', 'create_dataset']

BASEDATASETPATH = './datasets'
Batch = namedtuple('Batch', ['data', 'labels'])


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Header(object):

    """Manage with a simple interface binary headers.

    Example:

        header = Header(
            "<7if",
            [
                # Dataset values
                ("n_batch", 3),
                ("n_features", 4),  # size individual
                ("n_classes", 3),  # size labels
                # Elems
                ('n_train_elms', 120),
                ('n_validation_elms', 15),
                ('n_test_elms', 15),
                # Stats
                ('seed', -1),
                ('train_percentage', 0.8)
            ]
        )

        # get header labels
        print(header.n_classes, header.n_features)
        # set header labels
        print(header.set_label("n_classes", 9))
        # repr of header
        print(header)

        new_header = Header(
            "<7if",
            [
                "n_batch",
                "n_features",
                "n_classes",
                "n_train_elms",
                "n_validation_elms",
                "n_test_elms",
                "seed",
                "train_percentage"
            ],
            header.binary
        )

        print(new_header)
    """

    def __init__(self, fmt, labels, data=None):

        self.__fmt = fmt
        self.__data = data
        self.__p_data = labels

        if not data:
            self.__p_data = labels
            self.__data = struct.pack(
                self.__fmt,
                *[value for label, value in self.__p_data]
            )
        else:
            self.__data = data
            for idx, value in enumerate(struct.unpack(self.__fmt, self.__data)):
                self.__p_data[idx] = (labels[idx], value)

    def __len__(self):
        return struct.calcsize(self.__fmt)

    @property
    def binary(self):
        """Get binary of python data."""
        return self.__data

    def __getattr__(self, name):
        for label, value in self.__p_data:
            if label == name:
                return value
        raise AttributeError("'{}' is not a label of this header!".format(
            name
        ))

    def set_label(self, name, new_value):
        """Change an header label value."""
        for idx, _ in enumerate(self.__p_data):
            if self.__p_data[idx][0] == name:
                self.__p_data[idx] = (name, new_value)

                self.__data = struct.pack(
                    self.__fmt,
                    *[value for label, value in self.__p_data]
                )

                return self.__p_data[idx]

        raise Exception("'{}' is not a label of this header!".format(
            name
        ))

    def __repr__(self):
        byte_per_line = 8

        string = "+++++ HEADER +++++\n"
        string += "+ format string: '{}'\n".format(self.__fmt)
        string += "+----------\n"

        for label, value in self.__p_data:
            string += "+ {}: {}\n".format(label, value)

        string += "+----------\n"

        data = binascii.b2a_hex(self.__data)
        counter = 0
        for idx in range(0, len(data), 4):
            if counter == 0:
                string += "+ "
            elif counter == byte_per_line:
                counter = 0
                string += "\n+ "

            string += "{} ".format("".join([chr(data[idx + cur_i])
                                            for cur_i in range(4)]))
            counter += 2

        string += "\n+----------"

        return string


def create_dataset(dataset, loader, batch_size, name,
                   seed=None, train_percentage=0.8, validation_percentage=0.1,
                   debug=False, type_="double"):
    makedirs(BASEDATASETPATH, exist_ok=True)

    data, labels = globals().get(loader)(dataset, debug)

    # print(len(data), len(labels))

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
        partial_labels = partial_labels[indexes]

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
        labels = labels[indexes]

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

    header = Header(
        "<5if3L",
        [
            # Dataset values
            ("n_batch", len(train_data)),
            ("n_features", len(train_data[0][0])),  # size individual
            ("n_classes", len(train_labels[0][0])),  # size labels
            ##
            # Type of values:
            #   1 -> float
            #   2 -> double
            ('type', 2 if type_ == "double" else 1),
            # Stats
            ('seed', seed if seed is not None else -1),
            ('train_percentage', train_percentage),
            # Offset
            ('test_offset', 0),
            ('validation_offset', 0),
            ('train_offset', 0)
        ]
    )

    # print(header)
    # print(len(header))

    with gzip.GzipFile(path.join(BASEDATASETPATH, "{}_{}_batches.gz".format(name, batch_size)), mode='wb') as gz_file:

        test_size = struct.calcsize("{}{}".format(
            len(test_data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(test_data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )
        validation_size = struct.calcsize("{}{}".format(
            len(validation_data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(validation_data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )

        header.set_label("test_offset", len(header))
        header.set_label("validation_offset", len(header) + test_size)
        header.set_label("train_offset", len(header) +
                         test_size + validation_size)

        # print(header)

        gz_file.write(header.binary)

        ##
        # TEST
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        gz_file.write(struct.pack("<I", len(test_data)))
        gz_file.write(test_data.tobytes())
        gz_file.write(test_labels.tobytes())

        ##
        # VALIDATION
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        gz_file.write(struct.pack("<I", len(validation_data)))
        gz_file.write(validation_data.tobytes())
        gz_file.write(validation_labels.tobytes())

        ##
        # TRAIN
        #
        # [
        #   + current batch num (unsigned int)
        #   + num. elems (unsigned long)
        #   + data
        #   + labels
        # ]
        for index, t_data in enumerate(train_data):
            gz_file.write(struct.pack("<I", index))
            gz_file.write(struct.pack("<I", len(t_data)))
            gz_file.write(t_data.tobytes())
            gz_file.write(train_labels[index].tobytes())


class Dataset(object):

    def __init__(self, file_name):

        self.__file_name = file_name

        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            self.stats = Header(
                "<5if3L",
                [
                    "n_batch",
                    "n_features",
                    "n_classes",
                    "type",
                    "seed",
                    "train_percentage",
                    "test_offset",
                    "validation_offset",
                    "train_offset"
                ],
                gz_file.read(36)
            )

        # print(self.stats)
        self.__dtype = np.float64 if self.stats.type == 2 else np.float32
        self.__elm_size = 8 if self.stats.type == 2 else 4
        self.__size_elm_data = self.stats.n_features * self.__elm_size
        self.__size_elm_label = self.stats.n_classes * self.__elm_size

    def __read_from(self, offset, type_):
        """Read data from offset. 

        Args:
            offset: num of bytes to jump
            type: 0 if data, 1 if label
        """
        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(offset)
            num_elms = struct.unpack("<I", gz_file.read(4))[0]
            if type_ == 1:
                gz_file.seek(num_elms * self.__size_elm_data, SEEK_CUR)
                data = np.frombuffer(gz_file.read(
                    num_elms * self.__size_elm_label
                ), dtype=self.__dtype)
            else:
                data = np.frombuffer(gz_file.read(
                    num_elms * self.__size_elm_data
                ), dtype=self.__dtype)
            data = data.reshape([
                num_elms,
                self.stats.n_features if type_ == 0 else self.stats.n_classes
            ])
        return data

    @property
    def test_data(self):
        return self.__read_from(self.stats.test_offset, 0)

    @property
    def test_labels(self):
        return self.__read_from(self.stats.test_offset, 1)

    @property
    def validation_data(self):
        return self.__read_from(self.stats.validation_offset, 0)

    @property
    def validation_labels(self):
        return self.__read_from(self.stats.validation_offset, 1)

    @property
    def num_batches(self):
        return self.stats.n_batch

    def batches(self):
        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(self.stats.train_offset)

            for idx in range(self.num_batches):
                num_batch = struct.unpack("<I", gz_file.read(4))[0]
                num_elms = struct.unpack("<I", gz_file.read(4))[0]

                data = np.frombuffer(
                    gz_file.read(num_elms * self.__size_elm_data),
                    dtype=self.__dtype
                )
                data = data.reshape([num_elms, self.stats.n_features])

                labels = np.frombuffer(
                    gz_file.read(num_elms * self.__size_elm_label),
                    dtype=self.__dtype
                )
                labels = labels.reshape([num_elms, self.stats.n_classes])

                yield Batch(data, labels)

    def __getitem__(self, index):
        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(self.stats.train_offset)

            for idx in range(self.num_batches):
                num_batch = struct.unpack("<I", gz_file.read(4))[0]
                num_elms = struct.unpack("<I", gz_file.read(4))[0]

                if num_batch == index:
                    break
                else:
                    gz_file.seek(
                        num_elms * self.__size_elm_data +
                        num_elms * self.__size_elm_label, SEEK_CUR)

            data = np.frombuffer(
                gz_file.read(num_elms * self.__size_elm_data),
                dtype=self.__dtype
            )
            data = data.reshape([num_elms, self.stats.n_features])

            labels = np.frombuffer(
                gz_file.read(num_elms * self.__size_elm_label),
                dtype=self.__dtype
            )
            labels = labels.reshape([num_elms, self.stats.n_classes])

        return Batch(data, labels)


def gen_network(levels, options,
                cur_data, cur_label,
                validation_data, validation_labels,
                test_data, test_labels,
                rand_pop):
    target_ref = []
    pop_ref = []
    rand_pop_ref = []
    cur_pop_VAL = tf.placeholder(tf.float64, [options.NP])
    weights = []

    last_input_train = cur_data
    last_input_validation = validation_data
    last_input_test = test_data

    # print(cur_data)
    # print(cur_label)
    # print(validation_data)
    # print(validation_labels)
    # print(test_data)
    # print(test_labels)

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
            # NN VALIDATION
            y_validation = tf.matmul(
                last_input_validation, target_w) + target_b
            correct_prediction_validation = tf.equal(
                tf.argmax(y_validation, 1),
                tf.argmax(validation_labels, 1)
            )
            accuracy_validation = tf.reduce_mean(
                tf.cast(correct_prediction_validation, tf.float32))

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
            last_input_validation = getattr(tf.nn, type_)(
                tf.matmul(last_input_validation, target_w) + target_b)
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
        ('accuracy', accuracy),
        ('accuracy_validation', accuracy_validation)
    ])
