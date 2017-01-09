import tensorflow as tf
from tensorflow.python.client import device_lib
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

__all__ = ['Dataset', 'create_dataset']

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


def to_bin(data, type_):
    """Convert a numpy array do binary.

    The conversion will lose the shape, the resulting
    array is flat.

    Params:
        data (numpy array): the array to convert
        type_ (string): type of the elements, could be
                        "double" or "float"
    """

    return struct.pack("{}{}".format(
        data.size,
        "d" if type_ == "double" else "f"
    ), *data.flat)


def create_dataset(dataset, loader, size, name, n_shuffle=1, batch_size=True,
                   seed=None, train_percentage=0.8, validation_percentage=0.1,
                   debug=False, type_="double"):
    """Generate a dataset useful for DENN.

    Params:
        dataset (str): dataset file to convert (original)
        loader (str): name of the original data loader
        size (int): num. of elements in a batch or num. of bathes.
                    Set also 'batch_size' properly
        n_shuffle (integer, default=1): number of shuffle of the single dataset.
                                        Subsequent shuffles will be added at
                                        the end of the dataset
        batch_size (bool, default=True): indicate if size is the num. of elems.
                                         or the num. of batches
        seed (int, default=None): seed of random number generator
        train_percentage (float, default=0.8): size in percentage of the train
                                               set
        validation_percentage (float, default=0.1): size in percentage of the
                                                    validation set
        debug (bool, default=False): indicate if the loader enables the debug
                                     output
        type_ (str, default="double"): indicate the type of the elements.
                                       Values can be ["double", "float"]

    Notes:
        If the loaded dataset has already a test set (line MNIST) the
        subdivision will be the subsequent:

        With train_percentage = 0.8

           data                        test set
        +--------+                    +--------+
        |        |                    |        |
        |  80%   |<- train set        |        |
        |        |                    |        |
        |--------|                    |        |
        |  20%   |<- validation set   |        |
        +--------+                    +--------+

        Otherwise the data available will split depending on the percentages:

        With train_percentage = 0.8 and validation_percentage = 0.1

           data
        +--------+
        |        |
        |  80%   |<- train set
        |        |
        |--------|
        |  10%   |<- validation set
        |--------|
        |  10%   |<- test set
        +--------+

        Note also that the loader load the data in float64 format, so subsequent
        conversion are applied only after the creation of the dataset (random
        and shuffle part), before the split part.
    """
    makedirs(BASEDATASETPATH, exist_ok=True)

    print("+ Load data of {}".format(dataset))
    data, labels = globals().get(loader)(dataset, debug)

    # print(len(data), len(labels))

    np.random.seed(seed)

    print("++ Prepare data of {}".format(dataset))

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

    tmp_data = train_data.copy()
    tmp_labels = train_labels.copy()

    # print(tmp_data.shape, len(tmp_data), tmp_data.size)

    for num in range(1, n_shuffle):
        print("++ Add {} shuffle".format(num), end="\r")
        indexes = np.random.permutation(len(tmp_data))
        train_data = np.append(train_data, tmp_data[indexes], axis=0)
        train_labels = np.append(train_labels, tmp_labels[indexes], axis=0)

    # print(train_data.shape, train_labels.shape)

    if not batch_size:
        print("++ Calculate num batches")
        num_elms = len(tmp_data)
        elm_x_batch = size
        size = int(num_elms / size)

    # Convert to float if necessary
    if type_ == "float":
        train_data = train_data.astype(np.float32)
        train_labels = train_labels.astype(np.float32)

    train_data = np.array_split(train_data, size * n_shuffle)
    train_labels = np.array_split(train_labels, size * n_shuffle)

    print("++ Prepare Header")
    header = Header(
        "<5if3I",
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

    is_batch = "-B" if batch_size else "xB"

    print("+++ Create gz file")
    file_name = path.join(
        BASEDATASETPATH, "{}_{}{}_{}s.gz".format(name, size * n_shuffle if batch_size else elm_x_batch, is_batch, n_shuffle))

    with gzip.GzipFile(file_name, mode='wb') as gz_file:

        print("+++ Calculate test size")
        test_size = struct.calcsize("{}{}".format(
            len(test_data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(test_data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )
        print("+++ Calculate validation size")
        validation_size = struct.calcsize("{}{}".format(
            len(validation_data) * header.n_features,
            "d" if type_ == "double" else "f")
        ) + struct.calcsize("{}{}".format(
            len(validation_data) * header.n_classes,
            "d" if type_ == "double" else "f")
        )

        print("+++ Update offsets in header")
        header.set_label("test_offset", len(header))  # size header
        # size header + test size + num elm test
        header.set_label("validation_offset", len(header) + test_size + 4)
        header.set_label("train_offset", len(header) +
                         test_size + validation_size + 8)  # size header + test size + validation size + num elm test + num elm validation

        # print(header)

        print("+++ Write header")
        gz_file.write(header.binary)

        ##
        # TEST
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        print("+++ Write test data and labels")
        gz_file.write(struct.pack("<I", len(test_data)))
        gz_file.write(to_bin(test_data, type_))
        gz_file.write(to_bin(test_labels, type_))

        # print(gz_file.tell())

        ##
        # VALIDATION
        #
        # + num. elems (unsigned long)
        # + data
        # + labels
        print("+++ Write validation data and labels")
        gz_file.write(struct.pack("<I", len(validation_data)))
        gz_file.write(to_bin(validation_data, type_))
        gz_file.write(to_bin(validation_labels, type_))

        # print(gz_file.tell())

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
            print("+++ Write batch {}/{}".format(index +
                                                 1, len(train_data)), end="\r")
            gz_file.write(struct.pack("<I", index))
            gz_file.write(struct.pack("<I", len(t_data)))
            gz_file.write(to_bin(t_data, type_))
            gz_file.write(to_bin(train_labels[index], type_))

    print("+! Dataset {} completed!".format(file_name))


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
