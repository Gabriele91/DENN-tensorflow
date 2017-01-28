import numpy as np
from os import SEEK_CUR
import struct
import gzip
import binascii
from collections import namedtuple

__all__ = ['Dataset']

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
