import numpy as np
from os import SEEK_CUR
import struct
import gzip
import binascii
import json
from collections import namedtuple

__all__ = ['Dataset', 'calc_confusin_M', 'calc_TF',
           'precision_recall_acc', 'f1_score',
           'calc_bin_stats', 'calc_pop_diff',
           'TestResults', 'OutOptions']

Batch = namedtuple('Batch', ['data', 'labels'])
BinClassify = namedtuple('BinClassify', ['TP', 'FP', 'FN', 'TN'])


def calc_pop_diff(test_result_file, out_file="distances.json"):
    with open(test_result_file, "r") as res_file:
        res = json.load(res_file)['results']

    hem_reduce = {}
    eu_distance = {}

    hem_distance = lambda x, y: np.fabs(x - y)
    sq_distance = lambda x, y: np.square(x - y)

    for method in res:
        best = np.array(res[method]['best_of']['individual'])
        population = np.array(res[method]['population'])

        total_size = 0
        for elm in best:
            total_size += np.array(elm).size

        for type_, pop_type in enumerate(population):
            for idx, matrix in enumerate(pop_type):
                cur_matrix = np.array(matrix)
                sum_ = np.sum([
                    hem_distance(best[type_], cur_matrix)
                ])
                if idx not in hem_reduce:
                    hem_reduce[idx] = 0
                hem_reduce[idx] += sum_

                sum_ = np.sum([
                    sq_distance(best[type_], cur_matrix)
                ])
                if idx not in eu_distance:
                    eu_distance[idx] = 0
                eu_distance[idx] += sum_

        for key in eu_distance:
            eu_distance[key] = float(np.sqrt(eu_distance[key]))

    output = [(idx, [elm, eu_distance[idx], elm / total_size,
                     eu_distance[idx] / total_size]) for idx, elm in hem_reduce.items()]

    with open(out_file, "w") as out_file:
        json.dump(
            list(reversed(sorted(output, key=lambda elm: elm[1][0]))), out_file, indent=2)


def calc_confusin_M(labels, cur_y):
    size = labels.shape[-1]

    labels = [np.argmax(row) for row in labels]
    cur_y = [np.argmax(row) for row in cur_y]

    matrix = np.full((size, size), 0, dtype=np.int32)

    for idx, class_ in enumerate(labels):
        matrix[class_][cur_y[idx]] += 1

    return matrix.astype(int)


def calc_TF(confusion_m, class_):
    TP = confusion_m[class_][class_]
    FP = np.sum(confusion_m[class_]) - TP
    FN = np.sum(confusion_m[:, class_]) - TP
    TN = np.sum(confusion_m.diagonal()) - TP

    return BinClassify(TP, FP, FN, TN)


def precision_recall_acc(bc):
    precision = bc.TP / (bc.TP + bc.FP)
    recall = bc.TP / (bc.TP + bc.FN)

    acc = (bc.TP + bc.TN) / np.sum(bc)

    return precision, recall, acc


def f1_score(binclassify):
    precision, recall, acc = precision_recall_acc(binclassify)

    return 2. * (precision * recall) / (precision + recall)


def calc_bin_stats(confusion_m):
    tmp_0 = calc_TF(confusion_m, 0)
    tmp_1 = calc_TF(confusion_m, 1)
    p_0, r_0, acc_0 = precision_recall_acc(tmp_0)
    p_1, r_1, acc_1 = precision_recall_acc(tmp_1)
    f1_0 = f1_score(tmp_0)
    f1_1 = f1_score(tmp_1)

    return {
        'p_0': p_0,
        'r_0': r_0,
        'acc_0': acc_0,
        'p_1': p_1,
        'r_1': r_1,
        'acc_1': acc_1,
        'f1_0': f1_0,
        'f1_1': f1_1
    }


class TestResults(dict):

    """Container for de results."""

    def __init__(self, de_types):
        super(self.__class__, self).__init__()
        for de_t in de_types:
            self[de_t] = Results()


class Results(dict):

    """Container for the single benchmark results."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self.values = []
        self.best_of = {
            'accuracy': [0],
            'accuracy_val': [0],
            'individual': None,
            'F': None,
            'CR': None
        }
        self.reset_list = []
        self.F_population = None
        self.CR_population = None
        self.population_test = []
        self.population = None


class OutOptions(object):

    def __init__(self, job, num_batches):
        self.job = job
        self.num_batches = num_batches


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
        self.__out_size = 42

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

        string = "+----------- HEADER ".ljust(self.__out_size, "-") + '+\n'
        format_ = "| format string: '{}'".format(self.__fmt)
        string += format_.ljust(self.__out_size, " ") + '|\n'
        string += "+".ljust(self.__out_size, "-") + '+\n'

        for label, value in self.__p_data:
            cur_data = "| {}: {}".format(label, value)
            string += cur_data.ljust(self.__out_size, " ") + '|\n'

        string += "+".ljust(self.__out_size, "-") + '+\n'

        data = binascii.b2a_hex(self.__data)
        counter = 0
        cur_data = ''

        for idx in range(0, len(data), 4):
            if counter == 0:
                cur_data += "| "
            elif counter == byte_per_line:
                counter = 0
                string += cur_data.ljust(self.__out_size, " ") + '|\n'
                cur_data = "| "

            cur_data += "{} ".format("".join([chr(data[idx + cur_i])
                                            for cur_i in range(4)]))
            counter += 2

        string += "+".ljust(self.__out_size, "-") + '+\n'

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
                "<H5if3I",
                [
                    "version",
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
                gz_file.read(38)
            )

        # print(self.stats)
        self.type = 'double' if self.stats.type == 2 else 'float'
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
        if index > self.num_batches - 1:
            index %= self.num_batches

        with gzip.GzipFile(self.__file_name, mode='rb') as gz_file:
            gz_file.seek(self.stats.train_offset)

            for idx in range(self.num_batches):
                num_batch = struct.unpack("<I", gz_file.read(4))[0]
                num_elms = struct.unpack("<I", gz_file.read(4))[0]

                # print('Read item ->', num_batch, num_elms)

                if num_batch == index:
                    break
                else:
                    gz_file.seek(
                        num_elms * self.__size_elm_data +
                        num_elms * self.__size_elm_label, SEEK_CUR)

            # print('Read item ->', num_elms, self.__size_elm_data)
            data = np.frombuffer(
                gz_file.read(num_elms * self.__size_elm_data),
                dtype=self.__dtype
            )
            # print('Read item ->', data.shape)
            data = data.reshape([num_elms, self.stats.n_features])
            # print('Read item ->', data.shape)

            labels = np.frombuffer(
                gz_file.read(num_elms * self.__size_elm_label),
                dtype=self.__dtype
            )
            # print('Read item ->', labels.shape)
            labels = labels.reshape([num_elms, self.stats.n_classes])
            # print('Read item ->', labels.shape)

            # print(data[0])
            # print(labels[0])

        return Batch(data, labels)
