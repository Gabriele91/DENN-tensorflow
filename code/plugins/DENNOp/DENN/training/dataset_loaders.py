import numpy as np
import os
import sys
import gzip
import struct

__all__ = ['load_iris_data', 'load_letter_data', 'load_mnist_data']


def load_mnist_image(file_name):
    with gzip.open(file_name, 'rb') as mnist_file:
        magic_num, num_images, rows, cols = struct.unpack(
            ">IIII", mnist_file.read(16))
        buf = mnist_file.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = np.divide(data, 255.)
        data = data.reshape(num_images, rows*cols)
        return data


def load_mnist_label(file_name, num_classes=10):
    with gzip.open(file_name, 'rb') as mnist_file:
        magic_num, num_labels = struct.unpack(
            ">II", mnist_file.read(8))
        buf = mnist_file.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        data = np.zeros((num_labels, num_classes))
        data[np.arange(labels.size), labels] = 1
        return data


def load_mnist_data(base_path, output=False):
    if not output:
        old_descriptor = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    print('+ Load data ...')
    mnist_data = np.append(
        load_mnist_image(os.path.join(
            base_path, 'train-images-idx3-ubyte.gz')),
        load_mnist_image(os.path.join(
            base_path, 't10k-images-idx3-ubyte.gz')),
        axis=0
    )
    label_data = np.append(
        load_mnist_label(os.path.join(
            base_path, 'train-labels-idx1-ubyte.gz')),
        load_mnist_label(os.path.join(
            base_path, 't10k-labels-idx1-ubyte.gz')),
        axis=0
    )

    print('+ images: \n', mnist_data)
    print('+ labels: \n', label_data)

    print(mnist_data.shape)
    print(label_data.shape)

    print('+ loading done!')

    if not output:
        sys.stdout = old_descriptor

    return mnist_data, label_data


def load_letter_data(path, output=False):
    if not output:
        old_descriptor = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    letter_data = []
    label_data = []

    labels = []
    features_max = []

    print('+ Load data ...')
    with open(path + '.data', 'r') as iris_file:
        for line in iris_file.readlines():
            cur_line = [elm.strip() for elm in line.split(',')]

            if len(cur_line) == 17:
                cur_label = cur_line[0]
                if cur_label not in labels:
                    labels.append(cur_label)

                label_data.append(labels.index(cur_label))

                features = [float(elm) for elm in cur_line[1:]]
                if len(features_max) == 0:
                    features_max = [elm for elm in features]
                else:
                    for idx, feature in enumerate(features):
                        if features_max[idx] < feature:
                            features_max[idx] = feature

                letter_data.append(features)

    features_max = np.array(features_max, np.float64)
    letter_data = np.divide(np.array(letter_data, np.float64), features_max)
    ##
    # expand labels (one hot vector)
    tmp = np.zeros((len(label_data), len(labels)))
    tmp[np.arange(len(label_data)), label_data] = 1
    label_data = tmp

    print('+ letters: \n', letter_data)
    print('+ labels: \n', label_data)

    print('+ loading done!')

    if not output:
        sys.stdout = old_descriptor

    return letter_data, label_data


def load_iris_data(path, output=False):
    if not output:
        old_descriptor = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    flower_data = []
    label_data = []

    labels = []
    features_max = []

    print('+ Load data ...')
    with open(path + '.data', 'r') as iris_file:
        for line in iris_file.readlines():
            cur_line = [elm.strip() for elm in line.split(',')]

            if len(cur_line) == 5:
                cur_label = cur_line[-1]
                if cur_label not in labels:
                    labels.append(cur_label)

                label_data.append(labels.index(cur_label))

                features = [float(elm) for elm in cur_line[:-1]]
                if len(features_max) == 0:
                    features_max = [elm for elm in features]
                else:
                    for idx, feature in enumerate(features):
                        if features_max[idx] < feature:
                            features_max[idx] = feature

                flower_data.append(features)

    features_max = np.array(features_max, np.float64)

    flower_data = np.divide(np.array(flower_data, np.float64), features_max)
    ##
    # expand labels (one hot vector)
    tmp = np.zeros((len(label_data), len(labels)))
    tmp[np.arange(len(label_data)), label_data] = 1
    label_data = tmp

    print('+ flowers: \n', flower_data)
    print('+ labels: \n', label_data)

    print('+ loading done!')

    if not output:
        sys.stdout = old_descriptor

    return flower_data, label_data