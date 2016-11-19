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
        for _ in range(num_images):
            yield [elm / 255. for elm in struct.unpack(
                "B" * rows * cols, mnist_file.read(rows * cols))]


def load_mnist_label(file_name):
    with gzip.open(file_name, 'rb') as mnist_file:
        magic_num, num_labels = struct.unpack(
            ">II", mnist_file.read(8))
        for _ in range(num_labels):
            label = struct.unpack('B', mnist_file.read(1))[0]
            yield [0 if idx != label else 1 for idx in range(10)]


def load_mnist_data(base_path, output=False):
    if not output:
        old_descriptor = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    mnist_data = []
    label_data = []

    print('+ Load data ...')
    for image in load_mnist_image(os.path.join(base_path, 'train-images-idx3-ubyte.gz')):
        mnist_data.append(image)
    for image in load_mnist_image(os.path.join(base_path, 't10k-images-idx3-ubyte.gz')):
        mnist_data.append(image)
    for label in load_mnist_label(os.path.join(base_path, 'train-labels-idx1-ubyte.gz')):
        label_data.append(label)
    for label in load_mnist_label(os.path.join(base_path, 't10k-labels-idx1-ubyte.gz')):
        label_data.append(label)
    
    mnist_data = np.array(mnist_data, np.float64)
    label_data = np.array(label_data, np.float64)

    print('+ images: \n',mnist_data)
    print('+ labels: \n',label_data)

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
