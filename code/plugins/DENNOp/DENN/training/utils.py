import tensorflow as tf
import numpy as np
from random import shuffle
from random import seed as set_rnd_seed
from copy import copy


__all__ = ['gen_network', 'Dataset']


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Dataset(object):

    def __init__(self, data, label, seed=None, train_percentage=0.8):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        set_rnd_seed(seed)

        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        train_size = int(len(data) * train_percentage)

        indexes = [_ for _ in range(len(data))]

        shuffle(indexes)
        #print("+ indexes", indexes)
        for index in indexes:
            if len(train_data) < train_size:
                train_data.append(copy(data[index]))
                train_labels.append(copy(label[index]))
            else:
                test_data.append(copy(data[index]))
                test_labels.append(copy(label[index]))

        self.train_data = np.array(train_data, np.float64)
        self.train_labels = np.array(train_labels, np.float64)
        self.test_data = np.array(test_data, np.float64)
        self.test_labels = np.array(test_labels, np.float64)
        
        self.n_classes = len(train_labels[0])
        self.n_features = len(train_data[0])
        self.n_train_elms = len(train_data)

    def batch(self, size=None):
        """Extract a portion of the data to train."""
        if size is None:
            size = len(self.train_data)
        out_data = []
        out_label = []
        for index, elm in enumerate(self.train_data):
            if len(out_data) < size:
                out_data.append(elm)
                out_label.append(self.train_labels[index])
            else:
                yield (np.array(out_data, np.float64),
                       np.array(out_label, np.float64))
                out_data = []
                out_label = []
        # when size == len(self.train_data)
        if len(out_data) != 0:
            yield (np.array(out_data, np.float64),
                   np.array(out_label, np.float64))


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
