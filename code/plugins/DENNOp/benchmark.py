import DENN
import tensorflow as tf
import numpy as np
import dataset_loaders
from os import makedirs
from random import shuffle
from random import seed as set_rnd_seed
from copy import copy

from time import sleep

#sleep(6)

makedirs("./benchmark_results", exist_ok=True)


class Options(dict):

    def __init__(self, *args, **kwargs):
        super(Options, self).__init__(*args, **kwargs)
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

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

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


def load_data(datasets_data):
    for path_, loader, options in datasets_data:
        yield (*getattr(dataset_loaders, loader)(path_), options)


def main():
    ##
    # Datasets
    datasets = []

    datasets_data = [
        (
            "../../../minimal_dataset/data/bezdekIris",
            'load_iris_data',
            Options(
                [
                    ('GEN', 500),
                    ('NP', 100),
                    ('BATCH', 10),
                    ('W', 0.3),
                    ('CR', 0.5),
                    ('DE', 'rand/2/bin')
                ]
            )
        )
    ]

    ##
    # Load data
    for data, labels, options in load_data(datasets_data):
        datasets.append((Dataset(data, labels), options))

    for dataset, options in datasets:
        SIZE_W = [dataset.n_features, dataset.n_classes]
        SIZE_B = [dataset.n_classes]
        SIZE_X = [dataset.n_features]

        ##
        # DE W -> NN (W, B)
        deW_nnW = np.full(SIZE_W, options.W)
        deW_nnB = np.full(SIZE_B, options.W)

        graph = tf.Graph()
        with graph.as_default():
            ##
            # Random functions
            create_random_population_W = tf.random_uniform(
                [options.NP] + SIZE_W, dtype=tf.float64, seed=1,
                name="create_random_population_W")
            create_random_population_B = tf.random_uniform(
                [options.NP] + SIZE_B, dtype=tf.float64, seed=1,
                name="create_random_population_B")

            ##
            # Placeholder
            target_w = tf.placeholder(tf.float64, SIZE_W, name="target_0")
            target_b = tf.placeholder(tf.float64, SIZE_B, name="target_1")

            cur_pop_W = tf.placeholder(tf.float64, [options.NP] + SIZE_W)
            cur_pop_B = tf.placeholder(tf.float64, [options.NP] + SIZE_B)
            cur_pop_VAL = tf.placeholder(tf.float64, [options.NP])
            # input
            target_x = tf.placeholder(
                tf.float64, [dataset.n_train_elms] + SIZE_X)
            # correct labels
            target_y_label = tf.placeholder(
                tf.float64, [dataset.n_train_elms] + SIZE_B)

            ##
            # Variables
            # population of W
            cur_pop_W_var = tf.Variable(
                np.zeros([options.NP] + SIZE_W), dtype=tf.float64)
            # population of B
            cur_pop_B_var = tf.Variable(
                np.zeros([options.NP] + SIZE_B), dtype=tf.float64)
            # evalutaion of the population
            cur_pop_VAL_var = tf.Variable(
                np.zeros([options.NP]), dtype=tf.float64)
            # input
            target_x_var = tf.Variable(
                np.zeros([dataset.n_train_elms] + SIZE_X), dtype=tf.float64)
            # correct labels
            target_y_label_var = tf.Variable(
                np.zeros([dataset.n_train_elms] + SIZE_B), dtype=tf.float64)

            ##
            # Assign operations
            assign_pop_W = cur_pop_W_var.assign(cur_pop_W)
            assign_pop_B = cur_pop_B_var.assign(cur_pop_B)
            assign_pop_VAL = cur_pop_VAL_var.assign(cur_pop_VAL)
            assign_x = target_x_var.assign(target_x)
            assign_y_label = target_y_label_var.assign(target_y_label)

            ##
            # NN
            y = tf.matmul(target_x_var, target_w) + target_b
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    y, target_y_label_var), name="evaluate")

            with tf.Session() as sess:
                # init vars
                sess.run(tf.global_variables_initializer())

                ##
                # DENN op
                denn_op = DENN.create(  # input params
                    [1, 1],
                    [],  # FIRST EVAL
                    [deW_nnW, deW_nnB],  # PASS WEIGHTS
                    [cur_pop_W_var, cur_pop_B_var],  # POPULATIONS
                    # attributes
                    # space = 2,
                    graph=DENN.get_graph_proto(sess.graph.as_graph_def()),
                    CR=options.CR,
                    DE=options.DE
                )

                rand_w, rand_b = sess.run([
                    create_random_population_W,
                    create_random_population_B
                ])

                sess.run([assign_pop_W, assign_pop_B], feed_dict={
                    cur_pop_W: rand_w,
                    cur_pop_B: rand_b
                })

                for cur_data, cur_label in dataset.batch():
                    
                    sess.run([assign_x, assign_y_label], feed_dict={
                        target_x: cur_data,
                        target_y_label: cur_label
                    })

                    for gen in range(options.GEN):
                        results = sess.run(denn_op)
                        # get output
                        w_res = results.final_populations[0]
                        b_res = results.final_populations[1]
                        v_res = results.final_eval

                        sess.run([
                            assign_pop_W,
                            assign_pop_B,
                            assign_pop_VAL
                        ], feed_dict={
                            cur_pop_W: w_res,
                            cur_pop_B: b_res,
                            cur_pop_VAL: v_res
                        })


if __name__ == '__main__':
    main()
