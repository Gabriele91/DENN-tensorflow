import DENN
import tensorflow as tf
import numpy as np
import dataset_loaders
from os import makedirs
from os import path
from random import shuffle
from random import seed as set_rnd_seed
from copy import copy
from matplotlib import pyplot as plt
from plotter import my_plot

from time import sleep
from time import time

# sleep(6)

makedirs("./benchmark_results", exist_ok=True)


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


def load_data(datasets_data):
    for path_, loader, options in datasets_data:
        yield (*getattr(dataset_loaders, loader)(path_), options)


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


def write_results(name, results, description, separated=False, showDelimiter=False):
    colors = [
        "#dd0000",
        "#00dd00",
        "#0000dd",
        "#ffdd00"
    ]

    figures = []

    if not separated:
        all_data = [
            {
                'values': [range(len(result.values)), result.values],
                'color': colors[num],
                'label': name,
                'alpha': 0.9
            }
            for num, (name, result) in enumerate(
                sorted(results.items())
            )
        ]
        figures.append(
            {
                'data': all_data,
                'title': name,
                'type': "plot",
                'axis': (0, len(all_data[0]['values'][0]), 0.0, 1.0),
                'filename': path.join("benchmark_results", name),
                'plot': {
                    'x_label': "generation",
                    'y_label': "accuracy",
                }
            }
        )
    else:
        for num, (method_name, result) in enumerate(sorted(results.items())):
            figures.append(
                {
                    'data': [
                        {
                            'values': [range(len(result.values)), result.values],
                            'color': colors[0],
                            'label': method_name,
                            'alpha': 0.9
                        }
                    ],
                    'title': method_name,
                    'type': "plot",
                    'axis': (0, len(result.values), 0.0, 1.0),
                    'filename': path.join("benchmark_results", "{}_{}".format(name, method_name.replace("/", "_"))),
                    'plot': {
                        'x_label': "generation",
                        'y_label': "accuracy",
                    }
                }
            )

    for figure in figures:
        fig = plt.figure()
        fig.suptitle(figure['title'], fontsize=14, fontweight='bold')

        if figure['type'] == 'plot':
            print("+ Generating {} [{}] -> {}".format(
                figure['title'],
                figure['type'],
                figure['filename']
            ))
            my_plot(fig, figure['data'])
            if 'axis' in figure:
                plt.axis(figure['axis'])
            plt.xlabel(figure['plot']['x_label'])
            plt.ylabel(figure['plot']['y_label'])
            plt.grid(True)
            plt.figtext(.39, -.02, description)

            plt.legend(bbox_to_anchor=(1.32, 1.0))

            if showDelimiter:
                delimiters = [50, 100, 200, 400]
                for delimiter in delimiters:
                    plt.axvline(delimiter, color='k')

            plt.savefig(figure['filename'], dpi=400, bbox_inches='tight')


def main():
    ##
    # Datasets
    datasets = []
    EVALUATE_STEPS = [
        0, 50, 100  # , 200, 400, 500
    ]

    datasets_data = [
        (
            "../../../minimal_dataset/data/bezdekIris",
            'load_iris_data',
            ENDict(
                [
                    ('name', 'iris_dataset'),
                    ('GEN', 400),
                    ('NP', 100),
                    ('BATCH', 40),  # USE THIS!!!!!!!!! (TO DO)
                    ('W', 0.3),
                    ('CR', 0.552)
                ]
            )
        )
    ]

    ##
    # DE types
    de_types = [
        'rand/1/bin',
        'rand/1/exp',
        'rand/2/bin',
        'rand/2/exp'
    ]

    ##
    # Load data
    print("+ Load datasets")
    for data, labels, options in load_data(datasets_data):
        datasets.append((Dataset(data, labels), options))

    print("+ Start tests")
    time_start_test = time()

    for dataset, options in datasets:

        time_start_dataset = time()

        SIZE_W = [dataset.n_features, dataset.n_classes]
        SIZE_B = [dataset.n_classes]
        SIZE_X = [dataset.n_features]

        ##
        # test data collections
        test_results = dict(
            list(zip(de_types, [ENDict(
                [
                    ('values', []),
                    ('last_accuracy', 0.0)
                ]
            ) for _ in range(len(de_types))]))
        )

        prev_NN = dict(
            list(zip(de_types, [None for _ in range(len(de_types))]))
        )

        print("+ Batch for dataset: {}".format(options.name))
        for batch_num, (cur_data, cur_label) in enumerate(dataset.batch(options.BATCH)):
            print("+ Batch[{}]".format(batch_num + 1))

            graph = tf.Graph()
            with graph.as_default():

                res_nn = gen_network(
                    [
                        # 1 lvl
                        #(SIZE_W, SIZE_B)

                        # 2 lvl
                        # ([4, 8], [8]),
                        # ([8, 3], [3])

                        # 3 lvl
                        # ([4, 8], [8]),
                        # ([8, 8], [8]),
                        # ([8, 3], [3])

                        # 5 lvl
                        ([4, 8], [8]),
                        ([8, 8], [8]),
                        ([8, 8], [8]),
                        ([8, 8], [8]),
                        ([8, 3], [3])
                    ],
                    options,
                    cur_data,
                    cur_label,
                    dataset.test_data,
                    dataset.test_labels
                )

                with tf.Session() as sess:
                    # init vars
                    sess.run(tf.global_variables_initializer())

                    ##
                    # Random initialization of the NN
                    cur_pop = sess.run(res_nn.rand_pop)

                    for de_type in de_types:
                        ##
                        # DENN op
                        denn_op_first = DENN.create(  # input params
                            [1, True],  # [num_gen, eval_individual]
                            [],  # FIRST EVAL
                            res_nn.weights,  # PASS WEIGHTS
                            res_nn.populations,  # POPULATIONS
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                sess.graph.as_graph_def()),
                            names=[elm.name for elm in res_nn.targets],
                            CR=options.CR,
                            DE=de_type
                        )

                        denn_op_after = DENN.create(  # input params
                            [1, False],  # [num_gen, eval_individual]
                            res_nn.evaluated,  # FIRST EVAL
                            res_nn.weights,  # PASS WEIGHTS
                            res_nn.populations,  # POPULATIONS
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                sess.graph.as_graph_def()),
                            names=[elm.name for elm in res_nn.targets],
                            CR=options.CR,
                            DE=de_type
                        )

                        if prev_NN[de_type] is not None:
                            cur_pop = prev_NN[de_type]

                        time_start = time()
                        time_start_gen = time()
                        first_time = True

                        for gen in range(options.GEN):
                            if time() - time_start >= 1.:
                                time_start = time()
                                print(
                                    "+ Run gen. [{}] with DE [{}] on {}".format(gen + 1, de_type, options.name), end="\r")

                            if first_time:
                                results = sess.run(denn_op_first, feed_dict=dict(
                                    [
                                        (pop_ref, cur_pop[num])
                                        for num, pop_ref in enumerate(res_nn.populations)
                                    ]
                                ))
                                first_time = False
                            else:
                                results = sess.run(denn_op_after, feed_dict=dict(
                                    [
                                        (pop_ref, cur_pop[num])
                                        for num, pop_ref in enumerate(res_nn.populations)
                                    ]
                                    +
                                    [
                                        (res_nn.evaluated, v_res)
                                    ]
                                ))

                            # get output
                            cur_pop = results.final_populations
                            v_res = results.final_eval

                            best_idx = np.argmin(v_res)

                            cur_accuracy = sess.run(res_nn.accuracy, feed_dict=dict(
                                [
                                    (target, cur_pop[num][best_idx])
                                    for num, target in enumerate(res_nn.targets)
                                ]
                            ))

                            if gen in EVALUATE_STEPS:
                                test_results[de_type].values.append(
                                    cur_accuracy)
                                test_results[
                                    de_type].last_accuracy = cur_accuracy
                            else:
                                test_results[de_type].values.append(
                                    test_results[de_type].last_accuracy
                                )

                        print(
                            "+ DENN[{}] with {} gen on {} completed in {:.05} sec.!".format(de_type, gen + 1, options.name, time() - time_start_gen))

                        prev_NN[de_type] = cur_pop

        print("+ Completed all test on dataset {} in {} sec.".format(options.name,
                                                                     time() - time_start_dataset))
        print("+ Save results for {}".format(options.name))

        description = "NP: {}  W: {}  CR: {}".format(
            options.NP,
            options.W,
            options.CR
        )
        write_results(options.name, test_results, description)
        write_results(options.name, test_results, description, separated=True)

    print("+ Completed all test {} sec.".format(time() - time_start_test))


if __name__ == '__main__':
    main()
