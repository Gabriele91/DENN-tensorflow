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
    BATCH_SIZE = 40
    EVALUATE_STEPS = [
        0, 50, 100#, 200, 400, 500
    ]

    datasets_data = [
        (
            "../../../minimal_dataset/data/bezdekIris",
            'load_iris_data',
            ENDict(
                [
                    ('name', 'iris_dataset'),
                    ('GEN', 100),
                    ('NP', 100),
                    ('BATCH', 10),
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
        # 'rand/1/exp',
        # 'rand/2/bin',
        # 'rand/2/exp'
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
        # DE W -> NN (W, B)
        deW_nnW = np.full(SIZE_W, options.W)
        deW_nnB = np.full(SIZE_B, options.W)

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
        for batch_num, (cur_data, cur_label) in enumerate(dataset.batch(BATCH_SIZE)):
            print("+ Batch[{}]".format(batch_num+1))

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
                target_w = tf.placeholder(tf.float64, SIZE_W)
                target_b = tf.placeholder(tf.float64, SIZE_B)

                cur_pop_W = tf.placeholder(tf.float64, [options.NP] + SIZE_W)
                cur_pop_B = tf.placeholder(tf.float64, [options.NP] + SIZE_B)
                cur_pop_VAL = tf.placeholder(tf.float64, [options.NP])

                y_test = tf.matmul(dataset.test_data, target_w) + target_b
                correct_prediction = tf.equal(
                    tf.argmax(y_test, 1),
                    tf.argmax(dataset.test_labels, 1)
                )
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                ##
                # NN
                y = tf.matmul(cur_data, target_w) + target_b
                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        y, cur_label), name="evaluate")

                with tf.Session() as sess:
                    # init vars
                    sess.run(tf.global_variables_initializer())

                    ##
                    # Random initialization of the NN
                    w_res, b_res = sess.run([
                        create_random_population_W,
                        create_random_population_B
                    ])

                    for de_type in de_types:
                        ##
                        # DENN op
                        denn_op = DENN.create(  # input params
                            [1, True],  # [num_gen, eval_individual]
                            [],  # FIRST EVAL
                            [deW_nnW, deW_nnB],  # PASS WEIGHTS
                            [cur_pop_W, cur_pop_B],  # POPULATIONS
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                sess.graph.as_graph_def()),
                            names=[elm.name for elm in [target_w, target_b]],
                            CR=options.CR,
                            DE=de_type
                        )

                        if prev_NN[de_type] is not None:
                            w_res, b_res = prev_NN[de_type]

                        time_start = time()
                        time_start_gen = time()

                        for gen in range(options.GEN):
                            if time() - time_start >= 1.:
                                time_start = time()
                                print(
                                    "+ Run gen. [{}] with DE [{}] on {}".format(gen + 1, de_type, options.name), end="\r")

                            results = sess.run(denn_op, feed_dict={
                                cur_pop_W: w_res,
                                cur_pop_B: b_res
                            })
                            # get output
                            w_res, b_res = results.final_populations
                            v_res = results.final_eval

                            best_idx = np.argmin(v_res)

                            cur_accuracy = sess.run(accuracy, feed_dict={
                                target_w: w_res[best_idx],
                                target_b: b_res[best_idx]
                            })

                            if gen in EVALUATE_STEPS:
                                test_results[de_type].values.append(cur_accuracy)
                                test_results[de_type].last_accuracy = cur_accuracy
                            else:
                                test_results[de_type].values.append(
                                    test_results[de_type].last_accuracy
                                )

                        print(
                            "+ DENN[{}] with {} gen on {} completed in {:.05} sec.!".format(de_type, gen + 1, options.name, time() - time_start_gen))
                        
                        prev_NN[de_type] = (w_res, b_res)
        
        print("+ Completed all test on dataset {} in {} sec.".format(options.name, time() - time_start_dataset))
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
