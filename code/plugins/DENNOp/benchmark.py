import DENN
import tensorflow as tf
import numpy as np
import dataset_loaders
import json

from sys import argv
from time import sleep
from time import time

# sleep(6)


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class NDict(dict):
    pass


def load_data(datasets_data, debug=False):
    for path_, loader, options in datasets_data:
        yield (*getattr(dataset_loaders, loader)(path_, False), options)


def main():
    ##
    # PRINT OUTPUT
    LOADER = ['.  ', '.. ', '...']
    LOADER_COUNTER = 0

    ##
    # Datasets
    datasets = []

    with open(argv[1], 'r') as dataset_pf:
        datasets_data = json.load(dataset_pf)

    for config in datasets_data:
        config[2] = ENDict(config[2].items())

    ##
    # Load data
    print("+ Load datasets")
    for data, labels, options in load_data(datasets_data):
        datasets.append((DENN.training.Dataset(data, labels), options))

    print("+ Start tests")
    time_start_test = time()

    for dataset, options in datasets:

        time_start_dataset = time()

        NN_LEVELS = options.levels

        ##
        # test data collections
        test_results = NDict(
            list(
                zip(options.de_types, [
                    ENDict(
                        [
                            ('values', []),
                        ]
                    ) for _ in range(len(options.de_types))
                ])
            )
        )

        out_options = ENDict(
            [
                ('num_gen', options.N_GEN * options.GEN_STEP),
                ('levels', NN_LEVELS)
            ]
        )

        prev_NN = dict(
            list(zip(options.de_types, [None for _ in range(len(options.de_types))]))
        )

        print("+ Batch for dataset: {}".format(options.name))
        for batch_num, (cur_data, cur_label) in enumerate(dataset.batch(options.BATCH)):
            print("+ Batch[{}]".format(batch_num + 1))

            graph = tf.Graph()
            with graph.as_default():

                res_nn = DENN.training.gen_network(
                    NN_LEVELS,
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

                    for de_type in options.de_types:
                        ##
                        # DENN op
                        denn_op_first = DENN.create(  # input params
                            # [num_gen, eval_individual]
                            [options.GEN_STEP, True],
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
                            # [num_gen, eval_individual]
                            [options.GEN_STEP, False],
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

                        for gen in range(options.N_GEN):
                            if time() - time_start >= 1.:
                                time_start = time()
                                print(
                                    "+ Run gen. [{}] with DE [{}] on {} {}".format(
                                        (gen + 1) * options.GEN_STEP,
                                        de_type,
                                        options.name,
                                        LOADER[LOADER_COUNTER]
                                    ),
                                    end="\r")

                                LOADER_COUNTER += 1
                                LOADER_COUNTER %= len(LOADER)

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

                            for gen_done in range(options.GEN_STEP):
                                test_results[de_type].values.append(
                                    cur_accuracy)

                        print(
                            "+ DENN[{}] with {} gen on {} completed in {:.05} sec.!".format(
                                de_type, (gen + 1) * options.GEN_STEP, options.name, time() - time_start_gen))

                        prev_NN[de_type] = cur_pop

        print("+ Completed all test on dataset {} in {} sec.".format(options.name,
                                                                     time() - time_start_dataset))
        print("+ Save results for {}".format(options.name))

        description = "NP: {}  W: {}  CR: {}".format(
            options.NP,
            options.W,
            options.CR
        )
        DENN.training.write_all_results(
            options.name, test_results, description, out_options)

    print("+ Completed all test {} sec.".format(time() - time_start_test))


if __name__ == '__main__':
    main()
