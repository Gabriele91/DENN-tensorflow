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


def main():
    ##
    # jobs
    jobs = []

    ##
    # datasets
    datasets = []

    with open(argv[1], 'r') as job_f:
        jobs = json.load(job_f)

    for idx in range(len(jobs)):
        jobs[idx] = ENDict(jobs[idx].items())

    ##
    # Load data
    print("+ Load datasets")
    for job in jobs:
        datasets.append((DENN.training.Dataset(job.dataset_file), job))

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
                ('num_gen', options.TOT_GEN),
                ('levels', NN_LEVELS)
            ]
        )

        prev_NN = dict(
            list(zip(options.de_types, [
                 None for _ in range(len(options.de_types))]))
        )

        batch_counter = 0

        for gen in range(int(options.TOT_GEN / options.GEN_STEP)):

            print(
                "+ Start gen. [{}] with batch[{}]".format((gen + 1) * options.GEN_STEP, batch_counter))

            graph = tf.Graph()
            with graph.as_default():

                cur_batch = dataset[batch_counter]
                batch_counter = (batch_counter + 1) % dataset.num_batches

                cur_nn = DENN.training.gen_network(
                    NN_LEVELS,
                    options,
                    cur_batch.data,
                    cur_batch.labels,
                    dataset.test_data,
                    dataset.test_labels
                )

                with tf.Session() as sess:
                    # init vars
                    sess.run(tf.global_variables_initializer())

                    ##
                    # Random initialization of the NN
                    cur_pop = sess.run(cur_nn.rand_pop)

                    for de_type in options.de_types:
                        ##
                        # DENN op
                        denn_op = DENN.create(  # input params
                            # [num_gen, eval_individual]
                            [options.GEN_STEP, gen == 0],
                            [] if gen == 0 else cur_nn.evaluated,  # FIRST EVAL
                            cur_nn.weights,  # PASS WEIGHTS
                            cur_nn.populations,  # POPULATIONS
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                sess.graph.as_graph_def()),
                            names=[elm.name for elm in cur_nn.targets],
                            CR=options.CR,
                            DE=de_type
                        )

                        if prev_NN[de_type] is not None:
                            cur_pop = prev_NN[de_type]

                        time_start_gen = time()

                        if gen == 0:
                            results = sess.run(denn_op, feed_dict=dict(
                                [
                                    (pop_ref, cur_pop[num])
                                    for num, pop_ref in enumerate(cur_nn.populations)
                                ]
                            ))
                        else:
                            results = sess.run(denn_op, feed_dict=dict(
                                [
                                    (pop_ref, cur_pop[num])
                                    for num, pop_ref in enumerate(cur_nn.populations)
                                ]
                                +
                                [
                                    (cur_nn.evaluated, v_res)
                                ]
                            ))

                        # get output
                        cur_pop = results.final_populations
                        v_res = results.final_eval

                        best_idx = np.argmin(v_res)

                        cur_accuracy = sess.run(cur_nn.accuracy, feed_dict=dict(
                            [
                                (target, cur_pop[num][best_idx])
                                for num, target in enumerate(cur_nn.targets)
                            ]
                        ))

                        test_results[de_type].values.append(cur_accuracy)

                        print(
                            "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
                                de_type, (gen + 1) * options.GEN_STEP,
                                options.name,
                                time() - time_start_gen
                            )
                        )

                        prev_NN[de_type] = cur_pop

            tf.reset_default_graph()

        print("+ Completed all test on dataset {} in {} sec.".format(options.name,
                                                                     time() - time_start_dataset))
        print("+ Save results for {}".format(options.name))

        description = "NP: {}  W: {}  CR: {}".format(
            options.NP,
            options.W,
            options.CR
        )

        DENN.training.expand_results(
            test_results, options.GEN_STEP, options.de_types)

        DENN.training.write_all_results(
            options.name, test_results, description, out_options)

    print("+ Completed all test {} sec.".format(time() - time_start_test))


if __name__ == '__main__':
    main()
