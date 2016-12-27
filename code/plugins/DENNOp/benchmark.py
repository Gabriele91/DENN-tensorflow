import DENN
import tensorflow as tf

import numpy as np
import json

from sys import argv
from time import sleep
from time import time

#from memory_profiler import profile

# sleep(6)


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class NDict(dict):
    pass


#@profile
def main():
    ##
    # Select device
    DEVICE = None
    NUM_THREADS = 4

    session_config = tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS,
        inter_op_parallelism_threads=NUM_THREADS,
        log_device_placement=True
    )

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
                ('job', options),
                ('num_batches', dataset.num_batches)
            ]
        )

        prev_NN = dict(
            list(zip(options.de_types, [
                None for _ in range(len(options.de_types))]))
        )

        v_res = [0.0 for _ in range(options.NP)]

        batch_counter = 0

        cur_nn = DENN.training.gen_network(
            options,
            True,  # rand population only if gen is the first one
            type_=options.TYPE
        )

        with cur_nn.graph.as_default():
            with tf.Session(config=session_config) as sess:

                denn_operators = {}

                time_node_creation = time()

                for de_type in options.de_types:

                    ##
                    # Random initialization of the NN
                    cur_pop = sess.run(cur_nn.rand_pop)
                    prev_NN[de_type] = cur_pop

                    with tf.device("/cpu:0"):
                        ##
                        # DENN op
                        denn_operators[de_type] = DENN.create(
                            # input params
                            # [num_gen, eval_individual]
                            cur_nn.cur_gen_options,
                            cur_nn.label_placeholder,
                            cur_nn.input_placeholder,
                            cur_nn.evaluated,  # FIRST EVAL
                            cur_nn.weights,  # PASS WEIGHTS
                            cur_nn.populations,  # POPULATIONS
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                cur_nn.graph.as_graph_def()),
                            f_name_train=cur_nn.cross_entropy.name,
                            f_inputs=[elm.name for elm in cur_nn.targets],
                            f_input_labels=cur_nn.label_placeholder.name,
                            f_input_features=cur_nn.input_placeholder.name,
                            CR=options.CR,
                            DE=de_type,
                            # training=True
                        )

                print(
                    "++ Node creation {}".format(time() - time_node_creation))

                for de_type, denn_op in denn_operators.items():

                    first_time = True

                    for gen in range(int(options.TOT_GEN / options.GEN_STEP)):

                        print(
                            "+ Start gen. [{}] with batch[{}]".format((gen + 1) * options.GEN_STEP, batch_counter))

                        cur_batch = dataset[batch_counter]
                        batch_counter = (
                            batch_counter + 1) % dataset.num_batches

                        time_start_gen = time()

                        results = sess.run(denn_op, feed_dict=dict(
                            [
                                (pop_ref, prev_NN[de_type][num])
                                for num, pop_ref in enumerate(cur_nn.populations)
                            ]
                            +
                            [
                                (cur_nn.label_placeholder, cur_batch.labels),
                                (cur_nn.input_placeholder, cur_batch.data)
                            ]
                            +
                            [
                                (cur_nn.evaluated, v_res)
                            ]
                            +
                            [
                                (cur_nn.cur_gen_options, [
                                 options.GEN_STEP, first_time])
                            ]
                        ))

                        print("++ Op time {}".format(time() - time_start_gen))

                        # get output
                        cur_pop = results.final_populations
                        v_res = results.final_eval

                        # print(len(cur_pop))
                        # print(cur_pop[0].shape)
                        # print(cur_pop[1].shape)

                        evaluations = []

                        time_valutation = time()

                        for idx in range(options.NP):
                            cur_evaluation = sess.run(cur_nn.accuracy, feed_dict=dict(
                                [
                                    (target, cur_pop[num][idx])
                                    for num, target in enumerate(cur_nn.targets)
                                ]
                                +
                                [
                                    (cur_nn.label_placeholder,
                                        dataset.validation_labels),
                                    (cur_nn.input_placeholder,
                                        dataset.validation_data)
                                ]
                            ))
                            evaluations.append(cur_evaluation)

                        # print(evaluations)
                        print(
                            "++ Valutation {}".format(time() - time_valutation))

                        best_idx = np.argmin(evaluations)

                        time_test = time()

                        cur_accuracy = sess.run(cur_nn.accuracy, feed_dict=dict(
                            [
                                (target, cur_pop[num][best_idx])
                                for num, target in enumerate(cur_nn.targets)
                            ]
                            +
                            [
                                (cur_nn.label_placeholder, dataset.test_labels),
                                (cur_nn.input_placeholder, dataset.test_data)
                            ]
                        ))

                        test_results[de_type].values.append(cur_accuracy)

                        print("++ Test {}".format(time() - time_test))

                        print(
                            "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
                                de_type, (gen + 1) * options.GEN_STEP,
                                options.name,
                                time() - time_start_gen
                            )
                        )

                        prev_NN[de_type] = cur_pop
                        del cur_pop

                        first_time = False

        print("+ Completed all test on dataset {} in {} sec.".format(options.name,
                                                                     time() - time_start_dataset))
        print("+ Save results for {}".format(options.name))

        description = "NP: {}  W: {}  CR: {}".format(
            options.NP,
            options.F,
            options.CR
        )

        DENN.training.expand_results(
            test_results, options.GEN_STEP, options.de_types)

        DENN.training.write_all_results(
            options.name, test_results, description, out_options)

    print("+ Completed all test {} sec.".format(time() - time_start_test))


if __name__ == '__main__':
    main()
