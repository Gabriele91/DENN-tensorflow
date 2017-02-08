##
# Only for testing
import sys
sys.path.append("../")
#####

import DENN
import tensorflow as tf

import numpy as np

from sys import argv
from time import sleep
from time import time
from tqdm import tqdm

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

    TEST_PARALLEL_OP = False
    TEST_PARTIAL = False
    TEST_ALTERNATE_DEV = False

    ##
    # jobs
    jobs = []

    ##
    # datasets
    datasets = []

    jobs = DENN.training.open_task_list(argv[1])

    ##
    # Load data
    print("+ Load datasets")
    for job in jobs:
        datasets.append((DENN.training.Dataset(job.dataset_file), job))

    print("+ Start tests")
    time_start_test = time()

    for dataset, job in datasets:

        # print(job)

        time_start_dataset = time()

        ##
        # test data collections
        test_results = NDict(
            list(
                zip(job.de_types, [
                    ENDict(
                        [
                            ('values', []),
                        ]
                    ) for _ in range(len(job.de_types))
                ])
            )
        )

        out_options = ENDict(
            [
                ('job', job),
                ('num_batches', dataset.num_batches)
            ]
        )

        prev_NN = dict(
            list(zip(job.de_types, [
                None for _ in range(len(job.de_types))]))
        )

        v_res = [0.0 for _ in range(job.NP)]

        batch_counter = 0

        cur_nn = job.gen_network(
            True
        )

        if TEST_PARALLEL_OP:
            test_networks = []
            for num in range(job.NP):
                if TEST_ALTERNATE_DEV:
                    if num % 2 == 0:
                        job.levels[0].preferred_device = "CPU"
                    else:
                        job.levels[0].preferred_device = "GPU"
                test_networks.append(job.gen_network(True))

        with cur_nn.graph.as_default():

            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=job.num_intra_threads,
                inter_op_parallelism_threads=job.num_inter_threads,
                # log_device_placement=True
                log_device_placement=False
            )

            with tf.Session(config=session_config) as sess:

                denn_operators = {}

                time_node_creation = time()

                for de_type in job.de_types:

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
                            cur_nn.weights,  # PASS WEIGHTS
                            cur_nn.populations,  # POPULATIONS
                            cur_nn.ada_C_placeholder,
                            cur_nn.ada_EC_placeholder,
                            cur_nn.population_y_placeholder,
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                cur_nn.graph.as_graph_def()),
                            f_name_execute_net=cur_nn.nn_exec.name,
                            f_inputs=[elm.name for elm in cur_nn.targets],
                            f_input_labels=cur_nn.label_placeholder.name,
                            f_input_features=cur_nn.input_placeholder.name,
                            f_input_correct_predition=cur_nn.y_placeholder.name,
                            f_correct_predition=cur_nn.ada_label_diff.name,
                            f_cross_entropy=cur_nn.cross_entropy.name,
                            f_input_cross_entropy=cur_nn.y_placeholder.name,
                            ada_boost_alpha=job.ada_boost.alpha,
                            CR=job.CR,
                            DE=de_type,
                            f_min=job.clamp.min,
                            f_max=job.clamp.max
                        ).ada

                print(
                    "++ Node creation {}".format(time() - time_node_creation))

                for de_type, denn_op in denn_operators.items():

                    first_time = True

                    start_evolution = time()

                    print("+ Start evolution")

                    pbar = tqdm(total=job.TOT_GEN)

                    for cur_gen in range(int(job.TOT_GEN / job.GEN_STEP)):

                        gen = int(job.GEN_STEP / job.GEN_SAMPLES)
                        cur_batch = dataset[batch_counter]
                        ada_C, ada_EC, ada_pop_y = job.get_adaboost_C(
                            batch_counter, cur_batch
                        )

                        batch_counter = (
                            batch_counter + 1) % dataset.num_batches

                        for sample in range(job.GEN_SAMPLES):
                            ##
                            # Last iteration of odd division
                            if sample == job.GEN_SAMPLES - 1:
                                gen += (job.GEN_STEP % job.GEN_SAMPLES)

                            # print(
                            #     "+ Start gen. [{}] with batch[{}]".format((gen + 1) * job.GEN_STEP, batch_counter))
                            time_start_gen = time()

                            # print(ada_C)
                            # print(ada_EC)

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
                                    (cur_nn.cur_gen_options, [
                                        job.GEN_STEP, first_time])
                                ]
                                +
                                [
                                    (cur_nn.ada_C_placeholder, ada_C),
                                    (cur_nn.ada_EC_placeholder, ada_EC),
                                    (cur_nn.population_y_placeholder, ada_pop_y)
                                ]
                            ))

                            # print("++ Op time {}".format(time() - time_start_gen))

                            # get output
                            cur_pop = results.final_populations

                            # print(results.final_c)
                            # print(results.final_ec)

                            job.set_adaboost_C(
                                batch_counter,
                                results.final_c,
                                results.final_ec,
                                results.final_pop_y
                            )

                            # print(len(cur_pop))
                            # print(cur_pop[0].shape)
                            # print(cur_pop[1].shape)

                            evaluations = []

                            time_valutation = time()

                            for idx in range(job.NP):
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
                            # print(
                            #     "++ Valutation:\t\t{}".format(time() - time_valutation))

                            # ---------- TEST PARALLEL EVALUATIONS ----------
                            if TEST_PARALLEL_OP:
                                time_valutation = time()

                                accuracy_ops = []
                                feed_dict_ops = {}

                                for idx, network in enumerate(test_networks):
                                    accuracy_ops.append(network.accuracy)

                                    for num, target in enumerate(network.targets):
                                        feed_dict_ops[
                                            target] = cur_pop[num][idx]

                                    feed_dict_ops[
                                        network.label_placeholder] = dataset.validation_labels
                                    feed_dict_ops[
                                        network.input_placeholder] = dataset.validation_data

                                if not TEST_PARTIAL:
                                    evaluations_test = sess.run(
                                        accuracy_ops, feed_dict=feed_dict_ops)
                                else:
                                    handler = sess.partial_run_setup(
                                        accuracy_ops,
                                        [_ for _ in feed_dict_ops.keys()]
                                    )

                                    evaluations_test = sess.partial_run(
                                        handler,
                                        accuracy_ops,
                                        feed_dict=feed_dict_ops
                                    )

                                # print(evaluations_test)
                                # print(
                                #     "++ Valutation Test:\t{} | equal to eval: {}".format(
                                #         time() - time_valutation,
                                #         np.allclose(evaluations_test, evaluations)
                                #     )
                                # )

                            # ---------- END TEST ----------

                            best_idx = np.argmin(evaluations)

                            time_test = time()

                            cur_accuracy = sess.run(cur_nn.accuracy, feed_dict=dict(
                                [
                                    (target, cur_pop[num][best_idx])
                                    for num, target in enumerate(cur_nn.targets)
                                ]
                                +
                                [
                                    (cur_nn.label_placeholder,
                                     dataset.test_labels),
                                    (cur_nn.input_placeholder, dataset.test_data)
                                ]
                            ))

                            test_results[de_type].values.append(cur_accuracy)

                            # print("++ Test {}".format(time() - time_test))

                            # print(
                            #     "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
                            #         de_type, (gen + 1) * job.GEN_STEP,
                            #         job.name,
                            #         time() - time_start_gen
                            #     )
                            # )

                            first_time = False

                            prev_NN[de_type] = cur_pop

                            pbar.update(gen)

                    best = [
                        cur_pop[num][best_idx]
                        for num, target in enumerate(cur_nn.targets)
                    ]

                    job.times[de_type] = time() - start_evolution
                    job.accuracy[de_type] = cur_accuracy
                    job.best[de_type] = best

                    result = sess.run(cur_nn.y, feed_dict=dict(
                        [
                            (target, best[num])
                            for num, target in enumerate(cur_nn.targets)
                        ]
                        +
                        [
                            (cur_nn.label_placeholder,
                                dataset.test_labels),
                            (cur_nn.input_placeholder, dataset.test_data)
                        ]
                    ))

                    job.confusionM[de_type] = DENN.training.calc_confusin_M(
                        dataset.test_labels, result)

                    for class_ in range(job.confusionM[de_type][0].shape[0]):
                        elm_tf = DENN.training.calc_TF(
                            job.confusionM[de_type], class_)
                        if de_type not in job.stats:
                            job.stats[de_type] = []
                        job.stats[de_type].append(
                            DENN.training.precision_recall_acc(elm_tf))

                    pbar.close()

        print("+ Completed all test on dataset {} in {} sec.".format(job.name,
                                                                     time() - time_start_dataset))
        print("+ Save results for {}".format(job.name))

        description = "NP: {}  F: {}  CR: {}".format(
            job.NP,
            job.F,
            job.CR
        )

        job.time = time() - time_start_test

        DENN.training.export_results(test_results, int(
            job.GEN_STEP / job.GEN_SAMPLES),
            job.name, out_options
        )

        DENN.training.expand_results(
            test_results, int(job.GEN_STEP / job.GEN_SAMPLES), job.de_types)

        DENN.training.write_all_results(
            job.name, test_results, description, out_options)

        tf.reset_default_graph()

    print("+ Completed all test {} sec.".format(time() - time_start_test))


if __name__ == '__main__':
    main()
