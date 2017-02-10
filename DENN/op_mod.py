import tensorflow as tf
import numpy as np
import signal
from os import path
from time import time
from tqdm import tqdm
from . training import precision_recall_acc
from . training import calc_TF
from . training import calc_confusin_M
from . training import task_dumps
from . training import Dataset
from . utils import get_graph_proto
from . utils import OpListener
from types import MethodType


__all__ = ['Operation']

OUT_FOLDER = "benchmark_results"


def adaboost_run(self, sess, prev_NN, test_results, options={}):
    ##
    # get options
    test_networks = options.get('test_networks')
    TEST_PARALLEL_OP = options.get('TEST_PARALLEL_OP')
    TEST_PARTIAL = options.get('TEST_PARTIAL')

    batch_counter = 0
    v_res = [0.0 for _ in range(self.job.NP)]

    start_evolution = time()

    print("+ Start evolution")

    pbar = tqdm(total=self.job.TOT_GEN)

    for cur_gen in range(int(self.job.TOT_GEN / self.job.GEN_STEP)):

        gen = int(self.job.GEN_STEP / self.job.GEN_SAMPLES)
        cur_batch = self.dataset[batch_counter]
        batch_id = batch_counter
        batch_counter = (batch_counter + 1) % self.dataset.num_batches

        for sample in range(self.job.GEN_SAMPLES):

            ##
            # Last iteration of odd division
            if sample == self.job.GEN_SAMPLES - 1:
                gen += (self.job.GEN_STEP % self.job.GEN_SAMPLES)

            ##
            # Get C, EC, Y of last iteration
            ada_C, ada_EC, ada_pop_y, first_time = self.job.get_adaboost_C(
                batch_id,
                cur_batch
            )

            # print(
            #     "+ Start gen. [{}] with batch[{}]".format((gen + 1) * job.GEN_STEP, batch_counter))

            time_start_gen = time()

            results = sess.run(self.denn_op, feed_dict=dict(
                [
                    (pop_ref, prev_NN[self.de_type][num])
                    for num, pop_ref in enumerate(self.net.populations)
                ]
                +
                [
                    (self.net.label_placeholder, cur_batch.labels),
                    (self.net.input_placeholder, cur_batch.data)
                ]
                +
                [
                    (self.net.cur_gen_options, [self.job.GEN_STEP, first_time])
                ]
                +
                [
                    (self.net.ada_C_placeholder, ada_C),
                    (self.net.ada_EC_placeholder, ada_EC),
                    (self.net.population_y_placeholder, ada_pop_y)
                ]
            ))

            # print("++ Op time {}".format(time() - time_start_gen))

            ##
            # get output
            cur_pop = results.final_populations

            # print(results.final_c)
            # print(results.final_ec)

            ##
            # Update adaboost cache
            self.job.set_adaboost_C(
                batch_id,
                results.final_c,
                results.final_ec,
                results.final_pop_y
            )

            # print(len(cur_pop))
            # print(cur_pop[0].shape)
            # print(cur_pop[1].shape)

            evaluations = []

            time_valutation = time()

            for idx in range(self.job.NP):
                cur_evaluation = sess.run(self.net.accuracy, feed_dict=dict(
                    [
                        (target, cur_pop[num][idx])
                        for num, target in enumerate(self.net.targets)
                    ]
                    +
                    [
                        (self.net.label_placeholder,
                         self.dataset.validation_labels),
                        (self.net.input_placeholder, 
                         self.dataset.validation_data)
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
                        network.label_placeholder] = self.dataset.validation_labels
                    feed_dict_ops[
                        network.input_placeholder] = self.dataset.validation_data

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

            cur_accuracy = sess.run(self.net.accuracy, feed_dict=dict(
                [
                    (target, cur_pop[num][best_idx])
                    for num, target in enumerate(self.net.targets)
                ]
                +
                [
                    (self.net.label_placeholder, self.dataset.test_labels),
                    (self.net.input_placeholder, self.dataset.test_data)
                ]
            ))

            test_results[self.de_type].values.append(cur_accuracy)

            # print("++ Test {}".format(time() - time_test))

            # print(
            #     "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
            #         self.de_type, (gen + 1) * job.GEN_STEP,
            #         job.name,
            #         time() - time_start_gen
            #     )
            # )

            prev_NN[self.de_type] = cur_pop

            pbar.update(gen)

    best = [
        cur_pop[num][best_idx]
        for num, target in enumerate(self.net.targets)
    ]

    self.job.times[self.de_type] = time() - start_evolution
    self.job.accuracy[self.de_type] = cur_accuracy
    self.job.best[self.de_type] = best

    result = sess.run(self.net.y, feed_dict=dict(
        [
            (target, best[num])
            for num, target in enumerate(self.net.targets)
        ]
        +
        [
            (self.net.label_placeholder, self.dataset.test_labels),
            (self.net.input_placeholder, self.dataset.test_data)
        ]
    ))

    self.job.confusionM[self.de_type] = calc_confusin_M(
        self.dataset.test_labels, result)

    for class_ in range(self.job.confusionM[self.de_type][0].shape[0]):
        elm_tf = calc_TF(
            self.job.confusionM[self.de_type], class_)
        if self.de_type not in self.job.stats:
            self.job.stats[self.de_type] = []
        self.job.stats[self.de_type].append(precision_recall_acc(elm_tf))

    pbar.close()


def training_run(self, sess, prev_NN, test_results, options={}):
    start_job = options.get('start_job')

    cur_pop = prev_NN[self.de_type]

    # Soket listener
    with OpListener(tot_steps=self.job.TOT_GEN) as listener:
        ##
        # Handle SIGINT
        def my_handler(signal, frame):
            listener.interrupt()
        signal.signal(signal.SIGINT, my_handler)

        # time
        time_start_gen = time()
        # session run
        current_result = sess.run(self.denn_op, feed_dict=dict(
            [
                (pop_ref, cur_pop[num])
                for num, pop_ref in enumerate(self.net.populations)
            ]
        ))

        run_time = time() - time_start_gen
        print("++ Op time {}".format(run_time))

        # test time
        time_test = time()
        # test result
        cur_accuracy = sess.run(self.net.accuracy, feed_dict=dict(
            [
                (target, current_result[num])
                for num, target in enumerate(self.net.targets)
            ]
            +
            [
                (self.net.label_placeholder, self.dataset.test_labels),
                (self.net.input_placeholder, self.dataset.test_data)
            ]
        ))
        test_time = time() - time_test
        print(
            "++ Test {}, result {}".format(test_time, cur_accuracy))

        result = sess.run(self.net.y, feed_dict=dict(
            [
                (target, current_result[num])
                for num, target in enumerate(self.net.targets)
            ]
            +
            [
                (self.net.label_placeholder, self.dataset.test_labels),
                (self.net.input_placeholder, self.dataset.test_data)
            ]
        ))

        self.job.confusionM[self.de_type] = calc_confusin_M(
            self.dataset.test_labels, result)

        for class_ in range(self.job.confusionM[self.de_type][0].shape[0]):
            elm_tf = calc_TF(self.job.confusionM[self.de_type], class_)
            if self.de_type not in self.job.stats:
                self.job.stats[self.de_type] = []
            self.job.stats[self.de_type].append(precision_recall_acc(elm_tf))

        self.job.times[self.de_type] = run_time + test_time
        self.job.accuracy[self.de_type] = cur_accuracy
        self.job.best[self.de_type] = current_result

        ##
        # Save for any DE (Safe)
        out_file = "{}.json".format(self.job.name)
        self.job.time = time() - start_job
        with open(path.join(OUT_FOLDER, out_file), "w") as out_file:
            out_file.write(task_dumps(self.job))


def standard_run(self, sess, prev_NN, test_results, options={}):
    ##
    # get options
    test_networks = options.get('test_networks')
    TEST_PARALLEL_OP = options.get('TEST_PARALLEL_OP')
    TEST_PARTIAL = options.get('TEST_PARTIAL')

    first_time = True
    batch_counter = 0
    v_res = [0.0 for _ in range(self.job.NP)]

    start_evolution = time()

    print("+ Start evolution")

    pbar = tqdm(total=self.job.TOT_GEN)

    for cur_gen in range(int(self.job.TOT_GEN / self.job.GEN_STEP)):

        gen = int(self.job.GEN_STEP / self.job.GEN_SAMPLES)

        cur_batch = self.dataset[batch_counter]
        batch_counter = (
            batch_counter + 1) % self.dataset.num_batches

        for sample in range(self.job.GEN_SAMPLES):

            ##
            # Last iteration of odd division
            if sample == self.job.GEN_SAMPLES - 1:
                gen += (self.job.GEN_STEP % self.job.GEN_SAMPLES)

            # print(
            #     "+ Start gen. [{}] with batch[{}]".format((gen + 1) * job.GEN_STEP, batch_counter))

            time_start_gen = time()

            results = sess.run(self.denn_op, feed_dict=dict(
                [
                    (pop_ref, prev_NN[self.de_type][num])
                    for num, pop_ref in enumerate(self.net.populations)
                ]
                +
                [
                    (self.net.label_placeholder, cur_batch.labels),
                    (self.net.input_placeholder, cur_batch.data)
                ]
                +
                [
                    (self.net.evaluated, v_res)
                ]
                +
                [
                    (self.net.cur_gen_options, [
                        self.job.GEN_STEP, first_time])
                ]
            ))

            # print("++ Op time {}".format(time() - time_start_gen))

            # get output
            cur_pop = results.final_populations
            v_res = results.final_eval

            # print(len(cur_pop))
            # print(cur_pop[0].shape)
            # print(cur_pop[1].shape)

            evaluations = []

            time_valutation = time()

            for idx in range(self.job.NP):
                cur_evaluation = sess.run(self.net.accuracy, feed_dict=dict(
                    [
                        (target, cur_pop[num][idx])
                        for num, target in enumerate(self.net.targets)
                    ]
                    +
                    [
                        (self.net.label_placeholder,
                         self.dataset.validation_labels),
                        (self.net.input_placeholder, self.dataset.validation_data)
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
                        network.label_placeholder] = self.dataset.validation_labels
                    feed_dict_ops[
                        network.input_placeholder] = self.dataset.validation_data

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

            cur_accuracy = sess.run(self.net.accuracy, feed_dict=dict(
                [
                    (target, cur_pop[num][best_idx])
                    for num, target in enumerate(self.net.targets)
                ]
                +
                [
                    (self.net.label_placeholder, self.dataset.test_labels),
                    (self.net.input_placeholder, self.dataset.test_data)
                ]
            ))

            test_results[self.de_type].values.append(cur_accuracy)

            # print("++ Test {}".format(time() - time_test))

            # print(
            #     "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
            #         self.de_type, (gen + 1) * job.GEN_STEP,
            #         job.name,
            #         time() - time_start_gen
            #     )
            # )

            first_time = False

            prev_NN[self.de_type] = cur_pop

            pbar.update(gen)

    best = [
        cur_pop[num][best_idx]
        for num, target in enumerate(self.net.targets)
    ]

    self.job.times[self.de_type] = time() - start_evolution
    self.job.accuracy[self.de_type] = cur_accuracy
    self.job.best[self.de_type] = best

    result = sess.run(self.net.y, feed_dict=dict(
        [
            (target, best[num])
            for num, target in enumerate(self.net.targets)
        ]
        +
        [
            (self.net.label_placeholder, self.dataset.test_labels),
            (self.net.input_placeholder, self.dataset.test_data)
        ]
    ))

    self.job.confusionM[self.de_type] = calc_confusin_M(
        self.dataset.test_labels, result)

    for class_ in range(self.job.confusionM[self.de_type][0].shape[0]):
        elm_tf = calc_TF(
            self.job.confusionM[self.de_type], class_)
        if self.de_type not in self.job.stats:
            self.job.stats[self.de_type] = []
        self.job.stats[self.de_type].append(precision_recall_acc(elm_tf))

    pbar.close()


class Operation(object):
    """Create a DENN object."""

    def __init__(self, dataset, job, net, de_type):
        self.dataset = dataset
        self.job = job
        self.net = net
        self.de_type = de_type
        self._module = None

        if job.ada_boost is not None:
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOp_ada.so')
            )
            self.denn_op = self._module.denn(
                # input params
                # [num_gen, eval_individual]
                self.net.cur_gen_options,
                self.net.label_placeholder,
                self.net.input_placeholder,
                self.net.weights,  # PASS WEIGHTS
                self.net.populations,  # POPULATIONS
                self.net.ada_C_placeholder,
                self.net.ada_EC_placeholder,
                self.net.population_y_placeholder,
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                f_name_execute_net=self.net.nn_exec.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                f_input_correct_predition=self.net.y_placeholder.name,
                f_correct_predition=self.net.ada_label_diff.name,
                f_cross_entropy=self.net.cross_entropy.name,
                f_input_cross_entropy=self.net.y_placeholder.name,
                ada_boost_alpha=self.job.ada_boost.alpha,
                CR=self.job.CR,
                DE=self.de_type,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max
            )
            ##
            # add AdaBoost run
            self.run = MethodType(adaboost_run, self)
        elif job.training:
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOp_training.so')
            )
            self.denn_op = self._module.denn(
                # input params
                # [num_gen, step_gen, eval_individual]
                [self.job.TOT_GEN, self.job.GEN_STEP, False],
                np.array([], dtype=np.float64 if self.job.TYPE ==
                         "double" else np.float32),  # FIRST EVAL
                self.net.weights,  # PASS WEIGHTS
                self.net.populations,  # POPULATIONS
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                dataset=self.job.dataset_file,
                f_name_execute_net=self.net.cross_entropy.name,
                f_name_validation=self.net.accuracy.name,
                f_name_test=self.net.accuracy.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                CR=self.job.CR,
                DE=de_type,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max
            )
            ##
            # add training run
            self.run = MethodType(training_run, self)
        else:
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOp.so')
            )
            self.denn_op = self._module.denn(
                # input params
                # [num_gen, eval_individual]
                self.net.cur_gen_options,
                self.net.label_placeholder,
                self.net.input_placeholder,
                self.net.evaluated,  # FIRST EVAL
                self.net.weights,  # PASS WEIGHTS
                self.net.populations,  # POPULATIONS
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                f_name_execute_net=self.net.cross_entropy.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max,
                CR=self.job.CR,
                DE=self.de_type
            )
            ##
            # add standard run
            self.run = MethodType(standard_run, self)
