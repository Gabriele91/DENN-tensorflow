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


__all__ = ['Operation', 'create', 'update_best_of']

OUT_FOLDER = "benchmark_results"


def update_best_of(de_type, test_results, accuracy_test, accuracy_val, new_f, new_cr, new_individual, job):
    """Tracks the best of the whole evolution.

    Params:
        de_type (string): current DE type in use
        test_results (dict): container of all results
        accuracy_test (float): the accuracy of the new best individual of
                              the current population on test set
        accuracy_val (float): the accuracy of the new best individual of
                              the current population on validation set
        new_f (float) : scale factor of new best 
        new_cr (float) : crossover rate of new best 
        new_individual (numpy.ndarray): the current best individual 
                                        weights and biases
    """
    if test_results[de_type].best_of['accuracy_val'][-1] < accuracy_val:
        test_results[de_type].best_of['F'] = new_f
        test_results[de_type].best_of['CR'] = new_cr
        test_results[de_type].best_of['accuracy'].append(accuracy_test)
        test_results[de_type].best_of['accuracy_val'].append(accuracy_val)
        test_results[de_type].best_of['individual'] = new_individual
        job.best['F'] = new_f
        job.best['CR'] = new_cr
        job.best['accuracy'] = accuracy_test
        job.best['individual'] = new_individual
        return True
    else:
        # test
        accuracy_test = test_results[de_type].best_of['accuracy'][-1]
        test_results[de_type].best_of['accuracy'].append(accuracy_test)
        # validation
        accuracy_val = test_results[de_type].best_of['accuracy_val'][-1]
        test_results[de_type].best_of['accuracy_val'].append(accuracy_val)
        return False


class Operation(object):

    """Create a DENN object."""

    def __new__(cls, *args, **kwargs):
        """Binds of the correct run function in base of the job."""
        job = args[1]
        if job.ada_boost is not None and job.training:
            cls.run = cls.training_run
        elif job.ada_boost is not None:
            cls.run = cls.adaboost_run
        elif job.training:
            cls.run = cls.training_run
        else:
            cls.run = cls.standard_run
        new_instance = super(Operation, cls).__new__(cls)
        return new_instance

    def __init__(self, dataset, job, net, de_type):
        """Initializes the operation.

        Params:
            dataset (DENN.training.utils.Dataset): current dataset
            job (DENN.training.task.DETask): the job to do
            net (DENN.training.task.Network): the tensorflow network
            de_type (string): current DE type
        """
        self.dataset = dataset
        self.job = job
        self.net = net
        self.de_type = de_type
        self._module = None

        if job.ada_boost is not None and job.training:
            ###########################################
            # TRAINING DENN ADA BOOST
            ###########################################
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOpAdaTraining.so')
            )

            exists_reset_every = self.job.reset_every != False

            self.denn_op = self._module.denn_ada_training(
                # input params
                # [num_gen, step_gen ]
                [self.job.TOT_GEN, self.job.GEN_STEP],
                # POPULATIONS and F and CR
                self.net.F_placeholder,
                self.net.CR_placeholder,

                self.net.populations,
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                dataset=self.job.dataset_file,
                # NET
                f_name_execute_net=self.net.nn_exec.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                f_input_correct_predition=self.net.y_placeholder.name,
                f_correct_predition=self.net.ada_label_diff.name,
                # CROSS
                f_cross_entropy=self.net.objective_function.name,
                f_input_cross_entropy_y=self.net.y_placeholder.name,
                f_input_cross_entropy_c=self.net.ada_C_placeholder.name,
                # VALIDATION
                f_name_validation=self.net.accuracy.name,
                f_name_test=self.net.accuracy.name,
                # ADA
                ada_boost_alpha=self.job.ada_boost.alpha,
                ada_boost_c=self.job.ada_boost.C,
                ada_reset_c_on_change_bacth=self.job.ada_boost.reset_C_on_change_bacth,
                JDE=self.job.JDE,
                DE=de_type,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max,
                smoothing=self.job.smoothing,
                smoothing_n_pass=self.job.smoothing_n_pass,
                # TRAIN
                reset_type='execute' if exists_reset_every else 'none',
                reset_counter=self.job.reset_every[
                    'counter'] if exists_reset_every else 1,
                reset_f=self.net.F_init.name,
                reset_cr=self.net.CR_init.name,
                reset_rand_pop=[tfop.name for tfop in self.net.rand_pop],
                reinsert_best=self.job.reinsert_best,
                inheritance=self.job.inheritance
            )
        elif job.ada_boost is not None:
            ###########################################
            # DENN ADA BOOST
            ###########################################
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOpADA.so')
            )
            self.denn_op = self._module.denn_ada(
                # input params
                # [num_gen, eval_individual]
                self.net.cur_gen_options,
                self.net.label_placeholder,
                self.net.input_placeholder,
                # POPULATIONS and F and CR
                self.net.F_placeholder,
                self.net.CR_placeholder,
                self.net.populations,
                # ADA
                self.net.ada_C_placeholder,
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                f_name_execute_net=self.net.nn_exec.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                f_input_correct_predition=self.net.y_placeholder.name,
                f_correct_predition=self.net.ada_label_diff.name,
                f_cross_entropy=self.net.objective_function.name,
                f_input_cross_entropy_y=self.net.y_placeholder.name,
                f_input_cross_entropy_c=self.net.ada_C_placeholder.name,
                ada_boost_alpha=self.job.ada_boost.alpha,
                JDE=self.job.JDE,
                DE=self.de_type,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max,
                smoothing=self.job.smoothing,
                smoothing_n_pass=self.job.smoothing_n_pass,
                inheritance=self.job.inheritance
            )
        elif job.training:
            ###########################################
            # TRAINING DENN standard
            ###########################################
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOpTraining.so')
            )

            exists_reset_every = self.job.reset_every != False

            self.denn_op = self._module.denn_training(
                # input params
                # [num_gen, step_gen, eval_individual]
                [self.job.TOT_GEN, self.job.GEN_STEP, False],
                # F and CR
                self.net.F_placeholder,
                self.net.CR_placeholder,
                # FIRST EVAL [] void
                np.array([], dtype=np.float64 if self.job.TYPE ==
                         "double" else np.float32),
                # POPULATIONS
                self.net.populations,
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                dataset=self.job.dataset_file,
                # NET
                f_name_execute_net=self.net.objective_function.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                # VALIDATION
                f_name_validation=self.net.accuracy.name,
                f_name_test=self.net.accuracy.name,
                #
                JDE=self.job.JDE,
                DE=de_type,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max,
                smoothing=self.job.smoothing,
                smoothing_n_pass=self.job.smoothing_n_pass,
                reset_type='execute' if exists_reset_every else 'none',
                reset_counter=self.job.reset_every[
                    'counter'] if exists_reset_every else 1,
                reset_rand_pop=[tfop.name for tfop in self.net.rand_pop],
                reset_f=self.net.F_init.name,
                reset_cr=self.net.CR_init.name,
                reinsert_best=self.job.reinsert_best,
                inheritance=self.job.inheritance
            )
        else:
            ###########################################
            # DENN standard
            ###########################################
            self._module = tf.load_op_library(path.join(
                path.dirname(__file__), 'DENNOp.so')
            )
            self.denn_op = self._module.denn(
                # input params
                # [num_gen, eval_individual]
                self.net.cur_gen_options,
                self.net.label_placeholder,
                self.net.input_placeholder,
                # population infos
                self.net.F_placeholder,
                self.net.CR_placeholder,
                self.net.evaluated,  # FIRST EVAL
                self.net.populations,  # POPULATIONS
                # attributes
                graph=get_graph_proto(
                    self.net.graph.as_graph_def()),
                f_name_execute_net=self.net.objective_function.name,
                f_inputs=[elm.name for elm in self.net.targets],
                f_input_labels=self.net.label_placeholder.name,
                f_input_features=self.net.input_placeholder.name,
                f_min=self.job.clamp.min,
                f_max=self.job.clamp.max,
                JDE=self.job.JDE,
                DE=self.de_type,
                smoothing=self.job.smoothing,
                smoothing_n_pass=self.job.smoothing_n_pass,
                inheritance=self.job.inheritance
            )

    def run(self):
        """Runs the DE evolution."""
        raise NotImplementedError

    def __eval_individual(self, sess, individual):
        """Evaluates the NN with an individual.

        Params:
            sess (tensorflow.Session): the session of current job
            individual (numpy.ndarray): the current individual

        Returns:
            numpy.ndarray: the result of the NN for each record of
                           the test set 
        """
        return sess.run(self.net.y, feed_dict=dict(
            [
                (target, individual[num])
                for num, target in enumerate(self.net.targets)
            ]
            +
            [
                (self.net.label_placeholder, self.dataset.test_labels),
                (self.net.input_placeholder, self.dataset.test_data)
            ]
        ))

    def __eval_y(self, sess, best_idx, cur_pop):
        """Evaluates the NN with the best individual.

        Params:
            sess (tensorflow.Session): the session of current job
            best_idx (int): the id of the best individual in the
                            current population
            cur_pop (numpy.ndarray): the current population

        Returns:
            numpy.ndarray: the result of the NN for each record of
                           the test set 
        """
        best = [
            cur_pop[num][best_idx]
            for num, target in enumerate(self.net.targets)
        ]

        return sess.run(self.net.y, feed_dict=dict(
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

    def __eval_pop(self, sess, cur_pop):
        """Calculates accuracy on validation set for each individual.

        Params:
            sess (tensorflow.Session): the session of current job
            cur_pop (numpy.ndarray): the current population

        Returns:
            numpy.ndarray: the accuracy for each individual
        """
        evaluations = []
        for idx in range(self.job.NP):
            cur_evaluation = sess.run(self.net.accuracy, feed_dict=dict(
                [
                    (target, cur_pop[num][idx])
                    for num, target in enumerate(self.net.targets)
                ]
                +
                [
                    (self.net.label_placeholder, self.dataset.validation_labels),
                    (self.net.input_placeholder, self.dataset.validation_data),
                ]
            ))
            evaluations.append(cur_evaluation)

        return evaluations

    def __test(self, sess, cur_pop, evaluations):
        """Calculates the accuracy on test set.

        Params:
            sess (tensorflow.Session): the session of current job
            cur_pop (numpy.ndarray): the current population
            evaluations (numpy.ndarray): the evaluation of current
                                         population

        Returns:
            numpy.ndarray: the result of the NN for each record of
                           the test set 
        """
        best_idx = np.argmax(evaluations)

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

        return best_idx, cur_accuracy

    def __reinsert_best(self, cur_f, cur_cr, cur_pop, evaluations, test_results):
        """Reinserts the best evaluated individual in the population."""
        idx_worst = np.argmin(evaluations)
        cur_f[idx_worst] = test_results[self.de_type].best_of['F']
        cur_cr[idx_worst] = test_results[self.de_type].best_of['CR']
        for num in range(len(self.net.targets)):
            cur_pop[num][idx_worst] = test_results[
                self.de_type].best_of['individual'][num]
        return idx_worst

    def __reset_with_1_best(self, sess, cur_pop, best_f, best_cr, best, new_best_pos=0):
        """Reset the whole population and put one best individual."""
        ##
        # int new F & CR
        cur_f = sess.run(self.net.F_init)
        cur_cr = sess.run(self.net.CR_init)
        cur_f[new_best_pos] = best_f
        cur_cr[new_best_pos] = best_cr

        ##
        # Random initialization of the NN
        cur_pop = sess.run(self.net.rand_pop)

        ##
        # Initial population insertion
        for elm_idx, elem in enumerate(best):
            cur_pop[elm_idx][new_best_pos] = elem

        return (cur_f, cur_cr, cur_pop)

    def adaboost_run(self, sess, prev_F, prev_CR, prev_NN, test_results, options={}):
        """Run for AdaBoost jobs."""

        batch_counter = 0
        v_res = [0.0 for _ in range(self.job.NP)]

        ##
        # Reset checking
        if self.job.reset_every != False:
            reset_counter = 0
            reset_last_accuracy = 0.0

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
                ada_C = self.job.get_adaboost_cache(
                    batch_id,
                    cur_batch
                )

                # print(ada_C)
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
                        (self.net.F_placeholder, prev_F[self.de_type]),
                        (self.net.CR_placeholder, prev_CR[self.de_type])
                    ]
                    +
                    [
                        (self.net.cur_gen_options, [self.job.GEN_STEP, False])
                    ]
                    +
                    [
                        (self.net.ada_C_placeholder, ada_C)
                    ]
                ))

                # print("++ Op time {}".format(time() - time_start_gen))

                ##
                # get output
                cur_f = results.final_f
                cur_cr = results.final_cr
                cur_pop = results.final_populations

                # print(results.final_c)
                # print(results.final_ec)

                ##
                # Update adaboost cache
                self.job.set_adaboost_cache(
                    batch_id,
                    results.final_c
                )

                # print(len(cur_pop))
                # print(cur_pop[0].shape)
                # print(cur_pop[1].shape)

                # time_valutation = time()

                # accuracy test (not cross entropy)
                evaluations = self.__eval_pop(sess, cur_pop)

                # print(evaluations)
                # print(
                #     "++ Valutation:\t\t{}".format(time() - time_valutation))

                time_test = time()

                best_idx, cur_accuracy = self.__test(
                    sess, cur_pop, evaluations)

                test_results[self.de_type].values.append(cur_accuracy)

                # print("++ Test {}".format(time() - time_test))

                best_changed = update_best_of(
                    self.de_type,
                    test_results,
                    cur_accuracy,
                    evaluations[best_idx],
                    cur_f[best_idx],
                    cur_cr[best_idx],
                    [
                        cur_pop[num][best_idx] for num, target in enumerate(self.net.targets)
                    ],
                    self.job
                )

                test_results[self.de_type].F_population = cur_f
                test_results[self.de_type].CR_population = cur_cr
                test_results[self.de_type].population = cur_pop

                # print(
                #     "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
                #         self.de_type, (gen + 1) * job.GEN_STEP,
                #         job.name,
                #         time() - time_start_gen
                #     )
                # )

                prev_F[self.de_type] = cur_f
                prev_CR[self.de_type] = cur_cr
                prev_NN[self.de_type] = cur_pop

                pbar.update(gen)

                #### END SAMPLE ###

            # Reset C if request
            if self.job.ada_boost.reset_C_on_change_bacth:
                self.job.reset_a_C_adaboost_cache(batch_id)

            # Reinsert
            if self.job.reinsert_best and not best_changed:
                self.__reinsert_best(cur_f, cur_cr, cur_pop,
                                     evaluations, test_results)

            ##
            # Check reset
            if self.job.reset_every != False:
                ##
                # Reset test
                if evaluations[best_idx] <= reset_last_accuracy:
                    reset_counter += 1
                else:
                    reset_counter = 0
                ##
                # Update accuracy
                reset_last_accuracy = evaluations[best_idx]
                ##
                # Check for new population
                if reset_counter >= self.job.reset_every['counter']:
                    reset_counter = 0
                    cur_f, cur_cr, cur_pop = self.__reset_with_1_best(
                        sess,
                        cur_pop,
                        test_results[self.de_type].best_of['F'],
                        test_results[self.de_type].best_of['CR'],
                        test_results[self.de_type].best_of['individual']
                    )
                    prev_F[self.de_type] = cur_f
                    prev_CR[self.de_type] = cur_cr
                    prev_NN[self.de_type] = cur_pop
                    ##
                    # TO DO
                    # - save population before reset

        self.job.times[self.de_type] = time() - start_evolution
        self.job.accuracy[self.de_type] = cur_accuracy

        result_y = self.__eval_individual(
            sess, test_results[self.de_type].best_of['individual'])

        self.job.confusionM[self.de_type] = calc_confusin_M(
            self.dataset.test_labels, result_y)

        for class_ in range(self.job.confusionM[self.de_type][0].shape[0]):
            elm_tf = calc_TF(
                self.job.confusionM[self.de_type], class_)
            if self.de_type not in self.job.stats:
                self.job.stats[self.de_type] = []
            self.job.stats[self.de_type].append(precision_recall_acc(elm_tf))

        pbar.close()

        return cur_pop

    def training_run(self, sess, prev_F, prev_CR, prev_NN, test_results, options={}):
        """Run for training jobs."""
        start_job = options.get('start_job')

        cur_pop = prev_NN[self.de_type]
        cur_f = prev_F[self.de_type]
        cur_cr = prev_CR[self.de_type]

        # Soket listener
        with OpListener(tot_steps=self.job.TOT_GEN) as listener:
            ##
            # Handle SIGINT and SIGTERM
            def my_handler(signal, frame):
                listener.interrupt()
            signal.signal(signal.SIGINT, my_handler)
            signal.signal(signal.SIGTERM, my_handler)

            # time
            time_start_gen = time()
            # session run
            op_result = sess.run(self.denn_op, feed_dict=dict(
                [
                    (pop_ref, cur_pop[num])
                    for num, pop_ref in enumerate(self.net.populations)
                ]
                +
                [
                    (self.net.F_placeholder, cur_f),
                    (self.net.CR_placeholder, cur_cr)
                ]
            ))

            # print(op_result.final_eval_of_best_of_best)
            # print(dir(op_result))
            run_time = time() - time_start_gen
            print("++ Op time {}".format(run_time))

            # test time
            time_test = time()
            # test result
            cur_accuracy = sess.run(self.net.accuracy, feed_dict=dict(
                [
                    (target, op_result.final_best[num])
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

            ##
            # Extract population values
            test_results[self.de_type].values = op_result.final_eval_of_best
            test_results[self.de_type].population = op_result.final_populations
            test_results[self.de_type].F_population = op_result.final_f
            test_results[self.de_type].CR_population = op_result.final_cr
            ##
            # Extract best values
            test_results[self.de_type].best_of['F'] = op_result.final_best_f
            test_results[self.de_type].best_of['CR'] = op_result.final_best_cr
            test_results[self.de_type].best_of[
                'accuracy'] = op_result.final_eval_of_best_of_best
            test_results[self.de_type].best_of[
                'individual'] = op_result.final_best
            self.job.best['F'] = op_result.final_best_f
            self.job.best['CR'] = op_result.final_best_cr
            self.job.best['accuracy'] = op_result.final_eval_of_best_of_best
            self.job.best['individual'] = op_result.final_best

            result_y = self.__eval_individual(sess, op_result.final_best)

            self.job.confusionM[self.de_type] = calc_confusin_M(
                self.dataset.test_labels, result_y)

            for class_ in range(self.job.confusionM[self.de_type][0].shape[0]):
                elm_tf = calc_TF(self.job.confusionM[self.de_type], class_)
                if self.de_type not in self.job.stats:
                    self.job.stats[self.de_type] = []
                self.job.stats[self.de_type].append(
                    precision_recall_acc(elm_tf))

            self.job.times[self.de_type] = time() - time_start_gen
            self.job.accuracy[self.de_type] = cur_accuracy

            return op_result.final_populations

    def standard_run(self, sess, prev_F, prev_CR, prev_NN, test_results, options={}):
        """Run for standard jobs."""

        first_time = True
        batch_counter = 0
        v_res = [0.0 for _ in range(self.job.NP)]

        ##
        # Reset checking
        if self.job.reset_every != False:
            reset_counter = 0
            reset_last_accuracy = 0.0

        start_evolution = time()

        print("+ Start evolution")

        pbar = tqdm(total=self.job.TOT_GEN)

        for cur_gen in range(int(self.job.TOT_GEN / self.job.GEN_STEP)):

            gen = int(self.job.GEN_STEP / self.job.GEN_SAMPLES)

            cur_batch = self.dataset[batch_counter]
            batch_counter += 1

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
                        (self.net.F_placeholder,  prev_F[self.de_type]),
                        (self.net.CR_placeholder,  prev_CR[self.de_type])
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
                cur_f = results.final_f
                cur_cr = results.final_cr
                cur_pop = results.final_populations
                v_res = results.final_eval

                # print(len(cur_pop))
                # print(cur_pop[0].shape)
                # print(cur_pop[1].shape)

                time_valutation = time()

                evaluations = self.__eval_pop(sess, cur_pop)

                # print(evaluations)
                # print(
                #     "++ Valutation:\t\t{}".format(time() - time_valutation))

                time_test = time()

                best_idx, cur_accuracy = self.__test(
                    sess, cur_pop, evaluations)

                # print("++ Test {}".format(time() - time_test))

                test_results[self.de_type].values.append(cur_accuracy)

                best_changed = update_best_of(
                    self.de_type,
                    test_results,
                    cur_accuracy,
                    evaluations[best_idx],
                    cur_f[best_idx],
                    cur_cr[best_idx],
                    [
                        cur_pop[num][best_idx] for num, target in enumerate(self.net.targets)
                    ],
                    self.job
                )

                test_results[self.de_type].F_population = cur_f
                test_results[self.de_type].CR_population = cur_cr
                test_results[self.de_type].population = cur_pop
                first_time = False

                # print(
                #     "+ DENN[{}] up to {} gen on {} completed in {:.05} sec.".format(
                #         self.de_type, (gen + 1) * job.GEN_STEP,
                #         job.name,
                #         time() - time_start_gen
                #     )
                # )

                prev_F[self.de_type] = cur_f
                prev_CR[self.de_type] = cur_cr
                prev_NN[self.de_type] = cur_pop

                pbar.update(gen)

                ### END SAMPLE ###

            # reinsert best
            if self.job.reinsert_best and not best_changed:
                self.__reinsert_best(cur_f, cur_cr, cur_pop,
                                     evaluations, test_results)

            ##
            # Check reset
            if self.job.reset_every != False:
                ##
                # Reset test
                if evaluations[best_idx] <= reset_last_accuracy:
                    reset_counter += 1
                else:
                    reset_counter = 0
                ##
                # Update accuracy
                reset_last_accuracy = evaluations[best_idx]
                ##
                # Check for new population
                if reset_counter >= self.job.reset_every['counter']:
                    reset_counter = 0
                    cur_f, cur_cr, cur_pop = self.__reset_with_1_best(
                        sess,
                        cur_pop,
                        test_results[self.de_type].best_of['F'],
                        test_results[self.de_type].best_of['CR'],
                        test_results[self.de_type].best_of['individual']
                    )
                    prev_F[self.de_type] = cur_f
                    prev_CR[self.de_type] = cur_cr
                    prev_NN[self.de_type] = cur_pop
                    # force recomputation of eval of population
                    first_time = True
                    ##
                    # TO DO
                    # - salvarsi la popolazione prima del reset

        self.job.times[self.de_type] = time() - start_evolution
        self.job.accuracy[self.de_type] = cur_accuracy

        result_y = self.__eval_individual(
            sess, test_results[self.de_type].best_of['individual'])

        self.job.confusionM[self.de_type] = calc_confusin_M(
            self.dataset.test_labels, result_y)

        for class_ in range(self.job.confusionM[self.de_type][0].shape[0]):
            elm_tf = calc_TF(
                self.job.confusionM[self.de_type], class_)
            if self.de_type not in self.job.stats:
                self.job.stats[self.de_type] = []
            self.job.stats[self.de_type].append(precision_recall_acc(elm_tf))

        pbar.close()

        return cur_pop


class create(object):
    """Create a DENN object."""

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        self.__module = None

    def __call__(self, *args, **kwargs):
        module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp.so')
        )
        return module.denn(*args, **kwargs)

    @property
    def standard(self):
        self.__module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp.so')
        )
        return self.__module.denn(*self.__args, **self.__kwargs)

    @property
    def train(self):
        self.__module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp_training.so')
        )
        return self.__module.denn(*self.__args, **self.__kwargs)

    @property
    def ada(self):
        self.__module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp_ada.so')
        )
        return self.__module.denn(*self.__args, **self.__kwargs)
