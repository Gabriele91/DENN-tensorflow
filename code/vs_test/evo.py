import tensorflow as tf
import numpy as np
from time import time


class DifferentialEvolution(object):

    def __init__(self, NP, D, W, CR, external_target):
        ##
        # New OP
        self.DE = tf.load_op_library('./de_mod.so')

        ##
        # Variables
        self.cur_population = tf.Variable(
            np.zeros((NP, D), dtype=np.float64), name="population")

        ##
        # Placeholder
        self.gen_placeholder = tf.placeholder(
            tf.float64, (NP, D), name="generation")
        self.target = tf.placeholder(tf.float64, (D,), name="target")
        self.external_target = external_target
        self.target_index = tf.placeholder(tf.int32, (), name="target_index")

        ##
        # Actions
        self.gen_trial_vectors = self.DE.trial_vectors(
            self.cur_population, W, CR)
        self.assign_individual = self.DE.assign_individual(
            self.cur_population, self.target, self.target_index)
        self.assign_new_population = self.cur_population.assign(
            self.gen_placeholder)

        self.init = tf.initialize_all_variables()
        self.res_f_x = []

    def evolve(self, population, num_generations, f_x, new_assignment=True):

        tot_time = time()

        with tf.Session() as sess:
            sess.run(self.init)

            tmp_population = sess.run(self.assign_new_population, feed_dict={
                self.gen_placeholder: population
            })

            for generation in range(num_generations):

                start = time()

                trial_vectors = sess.run(
                    self.gen_trial_vectors)

                for index, individual in enumerate(trial_vectors):
                    f_x_old = sess.run(
                        f_x, feed_dict={
                            self.external_target: tmp_population[index]
                        })
                    f_x_new = sess.run(
                        f_x, feed_dict={
                            self.external_target: individual
                        })
                    # print(f_x_new, f_x_old)
                    if f_x_new < f_x_old:
                        if not new_assignment:
                            tmp_population[index] = individual
                        else:
                            sess.run(self.assign_individual, feed_dict={
                                self.target: individual,
                                self.target_index: index
                            })

                if not new_assignment:
                    tmp_population = sess.run(self.assign_new_population, feed_dict={
                        self.gen_placeholder: tmp_population
                    })
                else:
                    tmp_population = sess.run(self.cur_population)

                print("+ Calculated gen. {} in {}".format(
                    generation, time() - start), end="\r")
                
                results = sess.run(self.cur_population)
                # print(results)
                cur_min = 10**10

                for index, individual in enumerate(results):
                    f_x_res = sess.run(f_x, feed_dict={
                        self.external_target: individual
                    })
                    if f_x_res < cur_min:
                        cur_min = f_x_res
                
                self.res_f_x.append(cur_min)

            print("+ Done in {}".format(time() - tot_time))

            results = sess.run(self.cur_population)
            # print(results)
            index_min = -1
            cur_min = 10**10

            for index, individual in enumerate(results):
                f_x_res = sess.run(f_x, feed_dict={
                    self.external_target: individual
                })
                if f_x_res < cur_min:
                    cur_min = f_x_res
                    index_min = index

            best = results[index_min]

            # print("+ Results:\n{}".format(results))
            # print("+ Error: {}".format(f_x_res))
            print("+ Best vector: {} -> f(x) = {}".format(best, cur_min))

            return self.res_f_x
