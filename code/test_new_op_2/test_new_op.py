import tensorflow as tf
import numpy as np
from time import time

NP = np.int32(50)
D = np.int32(3)
W = tf.fill((D, ), np.float64(0.80))
CR = np.float64(0.5)

DE = tf.load_op_library('./de_mod.so')

##
# Variables
cur_population = tf.Variable(
    np.zeros((NP, D), dtype=np.float64), name="population")

##
# Placeholder
gen_placeholder = tf.placeholder(tf.float64, (NP, D), name="generation")
target = tf.placeholder(tf.float64, (D,), name="target")
target_index = tf.placeholder(tf.int32, 1, name="target_index")

##
# f(x)
f_x = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, target))

##
# Actions
gen_trial_vectors = DE.trial_vectors(cur_population, W, CR)
random_population = tf.random_uniform((NP, D), dtype=tf.float64)
assign_new_population = cur_population.assign(gen_placeholder)


init = tf.initialize_all_variables()

tot_time = time()

with tf.Session() as sess:
    sess.run(init)

    ##
    # init DE
    tmp_random_population = sess.run(random_population)
    tmp_population = sess.run(assign_new_population, feed_dict={
        gen_placeholder: tmp_random_population
    })

    for generation in range(100):

        start = time()

        trial_vectors = sess.run(gen_trial_vectors)

        # fx_trial_v = sess.run(apply_fx, feed_dict={
        #     gen_placeholder: trial_vectors
        # })

        # print(tmp_population, type(tmp_population))

        for index, individual in enumerate(trial_vectors):
            f_x_old = sess.run(f_x, feed_dict={target: tmp_population[index]})
            f_x_new = sess.run(f_x, feed_dict={target: individual})
            if f_x_new < f_x_old:
                tmp_population[index] = individual

        tmp_population = sess.run(assign_new_population, feed_dict={
            gen_placeholder: tmp_population
        })

        print("+ Calculated gen. {} in {}".format(
            generation, time() - start), end="\r")

    print("+ Done in {}".format(time() - tot_time))

    results = sess.run(cur_population)
    f_x_res = []

    for indiv in results:
        f_x_res.append(sum([abs(1.0 - elm) for elm in indiv]))

    # print("+ Results:\n{}".format(results))
    # print("+ Error: {}".format(f_x_res))
    print("+ Best vector: {}".format(results[f_x_res.index(min(f_x_res))]))
