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

##
# f(x)
def f_x(elms):
    return tf.reduce_sum(elms).eval()

##
# Actions
random_population = tf.random_uniform((NP, D), dtype=tf.float64)
assign_random_population = cur_population.assign(random_population)
apply_fx = tf.map_fn(f_x, gen_placeholder, dtype=tf.float64)



init = tf.initialize_all_variables()

tot_time = time()

with tf.Session() as sess:
    sess.run(init)

    ##
    # init DE
    sess.run([
        assign_random_population
    ])

    for generation in range(1):
        trial_vector = DE.trial_vectors(cur_population, W, CR).eval()

        fx_trial_v = sess.run(apply_fx, feed_dict={
            gen_placeholder: trial_vector
        })

        print(trial_vector)
        print(fx_trial_v)