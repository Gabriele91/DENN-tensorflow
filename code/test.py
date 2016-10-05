import tensorflow as tf
import numpy as np
from time import time

NP = np.int32(6)
D = np.int32(3)
CR = np.float64(0.5)

##
# Variables
cur_population = tf.Variable(
    np.zeros((NP, D), dtype=np.float64), name="population")

##
# Generators
gen_random_population = tf.random_uniform((NP, D), dtype=tf.float64)

##
# Actions
assign_random_population = cur_population.assign(gen_random_population)


init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    # print(dir(sess), sess.partial_run_setup())

    start = time()
    for x in range(100000):
        sess.run(assign_random_population)
    print("+ Elapsed: {}".format(time() - start))

    start = time()

    for x in range(100000):
        partial = sess.partial_run_setup([assign_random_population], [])
        sess.partial_run(partial, assign_random_population)
    print("+ Elapsed: {}".format(time() - start))
