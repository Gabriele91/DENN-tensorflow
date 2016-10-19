import tensorflow as tf
import numpy as np
from time import time
from evo import DifferentialEvolution

NP = np.int32(50)
D = np.int32(3)
W = tf.fill((D, ), np.float64(0.80))
CR = np.float64(0.5)
NUM_GEN = 100

##
# Variables
x = tf.Variable(np.zeros((D, ), dtype=np.float64), name="x")

##
# Placeholder
target = tf.placeholder(tf.float64, (D,), name="target")

##
# f(x)
f_x = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, target))
loss = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, x))

##
# Actions
random_population = tf.random_uniform((NP, D), dtype=tf.float64)
random_population_vector = tf.random_uniform(
    (D, ), minval=-10, maxval=10, dtype=tf.float64)

DE = DifferentialEvolution(NP, D, W, CR, target)

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
GD = opt.minimize(loss)

init = tf.initialize_all_variables()


def gradient_descend(sess):
    tot_time = time()

    for generation in range(NUM_GEN):
        start = time()

        sess.run(GD)
        # print(sess.run(x))

        print("+ Calculated gen. {} in {}".format(
            generation, time() - start), end="\r")

    print("+ Done in {}".format(time() - tot_time))
    print("+ Solution: {}".format(sess.run(x)))

with tf.Session() as sess:
    sess.run(init)

    ##
    # init
    tmp_population = sess.run(random_population)
    tmp_vector = sess.run(random_population_vector)

    print("+----- Differential Evolution -----")
    DE.evolve(tmp_population, NUM_GEN, f_x)

    print("+----- Gradient Descent -----")
    sess.run(x.assign(tmp_vector))

    gradient_descend(sess)
