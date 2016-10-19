import tensorflow as tf
import numpy as np
from time import time
from evo import DifferentialEvolution

NP = np.int32(100)
D = np.int32(1)
W = tf.fill((D, ), np.float64(0.1))
CR = np.float64(0.2)
NUM_GEN = 100

##
# Variables
x = tf.Variable(np.zeros((D, ), dtype=np.float64), name="x")

##
# Placeholder
target = tf.placeholder(tf.float64, (D,), name="target")

##
# f(x)
# f_x = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, target))
# loss = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, x))

##
# Easom's function
# f7(x)=sum(-x(i)·sin(sqrt(abs(x(i)))))
# i=1:n; -500<=x(i)<=500.
# f_x = tf.reduce_sum(tf.mul(tf.neg(target), tf.sin(tf.sqrt(tf.abs(target)))))
# loss = -tf.reduce_sum(tf.mul(tf.neg(x), tf.sin(tf.sqrt(tf.abs(x)))))

##
# Easom's function
# Moved axis parallel hyper-ellipsoid
# f1c(x)=sum(5*i·x(i)^2)
# i=1:n, -5.12<=x(i)<=5.12.
# f_x = tf.reduce_sum(tf.mul(tf.mul(np.float64(5), D), tf.square(target)))
# loss = -tf.reduce_sum(tf.mul(tf.mul(np.float64(5), D), tf.square(x)))

##
# Schwefel's function 7 
# f7(x)=sum(-x(i)·sin(sqrt(abs(x(i)))))
# i=1:n; -500<=x(i)<=500
# f(x)=-n·418.9829; x(i)=420.9687, i=1:n. 
f_x = tf.reduce_sum(tf.mul(tf.neg(target), tf.sin(tf.sqrt(tf.abs(target)))))
loss = -tf.reduce_sum(tf.mul(tf.neg(x), tf.sin(tf.sqrt(tf.abs(x)))))

##
# Rastrigin’s Function
# f6(x)=10·n+sum(x(i)^2-10·cos(2·pi·x(i)))
# i=1:n; -5.12<=x(i)<=5.12.
# f_x = tf.add(tf.mul(np.float64(10), NP), tf.reduce_sum(tf.sub(tf.square(target), tf.mul(np.float64(10), tf.cos(tf.mul(tf.mul(np.float64(2), np.pi), target))))))
# loss = -tf.add(tf.mul(np.float64(10), NP), tf.reduce_sum(tf.sub(tf.square(x),tf.mul(np.float64(10), tf.cos(tf.mul(tf.mul(np.float64(2), np.pi), x))))))

MIN_VAL = -500
MAX_VAL = 500

##
# Actions
random_population = tf.random_uniform(
    (NP, D), minval=MIN_VAL, maxval=MAX_VAL, dtype=tf.float64)
random_population_vector = tf.random_uniform(
    (D, ), minval=MIN_VAL, maxval=MAX_VAL, dtype=tf.float64)

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
    
    result = sess.run(x)
    print("+ Done in {}".format(time() - tot_time))
    print("+ Solution: {} -> f(x) = {}".format(result, sess.run(loss)))

with tf.Session() as sess:
    sess.run(init)

    # print(sess.run(f_x, feed_dict={target: [420.9687, 420.9687, 420.9687]}))

    ##
    # init
    tmp_population = sess.run(random_population)
    tmp_vector = sess.run(random_population_vector)

    print("+----- Differential Evolution -----")
    DE.evolve(tmp_population, NUM_GEN, f_x)

    print("+----- Gradient Descent -----")
    sess.run(x.assign(tmp_vector))

    gradient_descend(sess)
