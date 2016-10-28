import tensorflow as tf
import numpy as np
from time import time
from evo import DifferentialEvolution
from matplotlib import pyplot as plt
from plotter import my_plot
from plotter import my_hist

NP = np.int32(100)
D = np.int32(20)
W = tf.fill((D, ), np.float64(0.8))
CR = np.float64(0.5)
NUM_GEN = 1000

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
# loss = tf.reduce_sum(tf.mul(tf.neg(x), tf.sin(tf.sqrt(tf.abs(x)))))

##
# Easom's function
# Moved axis parallel hyper-ellipsoid
# f1c(x)=sum(5*i·x(i)^2)
# i=1:n, -5.12<=x(i)<=5.12.
# f_x = tf.reduce_sum(tf.mul(tf.mul(np.float64(5), D), tf.square(target)))
# loss = tf.reduce_sum(tf.mul(tf.mul(np.float64(5), D), tf.square(x)))

##
# Schwefel's function 7
# f7(x)=sum(-x(i)·sin(sqrt(abs(x(i)))))
# i=1:n; -500<=x(i)<=500
# f(x)=-n·418.9829; x(i)=420.9687, i=1:n.
# f_x = tf.reduce_sum(tf.mul(tf.neg(target), tf.sin(tf.sqrt(tf.abs(target)))))
# loss = -tf.reduce_sum(tf.mul(tf.neg(x), tf.sin(tf.sqrt(tf.abs(x)))))

##
# Griewangk's function
# f8(x)=sum(x(i)^2/4000)-prod(cos(x(i)/sqrt(i)))+1
# i=1:n -600<=x(i)<= 600.
# f(x)=0; x(i)=0, i=1:n.
f_x = tf.add(
    tf.sub(
        tf.reduce_sum(
            tf.div(
                tf.square(target),
                np.float64(4000)
            )
        ),
        tf.reduce_prod(
            tf.div(
                tf.cos(target),
                tf.sqrt(
                    tf.to_double(
                        tf.range(np.int32(1), np.int32(D + 1))
                    )
                )
            )
        )
    ),
    np.float64(1), name="evaluate")
loss = tf.add(
    tf.sub(
        tf.reduce_sum(
            tf.div(
                tf.square(x),
                np.float64(4000)
            )
        ),
        tf.reduce_prod(
            tf.div(
                tf.cos(x),
                tf.sqrt(
                    tf.to_double(
                        tf.range(np.int32(1), np.int32(D + 1))
                    )
                )
            )
        )
    ),
    np.float64(1), name="loss")

##
# Rastrigin’s Function
# f6(x)=10·n+sum(x(i)^2-10·cos(2·pi·x(i)))
# i=1:n; -5.12<=x(i)<=5.12.
# f_x = tf.add(tf.mul(np.float64(10), NP), tf.reduce_sum(tf.sub(tf.square(target), tf.mul(np.float64(10), tf.cos(tf.mul(tf.mul(np.float64(2), np.pi), target))))))
# loss = -tf.add(tf.mul(np.float64(10), NP), tf.reduce_sum(tf.sub(tf.square(x),tf.mul(np.float64(10), tf.cos(tf.mul(tf.mul(np.float64(2), np.pi), x))))))

MIN_VAL = -600
MAX_VAL = 600

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

    # f_x results
    res_f_x = []

    # init
    sess.run(x.assign(random_population_vector))

    for generation in range(NUM_GEN):
        start = time()

        sess.run(GD)
        # print(sess.run(x))

        print("+ Calculated gen. {} in {}".format(
            generation, time() - start), end="\r")
        
        # get variable
        tmp = sess.run(x)
        # calculate and add f_x
        res_f_x.append(sess.run(f_x, feed_dict={
            target: tmp
        }))

    result = sess.run(x)
    print("+ Done in {}".format(time() - tot_time))
    print("+ Solution: {} -> f(x) = {}".format(result, sess.run(loss)))

    return res_f_x

with tf.Session() as sess:
    sess.run(init)

    # print(sess.run(f_x, feed_dict={target: [420.9687, 420.9687, 420.9687]}))

    ##
    # init
    tmp_population = sess.run(random_population)
    tmp_vector = sess.run(random_population_vector)

    print("+----- Differential Evolution -----")
    res_f_x_DE = DE.evolve(tmp_population, NUM_GEN, f_x)

    print("+----- Gradient Descent -----")
    res_f_x_GD =gradient_descend(sess)

x_range = range(NUM_GEN)

FIGURES = [
    # ----- PLOTS -----
    {
        'data': [
            {
                'values': [x_range, res_f_x_DE],
                'color': "#dd0000",
                'label': "DE"
            },
            {
                'values': [x_range, res_f_x_GD],
                'color': "#0000dd",
                'label': "GD"
            }
        ],
        'title': "Griewangk's function",
        'type': "plot",
        #'axis': (0, GEN, -1, 20),
        'filename': "figures/bechmarck_griewangk",
        'plot': {
            'x_label': "iteration (gen)",
            'y_label': "f(x)",
        }
    }
]

for figure in FIGURES:
    fig = plt.figure()

    fig.suptitle(figure['title'], fontsize=14, fontweight='bold')

    if figure['type'] == 'plot':
        print("- Generating \"{}\" [{}] -> {}".format(
            figure['title'],
            figure['type'],
            figure['filename']
        ))
        my_plot(fig, figure['data'])
        if 'axis' in figure:
            plt.axis(figure['axis'])
        plt.xlabel(figure['plot']['x_label'])
        plt.ylabel(figure['plot']['y_label'])
        plt.grid(True)
        plt.savefig(figure['filename'], dpi=400)
    elif figure['type'] == 'hist':
        print("- Generating \"{}\" [{}] -> {}".format(
            figure['title'],
            figure['type'],
            figure['filename']
        ))
        my_hist(fig,
                figure['data'],
                figure['bins'],
                figure['range'],
                figure['colors'],
                figure['labels'],
                figure.get('normalized', False),
                max_y=0.5
                )
        plt.xlabel(figure['plot']['x_label'])
        plt.ylabel(figure['plot']['y_label'])
        plt.savefig(figure['filename'], dpi=400)

##
# Show all plots
plt.show()
