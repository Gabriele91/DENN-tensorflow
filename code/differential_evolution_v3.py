import tensorflow as tf
import numpy as np
from time import time

# tensor_1d = np.array([1.3, 1, 4.0, 23.99])
# tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

NP = 6
D = 3
CR = tf.constant(0.50, dtype=tf.float64)

cur_population = tf.Variable(
    np.zeros((NP, D), dtype=np.float64), name="population")
next_gen = tf.Variable(
    np.zeros((NP, D), dtype=np.float64), name="next_generation")

individual = tf.placeholder(tf.float64, (D,), name="individual")
x_a = tf.placeholder(tf.float64, (D,), name="X_A")
x_b = tf.placeholder(tf.float64, (D,), name="X_B")
x_c = tf.placeholder(tf.float64, (D,), name="X_C")
ph_target = tf.placeholder(tf.float64, (D,), name="target_placeholder")
ph_noisy_vector = tf.placeholder(
    tf.float64, (D,), name="trial_vector_placeholder")

target = tf.Variable(np.zeros((D, ), dtype=np.float64), name="target")
noisy_random_vector = tf.Variable(
    np.zeros((D, ), dtype=np.float64), name="noisy_random_vector")
trial_vector = tf.Variable(
    np.zeros((D, ), dtype=np.float64), name="trial_vector")

W = tf.fill((D, ), np.float64(0.80))

# f_x = tf.reduce_sum(tf.map_fn(lambda elm: elm**2, individual))
f_x = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, individual))

random_tensor = tf.random_uniform((D, ), dtype=tf.float64)
select_condition = tf.map_fn(
    lambda elm: tf.greater(elm, CR), random_tensor, dtype=tf.bool)
random_population = tf.random_uniform((NP, D), dtype=tf.float64)
random_int = tf.gather(tf.random_uniform((1, ), dtype=tf.int32, maxval=NP), 0)

gen_noisy_random_vector = tf.add(tf.mul(tf.sub(x_a, x_b), W), x_c)
gen_trial_vector = tf.select(select_condition, ph_target, ph_noisy_vector)


init = tf.initialize_all_variables()

start = time()

with tf.Session() as sess:
    sess.run(init)

    ##
    # init DE
    sess.run(
        [
            cur_population.assign(random_population),
            next_gen.assign(tf.fill((NP, D), np.float64(0.0)))
        ])

    print("+ Population:\n{}".format(sess.run(cur_population)))
    print("+ NextGen:\n{}".format(sess.run(next_gen)))

    for generation in range(40):

        tmp_new_gen = []

        for i_target in range(NP):
            ##
            # Random idexes and target
            _, (i_a, i_b, i_c) = sess.run([
                target.assign(tf.gather(cur_population, i_target)),
                tf.gather(
                    tf.random_shuffle(
                        tf.boolean_mask(
                            tf.range(0, NP),
                            tf.logical_not(
                                tf.cast(tf.one_hot(i_target, NP), tf.bool))
                        )
                    ),
                    [0, 1, 2]
                )
            ])

            # print("+ Target: {}".format(sess.run(target)))
            # print("+ Indexes: {}, {}, {}".format(i_a, i_b, i_c))

            sess.run(noisy_random_vector.assign(sess.run(
                gen_noisy_random_vector, feed_dict={
                    x_a: sess.run(tf.gather(cur_population, i_a)),
                    x_b: sess.run(tf.gather(cur_population, i_b)),
                    x_c: sess.run(tf.gather(cur_population, i_c))
                })))

            # print(
            #     "+ Noisy random vector: {}".format(
            #         sess.run(
            #             noisy_random_vector)))

            sess.run(trial_vector.assign(
                sess.run(gen_trial_vector, feed_dict={
                    ph_target: sess.run(target),
                    ph_noisy_vector: sess.run(noisy_random_vector)
                })))

            # print("+ Trial vector: {}".format(sess.run(trial_vector)))

            obj_fun_target = sess.run(f_x, feed_dict={
                individual: sess.run(target)
            })

            # print("+ Objective function target: {}".format(obj_fun_target))

            obj_fun_trial_v = sess.run(f_x, feed_dict={
                individual: sess.run(trial_vector)
            })

            # print("+ Objective function trial vector:
            # {}".format(obj_fun_trial_v))

            sess.run(target.assign(tf.select(
                tf.fill((D, ), tf.less(obj_fun_trial_v, obj_fun_target)),
                sess.run(trial_vector),
                sess.run(target))
            ))

            # print("+ Target result: {}".format(sess.run(target)))

            tmp_new_gen.append(tf.convert_to_tensor(sess.run(target)))

            # print("+ NextGen:\n{}".format(tmp_new_gen))

        ##
        # NEW GENERATION
        sess.run(next_gen.assign(tf.convert_to_tensor(tmp_new_gen)))
        # print(
        #     "+ Gen[{}][{}]:\n{}".format(
        #         generation, time() - start, sess.run(next_gen)))
        print("+ Calculated gen: {}".format(generation), end="\r")
        sess.run(cur_population.assign(next_gen))
        sess.run(next_gen.assign(tf.fill((NP, D), np.float64(0.0))))

        start = time()

    results = sess.run(cur_population)
    f_x_res = []

    for indiv in results:
        f_x_res.append(sum([abs(1.0 - elm) for elm in indiv]))

    print("+ Results:\n{}".format(results))
    print("+ Error: {}".format(f_x_res))
    print("+ Best vector: {}".format(results[f_x_res.index(min(f_x_res))]))

