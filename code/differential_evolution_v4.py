import tensorflow as tf
import numpy as np
from time import time

# tensor_1d = np.array([1.3, 1, 4.0, 23.99])
# tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

NP = np.int32(50)
D = np.int32(3)
CR = np.float64(0.5)

##
# variables
cur_population = tf.Variable(
    np.zeros((NP, D), dtype=np.float64), name="population")
next_gen = tf.Variable(
    np.zeros((NP, D), dtype=np.float64), name="next_generation")
target = tf.Variable(np.zeros((D, ), dtype=np.float64), name="target")
random_index_mask = tf.Variable(
    np.zeros((NP, ), dtype=np.bool), name="random_index_mask")
noisy_random_vector = tf.Variable(
    np.zeros((D, ), dtype=np.float64), name="noisy_random_vector")

##
# placeolder
new_gen = tf.placeholder(tf.float64, (NP, D), name="new_gen")
indexes = tf.placeholder(tf.int32, (3), name="indexes")
individual = tf.placeholder(tf.float64, (D,), name="individual")
x_a = tf.placeholder(tf.float64, (D,), name="X_A")
x_b = tf.placeholder(tf.float64, (D,), name="X_B")
x_c = tf.placeholder(tf.float64, (D,), name="X_C")
ph_target = tf.placeholder(tf.float64, (D,), name="target_placeholder")
ph_noisy_vector = tf.placeholder(
    tf.float64, (D,), name="trial_vector_placeholder")


obj_fun_target = tf.placeholder(dtype=tf.float64, name="fun_target")
obj_fun_trial_v = tf.placeholder(dtype=tf.float64, name="fun_trial")
target_index = tf.placeholder(dtype=tf.int32, name="target_index")

trial_vector = tf.Variable(
    np.zeros((D, ), dtype=np.float64), name="trial_vector")

W = tf.fill((D, ), np.float64(0.80))

# f_x = tf.reduce_sum(tf.map_fn(lambda elm: elm**2, individual))
f_x_target = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, target))
f_x_trial = tf.reduce_sum(tf.map_fn(lambda elm: (elm - 1)**2, trial_vector))

random_tensor = tf.random_uniform((D, ), dtype=tf.float64)
select_condition = tf.map_fn(
    lambda elm: tf.greater(elm, CR), random_tensor, dtype=tf.bool)
random_population = tf.random_uniform((NP, D), dtype=tf.float64)
random_int = tf.gather(tf.random_uniform((1, ), dtype=tf.int32, maxval=NP), 0)

##
# Generators
gen_assign_trial_vector = trial_vector.assign(
    tf.select(select_condition, target, noisy_random_vector))
gen_3_random_indexes = tf.gather(
    tf.random_shuffle(
        tf.boolean_mask(
            tf.range(0, NP),
            random_index_mask
        )
    ),
    [0, 1, 2]
)

##
# Actions
assign_next_gen = next_gen.assign(new_gen)
change_generation = cur_population.assign(next_gen)
assign_random_population = cur_population.assign(random_population)
assign_noisy_vector = noisy_random_vector.assign(
    tf.add(tf.mul(tf.sub(x_a, x_b), W), x_c))
clean_next_gen = next_gen.assign(tf.fill((NP, D), np.float64(0.0)))
assign_target = target.assign(tf.gather(cur_population, target_index))
assign_new_target = target.assign(tf.select(
    tf.fill((D, ), tf.less(obj_fun_trial_v, obj_fun_target)),
    trial_vector,
    target)
)
assign_mask = random_index_mask.assign(
    tf.logical_not(tf.cast(tf.one_hot(target_index, NP), tf.bool)))
get_indexes = tf.gather(cur_population, indexes)

init = tf.initialize_all_variables()

tot_time = time()

with tf.Session() as sess:
    sess.run(init)

    ##
    # init DE
    sess.run([
        assign_random_population,
        clean_next_gen
    ])

    # print("+ Population:\n{}".format(sess.run(cur_population)))
    # print("+ NextGen:\n{}".format(sess.run(next_gen)))

    for generation in range(100):

        start = time()

        tmp_new_gen = []

        for i_target in range(NP):

            ##
            # Random idexes and target
            sess.run(assign_mask, feed_dict={
                target_index: i_target
            })
            _, cur_indexes = sess.run([
                assign_target,
                gen_3_random_indexes
            ], feed_dict={
                target_index: i_target
            })

            # print("+ Target: {}".format(sess.run(target)))
            # print("+ Indexes: {}, {}, {}".format(*indexes))

            tmp_vectors = sess.run(get_indexes, feed_dict={
                indexes: cur_indexes
                })

            sess.run(assign_noisy_vector, feed_dict={
                x_a: tmp_vectors[0],
                x_b: tmp_vectors[1],
                x_c: tmp_vectors[2]
            })

            # print(
            #     "+ Noisy random vector: {}".format(
            #         sess.run(
            #             noisy_random_vector)))

            sess.run(gen_assign_trial_vector)

            # print("+ Trial vector: {}".format(sess.run(trial_vector)))

            tmp_fun_target, tmp_fun_trial = sess.run([f_x_target, f_x_trial])

            # print("+ Objective function target: {}".format(obj_fun_target))
            # print(
            #     "+ Objective function trial vector: {}".format(obj_fun_trial_v))

            sess.run(assign_new_target, feed_dict={
                obj_fun_target: tmp_fun_target,
                obj_fun_trial_v: tmp_fun_trial
            })

            # print("+ Target result: {}".format(sess.run(target)))

            tmp_new_gen.append(sess.run(target))

            # print("+ NextGen:\n{}".format(tmp_new_gen))

        ##
        # NEW GENERATION
        sess.run(assign_next_gen, feed_dict={
            new_gen: tmp_new_gen
        })

        # print(
        #     "+ Gen[{}][{}]:\n{}".format(
        #         generation, time() - start, sess.run(next_gen)))

        sess.run(change_generation)
        sess.run(clean_next_gen)

        # print("+ Calculated gen: {}".format(generation), end="\r")
        print("+ Calculated gen. {} in {}".format(
            generation, time() - start), end="\r")

    print("+ Done in {}".format(time() - tot_time))

    results = sess.run(cur_population)
    f_x_res = []

    for indiv in results:
        f_x_res.append(sum([abs(1.0 - elm) for elm in indiv]))

    print("+ Results:\n{}".format(results))
    print("+ Error: {}".format(f_x_res))
    print("+ Best vector: {}".format(results[f_x_res.index(min(f_x_res))]))
