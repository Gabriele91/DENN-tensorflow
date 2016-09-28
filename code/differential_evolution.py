import tensorflow as tf
import numpy as np

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

f_x = tf.reduce_sum(individual)

random_tensor = tf.random_uniform((D, ), dtype=tf.float64)
select_condition = tf.map_fn(
    lambda elm: tf.greater(elm, CR), random_tensor, dtype=tf.bool)
random_population = tf.random_uniform((NP, D), dtype=tf.float64)
random_int = tf.gather(tf.random_uniform((1, ), dtype=tf.int32, maxval=NP), 0)

gen_noisy_random_vector = tf.add(tf.mul(tf.sub(x_a, x_b), W), x_c)
gen_trial_vector = tf.select(select_condition, ph_target, ph_noisy_vector)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    cur_population = sess.run(cur_population.assign(random_population))
    print("+ Population:\n{}".format(cur_population))

    tmp_next_gen = sess.run(next_gen.assign(tf.fill((NP, D), np.float64(0.0))))

    for i_target in range(NP):

        tmp_next_gen = sess.run(next_gen.assign(tmp_next_gen))  

        target = sess.run(target.assign(tf.gather(cur_population, i_target)))
        print("+ Target: {}".format(target))

        i_a, i_b, i_c = 0, 0, 0

        while i_a == i_target:
            i_a = sess.run(random_int)
        while i_b == i_target or i_b == i_a:
            i_b = sess.run(random_int)
        while i_c == i_target or i_c == i_a or i_c == i_b:
            i_c = sess.run(random_int)

        print("+ Indexes: {}, {}, {}".format(i_a, i_b, i_c))

        noisy_random_vector = sess.run(noisy_random_vector.assign(sess.run(
            gen_noisy_random_vector, feed_dict={
                x_a: sess.run(tf.gather(cur_population, i_a)),
                x_b: sess.run(tf.gather(cur_population, i_b)),
                x_c: sess.run(tf.gather(cur_population, i_c))
            })))

        print("+ Noisy random vector: {}".format(noisy_random_vector))

        trial_vector = sess.run(trial_vector.assign(
            sess.run(gen_trial_vector, feed_dict={
                ph_target: target,
                ph_noisy_vector: noisy_random_vector
            })))

        print("+ Trial vector: {}".format(trial_vector))

        obj_fun_target = sess.run(f_x, feed_dict={
            individual: target
        })

        print("+ Objective function target: {}".format(obj_fun_target))

        obj_fun_trial_v = sess.run(f_x, feed_dict={
            individual: trial_vector
        })

        print("+ Objective function trial vector: {}".format(obj_fun_trial_v))

        target = sess.run(tf.select(
            tf.fill((D, ), tf.less(obj_fun_trial_v, obj_fun_target)),
            trial_vector,
            target)
        )

        print("+ Target result: {}".format(target))

        tmp_next_gen[i_target] = target

        print("+ Next gen:\n{}".format(tmp_next_gen))

    # print(sess.run(next_gen))

    # print(sess.run(W))
    # print(sess.run(tf.gather(population, 0)))
    # print(sess.run(tf.map_fn(lambda arr: tf.reduce_sum(arr), population)))
