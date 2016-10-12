import tensorflow as tf

DE = tf.load_op_library('./de_mod.so')


with tf.Session(''):                                 
    print(DE.trial_vectors([
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 12],
        [13, 14, 16]
        ], 0.8, 0.5).eval())
