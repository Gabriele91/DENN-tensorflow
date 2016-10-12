import tensorflow as tf

DE = tf.load_op_library('./de_mod.so')


with tf.Session(''):                                 
    print(DE.new_gen([[1, 2, 3], [4, 5, 1]]).eval())
