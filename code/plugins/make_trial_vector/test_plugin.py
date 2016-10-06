
from time import sleep
sleep(9)

import tensorflow as tf

zero_out_module = tf.load_op_library('./make_trial_vector.so')

with tf.Session(''):                                 
    print(zero_out_module.zero_out([[1, 2, 3], [4, 5, 1]]).eval())
