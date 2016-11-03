import tensorflow as tf
from tensorflow.python.framework import ops

import sys
import atexit
import numpy as np
from matplotlib import pyplot as plt
from plotter import my_plot
from plotter import my_hist
from time import time
from time import sleep
from os import path
import struct
from operator import itemgetter

#sleep(6)

def load_data(path):
    images_data = []
    label_data = []
    print('+ Load data ...')
    with open(path+'.img', 'rb') as images:
        num_imgs, x, y = struct.unpack('III', images.read(12))
        print('+ {} images {}x{}'.format(num_imgs, x, y))
        for img in range(num_imgs):
            images_data.append(
                [elm / 255. for elm in struct.unpack(
                    'B' * (x * y), images.read(x * y))])

    with open(path+'.lb', 'rb') as images:
        num_lbl = struct.unpack('I', images.read(4))[0]
        print('+ {} labels'.format(num_lbl))
        for img in range(num_lbl):
            cur_label = struct.unpack('B', images.read(1))[0]
            label_data.append(
                [0 if index != cur_label else 1 for index in range(2)])

    print('+ images: \n', images_data)
    print('+ labels: \n', label_data)

    print('+ loading done!')
    return images_data, label_data

def redefine_stdout():
    if len(sys.argv) > 1:
        #overwrite stdout
        sys.stdout = open(sys.argv[1],"a")
        #close at exit
        def close_stdout():
            sys.stdout.close()
        #reg function at exit
        atexit.register(close_stdout)

def batch(data, label, size):
    out_data = []
    out_label = []
    for index, elm in enumerate(data):
        if len(out_data) < size:
            out_data.append(elm)
            out_label.append(label[index])
        else:
            yield out_data, out_label
            out_data = []
            out_label = []

def get_graph_proto(graph_or_graph_def, as_text=True):
    """Return graph in binary format or as string.
        
        Reference:
        tensorflow/python/training/training_util.py
        """
    # tf.train.write_graph(session.graph.as_graph_def(), path.join(
    #     ".", "graph"), "{}.pb".format(name), False)
    if isinstance(graph_or_graph_def, ops.Graph):
        graph_def = graph_or_graph_def.as_graph_def()
    else:
        graph_def = graph_or_graph_def
    
    if as_text:
        return str(graph_def)
    else:
        return graph_def.SerializeToString()


#redefine stdout
redefine_stdout()

#data
images_data, label_data = load_data("../../../minimal_dataset/data/images")

#num class
N_CLASS = len(label_data[0])
#num dataset
N_DATASET = len(images_data)
#size data
N_SIZE_DATA = len(images_data[0])

GEN   = 2000
NP    = 200
BATCH = N_DATASET
W     = 0.35
CR    = 0.4
SIZE_W = [N_SIZE_DATA, N_CLASS]
SIZE_B = [N_CLASS]
SIZE_X = [N_SIZE_DATA]

print("|BATCH|: " + str(BATCH) + ", W: " + str(SIZE_W) + ", B" + str(SIZE_B) + ", X:" + str(SIZE_X))
#dataset
#dataset_batch = tf.Variable(tf.zeros([BATCH]+SIZE_X, dtype=np.float64))
dataset_batch_data  = np.array(images_data, np.float64)
dataset_batch_label = np.array(label_data, np.float64)

#W of DE
deW_nnW = np.full(SIZE_W, W)
deW_nnB = np.full(SIZE_B, W)
#random init
create_random_population_W  = tf.random_uniform([NP]+SIZE_W, dtype=tf.float64, name="create_random_population_W")
create_random_population_B  = tf.random_uniform([NP]+SIZE_B, dtype=tf.float64, name="create_random_population_B")


##
# lib
LIB = tf.load_op_library('DENNOp.so')

##
# Placeholder
target_w = tf.placeholder(tf.float64, SIZE_W, name="target_0") # w
target_b = tf.placeholder(tf.float64, SIZE_B, name="target_1") # b

##
# NN
y  = tf.matmul(dataset_batch_data, target_w) + target_b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, dataset_batch_label), name="evaluate")
##
# Random generation
##
# init
init = tf.initialize_all_variables()

tot_time = time()

with tf.Session() as sess:
    # init vars
    sess.run(init)
    ##inid DE
    de_op = LIB.denn(# input params
                     1,
                     [deW_nnW, deW_nnB],
                     [create_random_population_W, create_random_population_B],
                     # attributes
                     # space = 2,
                     graph = get_graph_proto(sess.graph.as_graph_def()),
                     CR = CR,
                     fmin=-1.0,
                     fmax= 1.0
                    )
    results = sess.run(de_op)
    #get output
    w_res = results.final_populations[0]
    b_res = results.final_populations[1]
    c_res = results.final_eval
    #min
    min_cross_id = min(enumerate(c_res), key=itemgetter(1))[0]
    min_cross = c_res[min_cross_id]
    min_res_w = w_res[min_cross_id]
    min_res_b = b_res[min_cross_id]

    print("w: ",min_res_w,"\nb: ",min_res_b,"\ncross_entropy:",min_cross)
    # Test trained model
    y_= dataset_batch_label
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("+ Accuracy: ", sess.run(accuracy, feed_dict={ target_w: min_res_w, target_b: min_res_b }))

