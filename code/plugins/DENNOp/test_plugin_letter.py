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
import DENN

#sleep(6)



#####################################################################

def load_data(path):

    letter_data = []
    label_data = []
    labels = []
    features_max = []
    print('+ Load data ...')
    with open(path+'.data', 'r') as iris_file:
        for line in iris_file.readlines():
            cur_line = [elm.strip() for elm in line.split(',')]
            if len(cur_line) == 17:
                cur_label = cur_line[0]
                if cur_label not in labels:
                    labels.append(cur_label)
                label_data.append(labels.index(cur_label))
                features = [float(elm) for elm in cur_line[1:]]
                if len(features_max) == 0:
                    features_max = [elm for elm in features]
                else:
                    for idx, feature in enumerate(features):
                        if features_max[idx] < feature:
                            features_max[idx] = feature
                letter_data.append(features)
    ##           
    features_max = np.array(features_max, np.float64)
    letter_data = np.divide(np.array(letter_data, np.float64), features_max)
    ##
    # expand labels (one hot vector)
    tmp = np.zeros((len(label_data), len(labels)))
    tmp[np.arange(len(label_data)), label_data] = 1
    label_data = tmp
    #too slow
    #print('+ letters: \n', letter_data)
    #print('+ labels: \n', label_data)
    print('+ loading done!')
    return letter_data, label_data

#####################################################################

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



#new stdout
redefine_stdout()

#data
images_data, label_data = load_data("../../../minimal_dataset/data/letter-recognition")

#num class
N_CLASS = len(label_data[0])
#num dataset
N_DATASET = len(images_data)
#size data
N_SIZE_DATA = len(images_data[0])

GEN   = 50
NP    = N_SIZE_DATA*10
BATCH = N_DATASET
W     = 0.3
CR    = 0.55
DE    = "rand/1/bin"
SIZE_W = [N_SIZE_DATA, N_CLASS]
SIZE_B = [N_CLASS]
SIZE_X = [N_SIZE_DATA]

print("NP: "+ str(NP) +", |BATCH|: " + str(BATCH) + ", W: " + str(SIZE_W) + ", B" + str(SIZE_B) + ", X:" + str(SIZE_X))
#dataset
#dataset_batch = tf.Variable(tf.zeros([BATCH]+SIZE_X, dtype=np.float64))
dataset_batch_data  = np.array(images_data, np.float64)
dataset_batch_label = np.array(label_data, np.float64)

#W of DE
deW_nnW = np.full(SIZE_W, W)
deW_nnB = np.full(SIZE_B, W)
#random init
create_random_population_W  = tf.random_uniform([NP]+SIZE_W, dtype=tf.float64, seed=3, name="create_random_population_W")
create_random_population_B  = tf.random_uniform([NP]+SIZE_B, dtype=tf.float64, seed=9, name="create_random_population_B")

##
# Placeholder
target_w = tf.placeholder(tf.float64, SIZE_W, name="target_0") # w
target_b = tf.placeholder(tf.float64, SIZE_B, name="target_1") # b

##
# NN
y  = tf.matmul(dataset_batch_data, target_w) + target_b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, dataset_batch_label), name="evaluate")

tot_time = time()

with tf.Session() as sess:
    # init vars
    sess.run(tf.global_variables_initializer())
    ##inid DE
    de_op = DENN.create(# input params
                     [GEN, True],
                     [],                                                        #FIRST EVAL
                     [deW_nnW, deW_nnB],                                        #PASS WEIGHTS
                     [create_random_population_W, create_random_population_B],  #POPULATIONS
                     # attributes
                     # space = 2,
                     graph = DENN.get_graph_proto(sess.graph.as_graph_def()),
                     CR = CR,
                     DE = DE,
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
    print("+ GEN: ", GEN)
    print("+ W: ", W)
    print("+ CR: ", CR)
    print("+ NP: ", NP)
    print("+ BATCH: ", BATCH)
    print("+ DE: ", DE)

