import argparse
import sys
sys.path.append("../")  # To import DENN
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import DENN
from tqdm import tqdm
from os import path
import json
from collections import namedtuple

FLAGS = None
OUT_FOLDER = "./benchmark_results"

BinClassify = namedtuple('BinClassify', ['TP', 'FP', 'FN', 'TN'])


def calc_TF(confusion_m, class_):
    TP = confusion_m[class_][class_]
    FP = np.sum(confusion_m[class_]) - TP
    FN = np.sum(confusion_m[:, class_]) - TP
    TN = np.sum(confusion_m.diagonal()) - TP

    return BinClassify(TP, FP, FN, TN)


def precision_recall_acc(bc):
    precision = bc.TP / (bc.TP + bc.FP)
    recall = bc.TP / (bc.TP + bc.FN)

    acc = (bc.TP + bc.TN) / np.sum(bc)

    return precision, recall, acc


def calc_confusin_M(labels, cur_y):
    size = labels.shape[-1]

    labels = [np.argmax(row) for row in labels]
    cur_y = [np.argmax(row) for row in cur_y]

    matrix = np.full((size, size), 0, dtype=np.int32)

    for idx, class_ in enumerate(labels):
        matrix[class_][cur_y[idx]] += 1

    return matrix.astype(int)


def main(input_args):
    # Import data
    if FLAGS.dataset == 'mnist':
        dataset = input_data.read_data_sets("./mnist-data", one_hot=True)
    else:
        dataset = DENN.training.Dataset(FLAGS.dataset)
        _, dataset_name = path.split(FLAGS.dataset)

    if FLAGS.type == 'float':
        data_type = tf.float32
    elif FLAGS.type == 'double':
        data_type = tf.float64
    else:
        raise Exception("Data type {} is not valid!".format(FLAGS.type))

    # Create the model
    x = tf.placeholder(data_type, [None, FLAGS.features])
    _Ws = []
    _bs = []
    y = None

    if FLAGS.hidden > 0:
        _Ws.append(tf.Variable(tf.zeros(
            [FLAGS.features, FLAGS.features * 2], dtype=data_type), dtype=data_type, name="input_W"))
        _bs.append(tf.Variable(tf.zeros(
            [FLAGS.features * 2], dtype=data_type), dtype=data_type, name="input_b"))

        for lvl in range(FLAGS.hidden - 1):
            _Ws.append(tf.Variable(tf.zeros([FLAGS.features * 2, FLAGS.features * 2],
                                            dtype=data_type), dtype=data_type, name="hidden_{}".format(lvl + 1)))
            _bs.append(tf.Variable(tf.zeros(
                [FLAGS.features * 2], dtype=data_type), dtype=data_type, name="hidden_{}".format(lvl)))

        _Ws.append(tf.Variable(tf.zeros(
            [FLAGS.features * 2, FLAGS.classes], dtype=data_type), dtype=data_type, name="output_W"))
        _bs.append(tf.Variable(
            tf.zeros([FLAGS.classes], dtype=data_type), dtype=data_type, name="output_b"))

        # print("Num layers: ", len(_Ws))
        for lvl in range(len(_Ws)):
            # print("-[{}] ".format(lvl), _Ws[lvl])
            # print("-[{}] ".format(lvl), _bs[lvl])
            if lvl == 0:
                y = tf.nn.sigmoid(tf.matmul(x, _Ws[lvl]) + _bs[lvl])
            elif lvl < len(_Ws) - 1:
                y = tf.nn.sigmoid(tf.matmul(y, _Ws[lvl]) + _bs[lvl])
            else:
                y = tf.matmul(y, _Ws[lvl]) + _bs[lvl]
    else:
        _Ws.append(tf.Variable(
            tf.zeros([FLAGS.features, FLAGS.classes], dtype=data_type), dtype=data_type))
        _bs.append(tf.Variable(
            tf.zeros([FLAGS.classes], dtype=data_type), dtype=data_type))
        y = tf.matmul(x, _Ws[0]) + _bs[0]

    # from_file = False

    # print(y)

    # if from_file:
    #     with open("test_results_with_smooth.json", "r") as input_file:
    #         import json
    #         res = json.load(input_file)
    #         new_W = W.assign(
    #             np.array(res['results']['rand/1/bin']['best_of']['individual'][0]))
    #         # new_b = b.assign(
    #         #     np.array(res['results']['rand/1/bin']['best_of']['individual'][1]))
    #         new_b = b.assign(np.zeros(FLAGS.classes))

    # Define loss and optimizer
    y_ = tf.placeholder(data_type, [None, FLAGS.classes])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    step_accuracies = []

    # if from_file:
    #     sess.run([new_W, new_b])
    # Train
    if FLAGS.dataset == 'mnist':
        for step in tqdm(range(FLAGS.steps), desc="Training official mnist"):
            batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batch_size)
            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_: batch_ys
            })
            if step % FLAGS.batch_size == 0:
                cur_accuracy = sess.run(accuracy, feed_dict={
                    x: dataset.test.images,
                    y_: dataset.test.labels
                })
                step_accuracies.append([num, cur_accuracy])
            # cur_accuracy = sess.run(accuracy, feed_dict={
            #                         x: dataset.test.images, y_: dataset.test.labels})
            # print(cur_accuracy)
        cur_accuracy = sess.run(accuracy, feed_dict={
            x: dataset.test.images,
            y_: dataset.test.labels
        })
        print("Accuracy: {}".format(cur_accuracy))
        cur_y = sess.run(y, feed_dict={
            x: dataset.test.images,
            y_: dataset.test.labels
        })
        confusion_matrix = calc_confusin_M(dataset.test.labels, cur_y)
        print("Confusion Matrix:\n", confusion_matrix)
    else:
        batch_size = len(dataset[0].data)
        batch_index = 0
        for num in tqdm(range(FLAGS.steps), desc="Training {}".format(dataset_name)):
            cur_batch = dataset[batch_index]
            sess.run(train_step, feed_dict={
                x: cur_batch.data,
                y_: cur_batch.labels
            })
            if num % batch_size == 0:
                cur_accuracy = sess.run(accuracy, feed_dict={
                    x: dataset.test_data,
                    y_: dataset.test_labels
                })
                step_accuracies.append([num, float(cur_accuracy)])
            if FLAGS.change_batch_each_step:
                batch_index += 1
            elif num % batch_size == 0 and num > 0:
                batch_index += 1
            # cur_accuracy=sess.run(accuracy, feed_dict={
            #                         x: dataset.test_data, y_: dataset.test_labels})
            # print(cur_accuracy)
        cur_accuracy = sess.run(accuracy, feed_dict={
            x: dataset.test_data,
            y_: dataset.test_labels
        })
        print("Accuracy: {}".format(cur_accuracy))
        cur_y = sess.run(y, feed_dict={
            x: dataset.test_data,
            y_: dataset.test_labels
        })
        confusion_matrix = calc_confusin_M(dataset.test_labels, cur_y)
        print("Confusion Matrix:\n", confusion_matrix)

    ##
    # Get results
    res = sess.run([_Ws, _bs])
    # tmp = 0.
    # for idx, mat in enumerate(res):
    #     for elm in mat.flat:
    #         if elm < tmp:
    #             tmp = elm
    #     print(np.max(res[idx]), tmp)
    stats = []
    for class_ in range(confusion_matrix.shape[0]):
        elm_tf = calc_TF(confusion_matrix, class_)
        stats.append(precision_recall_acc(elm_tf))
        print(elm_tf)

    all_W = [elm.tolist() for elm in res[0]]
    all_b = [elm.tolist() for elm in res[1]]

    res = {
        "W": all_W[0] if len(all_W) == 1 else all_W,
        "b": all_b[0] if len(all_b) == 1 else all_b,
        "accuracy": float(cur_accuracy),
        "confusionM": confusion_matrix.tolist(),
        "stats": stats,
        "steps": step_accuracies
    }
    with open(path.join(OUT_FOLDER, FLAGS.outfilename), "w") as result_file:
        json.dump(res, result_file, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-NN', type=str, help='Initial NN values')
    parser.add_argument('--steps', type=int,
                        help='Number of steps', default=1000)
    parser.add_argument('--change-batch-each-step', type=lambda x: x == "true" or x == "True",
                        help='Tell if the the batch have to be changed each step', default=True,
                        dest="change_batch_each_step")
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=100)
    parser.add_argument('--features', type=int,
                        help='Number of features', default=784)
    parser.add_argument('--type', type=str,
                        help='Data type', default='float')
    parser.add_argument('--classes', type=int,
                        help='Number of classes', default=10)
    parser.add_argument('--hidden', type=int,
                        help='Number of hidden layers (default size is double of the input and connected with a sigmoid)', default=0)
    parser.add_argument('--outfilename', type=str,
                        help='Output name', default="gd_results.json")
    parser.add_argument('--dataset', type=str,
                        help='Dataset to open. If equal to "mnist" it will use the official dataset',
                        default="../datasets/mnist_d_5perV_4000xB_5s.gz")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
