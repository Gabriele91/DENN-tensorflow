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

FLAGS = None
OUT_FOLDER = "./benchmark_results"


def main(input_args):
    # Import data
    if FLAGS.dataset == 'mnist':
        dataset = input_data.read_data_sets("./mnist-data", one_hot=True)
    else:
        dataset = DENN.training.Dataset(FLAGS.dataset)
        _, dataset_name = path.split(FLAGS.dataset)

    # Create the model
    x = tf.placeholder(tf.float32, [None, FLAGS.features])
    W = tf.Variable(tf.zeros([FLAGS.features, FLAGS.classes]))
    b = tf.Variable(tf.zeros([FLAGS.classes]))
    y = tf.matmul(x, W) + b

    # from_file = False

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
    y_ = tf.placeholder(tf.float32, [None, FLAGS.classes])

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

    # if from_file:
    #     sess.run([new_W, new_b])
    # Train
    if FLAGS.dataset == 'mnist':
        for _ in tqdm(range(FLAGS.steps), desc="Training official mnist"):
            batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batch_size)
            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_: batch_ys
            })
            # cur_accuracy = sess.run(accuracy, feed_dict={
            #                         x: dataset.test.images, y_: dataset.test.labels})
            # print(cur_accuracy)
        print("Accuracy: {}".format(
            sess.run(accuracy, feed_dict={x: dataset.test.images,
                                          y_: dataset.test.labels})
        ))
    else:
        for num in tqdm(range(FLAGS.steps), desc="Training {}".format(dataset_name)):
            cur_batch = dataset[num]
            sess.run(train_step, feed_dict={
                x: cur_batch.data,
                y_: cur_batch.labels
            })
            # cur_accuracy=sess.run(accuracy, feed_dict={
            #                         x: dataset.test_data, y_: dataset.test_labels})
            # print(cur_accuracy)
        print("Accuracy: {}".format(
            sess.run(accuracy, feed_dict={
                x: dataset.test_data,
                y_: dataset.test_labels
            })
        ))

    ##
    # Get results
    res = sess.run([W, b])
    # tmp = 0.
    # for idx, mat in enumerate(res):
    #     for elm in mat.flat:
    #         if elm < tmp:
    #             tmp = elm
    #     print(np.max(res[idx]), tmp)

    res = {
        "W": res[0].tolist(),
        "b": res[1].tolist()
    }
    with open(path.join(OUT_FOLDER, "gd_results.json"), "w") as result_file:
        json.dump(res, result_file, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-NN', type=str, help='Initial NN values')
    parser.add_argument('--steps', type=int,
                        help='Number of steps', default=1000)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=100)
    parser.add_argument('--features', type=int,
                        help='Number of features', default=784)
    parser.add_argument('--classes', type=int,
                        help='Number of classes', default=10)
    parser.add_argument('--dataset', type=str,
                        help='Dataset to open. If equal to "mnist" it will use the official dataset',
                        default="../datasets/mnist_d_5perV_4000xB_5s.gz")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
