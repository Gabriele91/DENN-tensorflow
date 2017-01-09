import DENN
import tensorflow as tf
from tensorflow.python.client import device_lib

import numpy as np
from os import path
from sys import argv
from time import sleep
from time import time

#from memory_profiler import profile

# sleep(6)


class ENDict(dict):

    def __init__(self, *args, **kwargs):
        super(ENDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class NDict(dict):
    pass


#@profile
def main():
    ##
    # Select device
    DEVICE = None
    NUM_THREADS = 4

    session_config = tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS,
        inter_op_parallelism_threads=NUM_THREADS
    )

    ##
    # jobs
    jobs = []

    jobs = DENN.training.open_task_list(argv[1])

    # test results
    results_tests = []

    # for all jobs first
    for job in jobs:
        # start
        print("+ Start tests")
        # network
        cur_nn = job.gen_network(
            True  # rand population only if gen is the first one
        )
        # network graph
        with cur_nn.graph.as_default():
            # start session
            with tf.Session(config=session_config) as sess:
                ##
                # init vars
                # We don't need this, we don't have variables at the moment
                # sess.run(tf.global_variables_initializer())
                for de_type in job.de_types:
                    ##
                    # Random initialization of the NN
                    cur_pop = sess.run(cur_nn.rand_pop)
                    # take time
                    time_node_creation = time()
                    with tf.device("/cpu:0"):
                        ##
                        # DENN op
                        denn_op = DENN.create(
                            # input params
                            # [num_gen, step_gen, eval_individual]
                            [job.TOT_GEN, job.GEN_STEP, False],
                            np.array([], dtype=np.float64 if job.TYPE ==
                                     "double" else np.float32),  # FIRST EVAL
                            cur_nn.weights,  # PASS WEIGHTS
                            cur_nn.populations,  # POPULATIONS
                            # attributes
                            # space = 2,
                            graph=DENN.get_graph_proto(
                                cur_nn.graph.as_graph_def()),
                            dataset=job.dataset_file,
                            f_name_train=cur_nn.cross_entropy.name,
                            f_name_validation=cur_nn.accuracy.name,
                            f_name_test=cur_nn.accuracy.name,
                            f_inputs=[elm.name for elm in cur_nn.targets],
                            f_input_labels=cur_nn.label_placeholder.name,
                            f_input_features=cur_nn.input_placeholder.name,
                            CR=job.CR,
                            DE=de_type,
                            training=True
                        )

                    print(
                        "++ Node creation {}".format(time() - time_node_creation))
                    # Soket listener
                    with DENN.OpListener() as listener:
                        # time
                        time_start_gen = time()
                        # session run
                        current_result = sess.run(denn_op, feed_dict=dict(
                            [
                                (pop_ref, cur_pop[num])
                                for num, pop_ref in enumerate(cur_nn.populations)
                            ]
                        ))
                        run_time = time() - time_start_gen
                        print("++ Op time {}".format(run_time))
                        #######################################################
                        # Open dataset
                        dataset = DENN.training.Dataset(job.dataset_file)
                        # test time
                        time_test = time()
                        # test result
                        cur_accuracy = sess.run(cur_nn.accuracy, feed_dict=dict(
                            [
                                (target, current_result[num])
                                for num, target in enumerate(cur_nn.targets)
                            ]
                            +
                            [
                                (cur_nn.label_placeholder,
                                 dataset.test_labels),
                                (cur_nn.input_placeholder, dataset.test_data)
                            ]
                        ))
                        test_time = time() - time_test
                        print(
                            "++ Test {}, result {}".format(test_time, cur_accuracy))
                        #######################################################

                        job.time = run_time + test_time
                        job.accuracy = cur_accuracy
                        job.best = [arr.tolist()
                                    for arr in current_result]

                        with open(path.join("benchmark_results", argv[1]), "w") as out_file:
                            out_file.write(DENN.training.task_dumps(job))
                        ###################################################
        # clena graph
        tf.reset_default_graph()


if __name__ == '__main__':
    main()
