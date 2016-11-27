from tensorflow.python.framework import ops
from tensorflow import Session
from multiprocessing import Process
import socket
import struct
from select import select

__all__ = ['get_graph_proto', 'get_best_vector', 'OpListener']


class OpListener(object):

    def __init__(self, host='', port=6545):
        self.db_listner = DebugListner(host, port)

    def __enter__(self):
        self.db_listner.start()
        return self.db_listner

    def __exit__(self, ex_type, ex_value, traceback):
        self.db_listner.stop_run()
        self.db_listner.join()


class DebugListner(Process):

    def __init__(self, host, port):

        super(DebugListner, self).__init__()

        print(
            "+ Connect to Op: host->[{}] port->[{}]".format(host, port))
        self._sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_NONBLOCK)
        self._sock.setblocking(False)

        self._run = True

        self.host = host
        self.port = port

        # struct type correspondance
        self.__msg_types = {
            0: 'i',
            1: 'f',
            2: 'd',
            3: 'c',
        }

    def stop_run(self):
        self._run = False
        return self._run

    def run(self):

        print("+ DebugListner: start main loop")

        res = None

        while res != 0 and self._run:
            res = self._sock.connect_ex((self.host, self.port))

        print("+ DebugListner: connected")

        while self._run:

            readables, writables, specials = select(
                [self._sock], [], [])

            # print("Available:", readables, writables)

            for readable in readables:
                data = readable.recv(4)
                if data:
                    type_ = struct.unpack("<i", data)[0]
                    #print("+ data -> {} -> {}".format(data, type_))
                    if type_ in self.__msg_types:
                        #print(type_)
                        msg = self.read_msg(
                            readable, self.__msg_types[type_])
                        print("+ [msg]-> {}".format(msg))

    def __del__(self):
        # self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()

    @staticmethod
    def read_msg(conn, type_):
        if type_ != 'c':
            data = conn.recv(struct.calcsize(type_))
            return struct.unpack("<{}".format(type_), data)[0]
        else:
            size = struct.unpack("<i", conn.recv(4))[0]
            data = conn.recv(struct.calcsize(type_) * size)
            return struct.unpack("<{}".format(type_ * size), data)[0]


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


def get_best_vector(results, f_x, target):
    """Return best vector due f_x.

    Params:
        results (numpy array, tf result)
        f_x (function written in tf)
        target (placeholder for f(x) input)
    """
    index_min = -1
    cur_min = 10**10

    with Session() as sess:
        for index, individual in enumerate(results):
            f_x_res = sess.run(f_x, feed_dict={
                target: individual
            })
            if f_x_res < cur_min:
                cur_min = f_x_res
                index_min = index

    best = results[index_min]

    return best
