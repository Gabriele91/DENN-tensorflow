from tensorflow.python.framework import ops
from tensorflow import Session
from multiprocessing import Process
from multiprocessing import Event
import socket
import struct
import time
import os
import errno
import fcntl
from select import select

__all__ = ['get_graph_proto', 'get_best_vector', 'OpListener']


CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


class OpListener(object):

    def __init__(self, host='127.0.0.1', port=8484, msg_header="msg"):
        self.db_listener = DebugListner(host, port, msg_header)

    def __enter__(self):
        self.db_listener.start()
        return self.db_listener

    def __exit__(self, ex_type, ex_value, traceback):
        self.db_listener.stop_run()
        print("++ DebugListner: stop to listen and exit", end='\r')
        #stop process
        self.db_listener.join(2.)
        #remove
        #del self.db_listener
        #self.db_listener = None
        #print
        print("++ DebugListner: exited..."+" "*10)


class DebugListner(Process):

    def __init__(self, host, port, msg_header):
        super(DebugListner, self).__init__()
        # print("+ Connect to Op: host->[{}] port->[{}]".format(host, port))
        self._sock = None
        self._connected = False
        self._exit = Event()

        self.host = host
        self.port = port
        self.msg_header = msg_header

        # struct type correspondance
        self.__msg_types = {
            0: ('i', 'int'),
            1: ('f', 'float'),
            2: ('d', 'double'),
            3: ('c', 'string'),
            4: ('', 'close')
        }
        self.__str_to_msg_type = {
            'int'   :0,
            'float' :1,
            'double':2,
            'string':3,
            'close' :4
        }

    def create_socket(self):
        self.close_connection()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        fcntl.fcntl(self._sock, fcntl.F_SETFL, os.O_NONBLOCK)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def stop_run(self):
        self._exit.set()


    def run(self):
        # create a new socket
        self.create_socket()
        # main vars
        res = None
        # not connected
        self._connected = False
        # try to connect
        print("++ DebugListner: connecting", end='\r')
        # exit cases
        ok_res = [errno.EISCONN]
        # wait
        while res != 0 and (not res in ok_res) and (not self._exit.is_set()):
            #try
            res = self._sock.connect_ex((self.host, self.port))
            #OSX, must to recreate socket
            if res == errno.ECONNREFUSED:
                self.create_socket()
            #debug
            if res != 0:
                print("++ DebugListner: connecting({},{})".format(res, os.strerror(res))+" "*10, end='\r')
                time.sleep(0.5)
        # kill thread? 
        if self._exit.is_set():
            return 
        # or connected 
        self._connected = True
        print("++ DebugListner: connected", end='\r')
        print("++ DebugListner: start main loop")
        print(CURSOR_UP_ONE + ERASE_LINE, end='\r')

        while not self._exit.is_set() and self._connected:
            # read
            readables, writables, specials = select([self._sock], [], [], 0.1)
            # print("Available:", readables, writables)
            for readable in readables:
                data = readable.recv(4)
                if data:
                    #get type
                    type_ = struct.unpack("<i", data)[0]
                    #return
                    if type_ in self.__msg_types:
                        msg = self.read_msg(readable, self.__msg_types[type_])
                        print("++ [{}]-> {}".format(self.msg_header, msg), end='\r')
        # close connection (anyway)
        self.close_connection()

    def __del__(self):
        self.close_connection()

    def close_connection(self):
        #close socket
        if self._sock != None:
            if self._connected:
                self._sock.setblocking(True)
                try: 
                    self._sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                finally:
                    self._sock.close()
                self._connected = False
            else:
                self._sock.close()
        #delete socket
        self._sock = None

    def send_close_message(self):
        self._sock.send(struct.pack('<i',self.__str_to_msg_type['close']))

    @staticmethod
    def read_msg(conn, type_):
        if type_[1] == 'string':
            size = struct.unpack("<i", conn.recv(4))[0]
            data = conn.recv(struct.calcsize(type_[0]) * size)
            return struct.unpack("<{}".format(type_[0] * size), data)[0]
        if type_[1] == 'close':
            return (None,'close')
        else:
            data = conn.recv(struct.calcsize(type_[0]))
            return struct.unpack("<{}".format(type_[0]), data)[0]


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
