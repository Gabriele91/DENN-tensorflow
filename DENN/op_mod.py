import tensorflow as tf
from os import path

__all__ = ['create']


class create(object):
    """Create a DENN object."""

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        self.__module = None

    def __call__(self, *args, **kwargs):
        module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp.so')
        )
        return module.denn(*args, **kwargs)
    
    @property
    def standard(self):
        self.__module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp.so')
        )
        return self.__module.denn(*self.__args, **self.__kwargs)
    
    @property
    def train(self):
        self.__module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp_training.so')
        )
        return self.__module.denn(*self.__args, **self.__kwargs)
    
    @property
    def ada(self):
        self.__module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp_ada.so')
        )
        return self.__module.denn(*self.__args, **self.__kwargs)
