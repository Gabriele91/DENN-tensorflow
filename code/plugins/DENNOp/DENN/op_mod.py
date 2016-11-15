import tensorflow as tf
from os import path

__all__ = ['create']

MODULE = tf.load_op_library(path.join(
    path.dirname(__file__), 'DENNOp.so')
)


def create(*args, **kwargs):
    """Create a DENN object"""
    return MODULE.denn(*args, **kwargs)
