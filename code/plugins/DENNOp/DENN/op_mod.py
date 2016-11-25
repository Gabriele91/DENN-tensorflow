import tensorflow as tf
from os import path

__all__ = ['create']


def create(*args, **kwargs):
    """Create a DENN object"""
    training = kwargs.get("training", False)

    module = None

    if not training:
        module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp.so')
        )
    else:
        module = tf.load_op_library(path.join(
            path.dirname(__file__), 'DENNOp_training.so')
        )

    del kwargs['training']

    return module.denn(*args, **kwargs)
