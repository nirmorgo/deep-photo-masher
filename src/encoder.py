import tensorflow as tf 
import numpy as np

from src.net import autoencoder

class AE():
    def __init__(self, **kwargs):
        self.sess = get_session()
        if kwargs is not None:
            self.params = kwargs
        self.temp_folder = './tmp/'

        with tf.variable_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, [None,None,None,3], name='X')
            self.global_step = tf.Variable(0, name='global_step', trainable=False) 

        with tf.variable_scope('AutoEncoder'):
            self.net = autoencoder(self)


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session