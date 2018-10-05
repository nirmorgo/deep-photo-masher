'''
This file contains all the net architecture functions
'''

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose

def build_full_conv_autoencoder(self):
    # encoder
    with tf.variable_scope('conv1'):
        conv1 = _conv_layer(self.X, num_filters=32, filter_size=9, strides=1)
    with tf.variable_scope('conv2'):
        conv2 = _conv_layer(conv1, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('conv3'):
        conv3 = _conv_layer(conv2, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('conv4'):
        conv4 = _conv_layer(conv3, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('res_block1'):
        resid1 = _residual_block(conv4, filter_size=3, filter_num=32)
    with tf.variable_scope('res_block2'):
        resid2 = _residual_block(resid1, filter_size=3, filter_num=32)
    
    # embedded space
    self.z = resid2

    # decoder
    with tf.variable_scope('res_block3'):
        resid3 = _residual_block(self.z, filter_size=3, filter_num=32)
    with tf.variable_scope('res_block4'):
        resid4 = _residual_block(resid3, filter_size=3, filter_num=32)
    with tf.variable_scope('upsample1'):
        upsample1 = _upsample(resid4, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('upsample2'):
        upsample2 = _upsample(upsample1, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('upsample3'):
        upsample3 = _upsample(upsample2, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('smoothing'):
        self.net_out = _conv_layer(upsample3, num_filters=3, filter_size=3, strides=1, relu=False)



def _conv_layer(net, num_filters, filter_size, strides, relu=True, name='conv2d'):
    net = conv2d(net, num_filters, filter_size, 
                                         strides, padding='SAME', 
                                         activation_fn=None,
                                         scope=name)
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net


def _residual_block(net, filter_size=3, filter_num=128):
    tmp = _conv_layer(net, filter_num, filter_size, 1, name='conv2d_1')
    return net + _conv_layer(tmp, filter_num, filter_size, 1, relu=False, name='conv2d_2')


def _instance_norm(net, train=True):
    with tf.variable_scope('instance_norm'):
        _, _, _, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        out = scale * normalized + shift
    return out


def _upsample(net, num_filters, filter_size, strides):
    H = tf.shape(net)[1]
    W = tf.shape(net)[2]
    net = tf.image.resize_nearest_neighbor(net,(H*strides, W*strides),
                                               align_corners=False, name='resize')
    net = conv2d(net, num_filters, filter_size, 1, padding='SAME',
                 activation_fn=None, scope='conv2d')
    net = _instance_norm(net)
    return tf.nn.relu(net)