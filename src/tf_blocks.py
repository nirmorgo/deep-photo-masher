import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose


def conv_inst_norm(net, num_filters, filter_size, strides, relu=True, name='conv2d'):
    net = conv2d(net, num_filters, filter_size, 
                                         strides, padding='SAME', 
                                         activation_fn=None,
                                         scope=name)
    net = instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net

def deconv_inst_norm(net, num_filters, filter_size, strides, relu=True, name='conv2d'):
    net = conv2d_transpose(net, num_filters, filter_size, 
                                                    strides, padding='SAME', 
                                                    activation_fn=None,
                                                    scope=name)
    net = instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net

def residual_block(net, filter_size=3, filter_num=128):
    tmp = conv_inst_norm(net, filter_num, filter_size, 1, name='conv2d_1')
    return net + conv_inst_norm(tmp, filter_num, filter_size, 1, relu=False, name='conv2d_2')


def instance_norm(net, train=True):
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


def upsample(net, num_filters, filter_size, strides, inst_norm=True):
    H = tf.shape(net)[1]
    W = tf.shape(net)[2]
    net = tf.image.resize_nearest_neighbor(net,(H*strides, W*strides),
                                               align_corners=False, name='resize')
    net = conv2d(net, num_filters, filter_size, 1, padding='SAME',
                 activation_fn=None, scope='conv2d')
    if inst_norm:
        net = instance_norm(net)
    return tf.nn.relu(net)