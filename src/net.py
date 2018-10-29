'''
This file contains all the net architecture functions
'''

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected

from src.tf_blocks import conv_inst_norm, residual_block, upsample

def build_vae_32(self):
    '''
    build a variational autoencoder with convolutional layers instance norms.
    the input image size must be 32 X 32
    similar architecture to https://arxiv.org/pdf/1610.00291.pdf with some 
    minor modifications, for example: used instance norm instead of batch norm
    '''
    # encoder
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('conv1'):
            net = conv_inst_norm(self.X, num_filters=64, filter_size=4, strides=2, leaky_relu=True)
        with tf.variable_scope('conv2'):
            net = conv_inst_norm(net, num_filters=128, filter_size=4, strides=2, leaky_relu=True)  
        with tf.variable_scope('conv3'):
            net = conv_inst_norm(net, num_filters=256, filter_size=4, strides=2, leaky_relu=True)          
        with tf.variable_scope('Flatten'):
            encoder_out = tf.layers.flatten(net)
       
    # embedded space
    with tf.variable_scope('embedded_space'):
        self.z_mu = fully_connected(encoder_out, 512, activation_fn=None, scope='z_mean')
        self.z_log_sigma_sq = fully_connected(encoder_out, 512, activation_fn=None, scope='z_sigma') + 1e-6
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.exp(self.z_log_sigma_sq) * eps

    # decoder
    with tf.variable_scope('Decoder'):
        with tf.variable_scope('reshape'):
            net = fully_connected(self.z, 4096, activation_fn=None)
            net = tf.reshape(net, (tf.shape(net)[0], 4, 4, 256))
        with tf.variable_scope('upsample1'):
            net = upsample(net, num_filters=128, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('upsample2'):
            net = upsample(net, num_filters=64, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('upsample3'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('smoothing'):
            net = conv2d(net, 3, 3, 1, padding='SAME', activation_fn=None)
        
        self.net_out = tf.nn.tanh(net)
        
        
def build_vae_64(self):
    '''
    build a variational autoencoder with convolutional layers instance norms.
    the input image size must be 64 X 64
    similar architecture to https://arxiv.org/pdf/1610.00291.pdf with some 
    minor modifications, for example: used instance norm instead of batch norm
    '''
    # encoder
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('conv1'):
            net = conv_inst_norm(self.X, num_filters=32, filter_size=4, strides=2, leaky_relu=True)
        with tf.variable_scope('conv2'):
            net = conv_inst_norm(net, num_filters=64, filter_size=4, strides=2, leaky_relu=True)
        with tf.variable_scope('conv3'):
            net = conv_inst_norm(net, num_filters=128, filter_size=4, strides=2, leaky_relu=True)  
        with tf.variable_scope('conv4'):
            net = conv_inst_norm(net, num_filters=256, filter_size=4, strides=2, leaky_relu=True)          
        with tf.variable_scope('Flatten'):
            encoder_out = tf.layers.flatten(net)
       
    # embedded space
    with tf.variable_scope('embedded_space'):
        self.z_mu = fully_connected(encoder_out, 512, activation_fn=None, scope='z_mean')
        self.z_log_sigma_sq = fully_connected(encoder_out, 512, activation_fn=None, scope='z_sigma') + 1e-6
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.exp(self.z_log_sigma_sq) * eps

    # decoder
    with tf.variable_scope('Decoder'):
        with tf.variable_scope('reshape'):
            net = fully_connected(self.z, 4096, activation_fn=None)
            net = tf.reshape(net, (tf.shape(net)[0], 4, 4, 256))
        with tf.variable_scope('upsample1'):
            net = upsample(net, num_filters=128, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('upsample2'):
            net = upsample(net, num_filters=64, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('upsample3'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('upsample4'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2, leaky_relu=True)
        with tf.variable_scope('smoothing'):
            net = conv2d(net, 3, 3, 1, padding='SAME', activation_fn=None)
        
        self.net_out = tf.nn.tanh(net)

def build_cifar10_vae(self):
    '''
    a basic vanila version of a vae for cifar10
    used to initialize the net architecture of an VAE() object
    The current architecture is quite big though, and it has no convolutional up/downsampling. probably an overkill.
    '''
    with tf.variable_scope('conv1'):
        net = conv2d(self.X, 3, 2, 1, padding='SAME', activation_fn=tf.nn.relu)
    with tf.variable_scope('conv2'):
        net = conv2d(net, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
    with tf.variable_scope('conv3'):
        net = conv2d(net, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
    with tf.variable_scope('conv4'):
        net = conv2d(net, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
    with tf.variable_scope('flatten'):
        net = tf.layers.flatten(net)
    with tf.variable_scope('dense_1'):
            net = tf.layers.dense(net, units=512, activation=tf.nn.relu)
    
    # embedded space
    with tf.variable_scope('embedded_space'):
        with tf.variable_scope('z_mu'):
            self.z_mu = tf.layers.dense(net, units=512)
        with tf.variable_scope('z_log_sigma_sq'):
            self.z_log_sigma_sq = tf.layers.dense(net, units=512)
        with tf.variable_scope('noise'):
            eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        with tf.variable_scope('z'):
            self.z = self.z_mu + tf.exp(self.z_log_sigma_sq) * eps
    
    # decoder
    with tf.variable_scope('dense3'):
        net = tf.layers.dense(self.z, units=32*32*32, activation=tf.nn.relu)
    with tf.variable_scope('reshape'):
        net = tf.reshape(net, [-1, 32, 32, 32])
    with tf.variable_scope('conv5'):
        net = conv2d(net, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
    with tf.variable_scope('conv6'):
        net = conv2d(net, 32, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
#    with tf.variable_scope('upsample'):
#        net = _upsample(net, 32, filter_size=3, strides=2, inst_norm=False)
    with tf.variable_scope('final_conv'):
        net = conv2d(net, 3, 3, 1, padding='SAME', activation_fn=None)
    
    self.net_out = net
        

