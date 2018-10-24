'''
This file contains all the net architecture functions
'''

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected

from src.tf_blocks import conv_inst_norm, residual_block, upsample

def build_full_conv_autoencoder(self):
    # encoder
    with tf.variable_scope('conv1'):
        conv1 = conv_inst_norm(self.X, num_filters=32, filter_size=9, strides=1)
    with tf.variable_scope('conv2'):
        conv2 = conv_inst_norm(conv1, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('conv3'):
        conv3 = conv_inst_norm(conv2, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('conv4'):
        conv4 = conv_inst_norm(conv3, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('res_block1'):
        resid1 = residual_block(conv4, filter_size=3, filter_num=32)
    with tf.variable_scope('res_block2'):
        resid2 = residual_block(resid1, filter_size=3, filter_num=32)
    
    # embedded space
    self.z = resid2

    # decoder
    with tf.variable_scope('res_block3'):
        resid3 = residual_block(self.z, filter_size=3, filter_num=32)
    with tf.variable_scope('res_block4'):
        resid4 = residual_block(resid3, filter_size=3, filter_num=32)
    with tf.variable_scope('upsample1'):
        upsample1 = upsample(resid4, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('upsample2'):
        upsample2 = upsample(upsample1, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('upsample3'):
        upsample3 = upsample(upsample2, num_filters=32, filter_size=3, strides=2)
    with tf.variable_scope('smoothing'):
        self.net_out = conv_inst_norm(upsample3, num_filters=3, filter_size=3, strides=1, relu=False)


def build_vae_128(self):
    '''
    build a variational autoencoder with convolutional layers instance norms.
    the input image size must be 128 X 128
    '''
    # encoder
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('conv1'):
            net = conv_inst_norm(self.X, num_filters=32, filter_size=9, strides=1)
        with tf.variable_scope('conv2'):
            net = conv_inst_norm(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('res_block1'):
            net = residual_block(net, filter_size=3, filter_num=32)
        with tf.variable_scope('conv3'):
            net = conv_inst_norm(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('res_block2'):
            net = residual_block(net, filter_size=3, filter_num=32)   
        with tf.variable_scope('conv4'):
            net = conv_inst_norm(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('res_block3'):
            net = residual_block(net, filter_size=3, filter_num=32)
        with tf.variable_scope('conv5'):
            conv_out = conv_inst_norm(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('Flatten'):
            encoder_out = tf.layers.flatten(conv_out)
        
    # embedded space
    with tf.variable_scope('embedded_space'):
        self.z_mu = fully_connected(encoder_out, 128, activation_fn=None, scope='z_mean')
        self.z_log_sigma_sq = fully_connected(encoder_out, 128, activation_fn=None, scope='z_sigma')
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

    # decoder
    with tf.variable_scope('Decoder'):
        with tf.variable_scope('reshape'):
            net = fully_connected(self.z, 2048)
            net = tf.reshape(net, (tf.shape(net)[0], 8, 8, 32))
        with tf.variable_scope('deconv1'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('deconv2'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('res_block4'):
            net = residual_block(net, filter_size=3, filter_num=32)
        with tf.variable_scope('res_block5'):
            net = residual_block(net, filter_size=3, filter_num=32)
        with tf.variable_scope('deconv3'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('upsample4'):
            net = upsample(net, num_filters=32, filter_size=3, strides=2)
        with tf.variable_scope('smoothing'):
            self.net_out = conv_inst_norm(net, num_filters=3, filter_size=3, strides=1, relu=False)



def build_vanila_cifar10_vae(self):
    '''
    a basic vanila version of a vae for cifar10
    used to initialize the net architecture of an AE() object
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
        

