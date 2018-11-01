import numpy as np
import scipy.misc
import scipy.io
import tensorflow as tf

VGG_MODEL = 'saved_models/VGG19/imagenet-vgg-verydeep-19.mat'
# The mean to subtract from the input to the VGG model. This is the mean that
# when the VGG was used to train. Minor changes to this will make a lot of
# difference to the performance of model.
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

class VGG19():
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    """
    def __init__(self, data_path=VGG_MODEL):
        self.data_path = data_path
        self.data = scipy.io.loadmat(data_path)
        self.mean_pixel = MEAN_VALUES
        self.layers = self.data['layers']
    
    def _weights(self, layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        # Parsing the Mat file to get the pre-trained weights values. kind of ugly, i know...
        wb = self.layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = self.layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _conv2d_relu(self, prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        with tf.variable_scope(layer_name):
            W, b = self._weights(layer, layer_name)
            with tf.variable_scope('weights'):
                W = tf.constant(W)
                b = tf.constant(np.reshape(b, (b.size)))
            out = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
            out = tf.nn.relu(out)
        return out

    def _avgpool(self, prev_layer, layer_name):
        """
        Return the AveragePooling layer.
        """
        with tf.variable_scope(layer_name):
            out = tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return out
    
    def preprocess(self, image):
        '''
        transfer from [-1,1] range to [0,255], and normalize by VGG19 mean value
        '''
        return (255/2) * image + (255/2) - self.mean_pixel

    def unprocess(self, image):
        return image + self.mean_pixel
    
    def net(self, img):
    # Constructs the graph model.
        graph = {}
        graph['input'] = self.preprocess(img)
        graph['conv1_1']  = self._conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2']  = self._conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self._avgpool(graph['conv1_2'], 'avgpool1')
        graph['conv2_1']  = self._conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = self._conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self._avgpool(graph['conv2_2'], 'avgpool2')
        graph['conv3_1']  = self._conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = self._conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = self._conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = self._conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self._avgpool(graph['conv3_4'],'avgpool3')
        graph['conv4_1']  = self._conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = self._conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = self._conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = self._conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self._avgpool(graph['conv4_4'], 'avgpool4')
        graph['conv5_1']  = self._conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = self._conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = self._conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = self._conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self._avgpool(graph['conv5_4'], 'avgpool5')
        return graph