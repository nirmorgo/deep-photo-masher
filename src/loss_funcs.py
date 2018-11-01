'''
collection of usefull loss functions 
'''

import tensorflow as tf
from src.vgg19net import VGG19

def l1_loss(img_in, img_out):
    '''
    Compute l1 loss.
    
    Inputs:
    - img_in: Tensor of shape (B, H, W, 3) holding an input image.
    - img_out: Tensor of shape (B, H, W, 3) holding an VAE output image.
    
    Returns:
    - loss: Tensor holding a scalar giving the l1 loss
    '''
    with tf.variable_scope('inputs_shape'):
        N = tf.shape(img_in)[0]
        H = tf.shape(img_in)[1]
        W = tf.shape(img_in)[2]
        C = tf.shape(img_in)[3]
        elem_num = tf.cast(N*H*W*C, tf.float32)
    
    with tf.variable_scope('abs_diff'):
        l1_loss = tf.losses.absolute_difference(img_in, img_out) / elem_num
    return l1_loss


def l2_loss(img_in, img_out):
    '''
    Compute l2 loss.
    
    Inputs:
    - img_in: Tensor of shape (B, H, W, 3) holding an input image.
    - img_out: Tensor of shape (B, H, W, 3) holding an VAE output image.
    
    Returns:
    - loss: Tensor holding a scalar giving the l2 loss
    '''
    with tf.variable_scope('inputs_shape'):
        N = tf.shape(img_in)[0]
        H = tf.shape(img_in)[1]
        W = tf.shape(img_in)[2]
        C = tf.shape(img_in)[3]
        elem_num = tf.cast(N*H*W*C, tf.float32)
        
    with tf.variable_scope('L2_loss'):
        l2_loss = tf.nn.l2_loss(img_out - img_in) / elem_num
    return l2_loss


def tv_loss(img):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (B, H, W, 3) holding an input image.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    with tf.variable_scope('inputs_shape'):
        N = tf.shape(img)[0]
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]

    def _tensor_size(tensor):
        H = tf.shape(tensor)[1]
        W = tf.shape(tensor)[2]
        C = tf.shape(tensor)[3]
        return tf.cast(H*W*C, tf.float32)
    
    tv_y_size = _tensor_size(img[:,1:,:,:])
    tv_x_size = _tensor_size(img[:,:,1:,:])
    imgoy = tf.slice(img, [0,0,0,0],[N,H-1,W,3])
    imgy = tf.slice(img, [0,1,0,0],[N,H-1,W,3])
    imgox = tf.slice(img, [0,0,0,0],[N,H,W-1,3])
    imgx = tf.slice(img, [0,0,1,0],[N,H,W-1,3])
    y_tv = tf.nn.l2_loss(imgoy-imgy)
    x_tv = tf.nn.l2_loss(imgox-imgx)
    tv_loss = 2*(x_tv/tv_x_size + y_tv/tv_y_size)/tf.cast(N,tf.float32)
    return tv_loss


def kl_div_loss(img_in, z_mu, z_log_sigma_sq):
    with tf.variable_scope('inputs_shape'):
        N = tf.shape(img_in)[0]
        H = tf.shape(img_in)[1]
        W = tf.shape(img_in)[2]
        C = tf.shape(img_in)[3]
        elem_num = tf.cast(N*H*W*C, tf.float32)
        
    KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq),1))
    KL_loss = KL_loss / elem_num
    return KL_loss


def content_loss(content_current, content_target):
    """
    Compute the content loss from layers of a pre-trained CNN.
    
    Inputs:
    - content_current: features of the current image, Tensor with shape [batch_size, height, width, channels]
    - content_target: features of the content image, Tensor with shape [batch_size, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    B = tf.shape(content_current)[0]
    H = tf.shape(content_current)[1]
    W = tf.shape(content_current)[2]
    C = tf.shape(content_current)[3]
    content_current = tf.reshape(content_current, (B*H*W,C))    
    content_target = tf.reshape(content_target, (B*H*W,C))

    size = tf.cast(B*H*W*C, tf.float32)
    loss = tf.nn.l2_loss(content_current - content_target) / size
    return loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (B, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    B = tf.shape(features)[0]
    H = tf.shape(features)[1]
    W = tf.shape(features)[2]
    C = tf.shape(features)[3]
    features = tf.reshape(features, (B,H*W,C))
    featuresT = tf.transpose(features, perm=[0,2,1])
    gram = tf.matmul(featuresT, features)
    if normalize:
        gram = tf.divide(gram, tf.cast(H*W*C, tf.float32))
    return gram

def style_loss(gen_feats, style_targets, style_layers, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - gen_feats: List of the same length as style_layers of generated image, where gen_feats[i] is
      a Tensor giving the Gram matrix  at layer style_layers[i].
    - style_targets: List of the same length as style_layers of target image, where style_targets[i] is
      a Tensor giving the Gram matrix  at layer style_layers[i].
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    losses = 0
    N = len(style_layers)
    for n in range(N):
        gram_generated = gen_feats[n]
        gram_target = style_targets[n]
        B = tf.shape(gram_target)[0]
        H = tf.shape(gram_target)[1]
        W = tf.shape(gram_target)[2]
        elem_num = tf.cast(B * H * W, tf.float32)
        loss = tf.divide((gram_generated - gram_target)**2, elem_num)
        losses += tf.reduce_sum(loss) * style_weights[n]
        
    return losses 
