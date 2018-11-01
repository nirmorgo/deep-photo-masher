import tensorflow as tf 
from src.loss_funcs import l1_loss, l2_loss, kl_div_loss, tv_loss
from src.loss_funcs import gram_matrix, content_loss, style_loss
from src.vgg19net import VGG19

class VAE():
    def __init__(self, net_func, **kwargs):
        '''
        creates an autoencoder instance
        input params:
        img_size (optional) - integer. determines the size of the images that will be used in the autoencoder. default - 128.
        learning_rate (optional) - determines the initial learning rate that will be set for the training process. default - 0.001
        c_l1 (optional) - weight of L1 loss, default - 1
        c_l2 (optional) - weight of L2 loss, default - 1
        c_tv (optional) - weight of TV loss, default - 1
        c_kl (optional) - weight of KL-divergence loss, default - 1e-6
        '''
        self.sess = get_session()
        self.params = {}
        if kwargs is not None:
            self.params = kwargs
        self.img_size = self.params.get('img_size', 32)
        self.c_l1 = self.params.get('c_l1', 1)
        self.c_l2 = self.params.get('c_l2', 1)
        self.c_tv = self.params.get('c_tv', 1)
        self.c_kl = self.params.get('c_kl', 1e-6)
        self.c_content = self.params.get('c_content', 1e-7)
        self.c_style = self.params.get('c_style', 1e-7)
        self.temp_folder = './tmp/'
        self.best_loss = 1e25   # Will be used to monitor the best loss and save it

        with tf.variable_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, [None,self.img_size,self.img_size,3], name='X')
            self.global_step = tf.Variable(0, name='global_step', trainable=False) 
        with tf.variable_scope('AutoEncoder'):
            net_func(self)
        with tf.variable_scope('Loss'):
            self.loss = self.loss_func()
        with tf.variable_scope('Semantic_Loss'):
            self.loss += self.semantic_loss_func()
        with tf.variable_scope('Optimizer'):
            self.train_step = self.train_step_func()  
        tf.summary.scalar('Total_Loss', self.loss)
        self.scalars = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.temp_folder + '/tensorflow_logs/', graph=self.sess.graph)

        with self.sess.as_default():
            tf.global_variables_initializer().run()
        
#        inputs = tf.keras.
#        self.Model = tf.keras.Model(inputs=inputs, outputs=self.net_out)
    
    def loss_func(self):
        with tf.variable_scope('L1_loss'):
            l1_loss_ = l1_loss(self.X, self.net_out)
        tf.summary.scalar('L1_loss', l1_loss_) 
        loss = self.c_l1 * l1_loss_
        
        with tf.variable_scope('L2_loss'):
            l2_loss_ = l2_loss(self.X, self.net_out)
        tf.summary.scalar('L2_loss', l2_loss_)
        loss += self.c_l2 * l2_loss_
        
        with tf.variable_scope('TV_loss'):
            tv_loss_ = tv_loss(self.net_out)
        tf.summary.scalar('TV_loss', tv_loss_)
        loss += self.c_tv * tv_loss_

        with tf.variable_scope('KL_div_loss'):
            kl_loss_ = kl_div_loss(self.X, self.z_mu, self.z_log_sigma_sq)
        tf.summary.scalar('KL_div_loss', kl_loss_)
        loss += self.c_kl * kl_loss_

        return loss
    
    def semantic_loss_func(self):
        content_layer = 'conv4_2'
        style_layers_list = [
                    ('conv1_1', 0.2),
                    ('conv2_1', 0.2),
                    ('conv3_1', 0.2),
                    ('conv4_1', 0.2),
                    ('conv5_1', 0.2)]
        style_layers = [layer[0] for layer in style_layers_list]
        style_weights = [layer[1] for layer in style_layers_list]
        vgg19 = VGG19()
        with tf.variable_scope('Input_Features'):
            with tf.variable_scope('VGG19_layers'):
                X_vgg = vgg19.preprocess(self.X)
                X_features = vgg19.net(X_vgg)
            # Extract content features from inputs  
            X_content = X_features[content_layer]
            X_feat_vars = [X_features[layer] for layer in style_layers]                
            # Compute list of TensorFlow Gram matrices
            X_feats = []
            for X_feat_var in X_feat_vars:
                X_feats.append(gram_matrix(X_feat_var))
        
        with tf.variable_scope('Generated_Features'):             
            with tf.variable_scope('VGG19_layers'):
                net_out_vgg = vgg19.preprocess(self.net_out)
                gen_features = vgg19.net(net_out_vgg)
            gen_content = gen_features[content_layer]
            # Extract style features from target  
            gen_style_feat_vars = [gen_features[layer] for layer in style_layers]                
            # Compute list of TensorFlow Gram matrices
            gen_style_feats = []
            for style_feat_var in gen_style_feat_vars:
                gen_style_feats.append(gram_matrix(style_feat_var))
            
        with tf.variable_scope('Content_loss'):
            c_loss = content_loss(gen_content, X_content)
        tf.summary.scalar('Content_Loss', c_loss)
        
        with tf.variable_scope('Style_loss'):
            s_loss = style_loss(gen_style_feats, X_feats, style_layers, style_weights)
        tf.summary.scalar('Style_Loss', s_loss)
        
        return self.c_content * c_loss + self.c_style * s_loss
    
    def train_step_func(self):
        learning_rate = self.params.get('learning_rate', 0.001)
        self.LR = tf.Variable(learning_rate, name="LR")
        return tf.train.AdamOptimizer(self.LR).minimize(self.loss, global_step=self.global_step)

    def train(self, data, iters=2000, batch_size=4, learning_rate=None):
        print('Initialized Train!')
        #iters_per_epoch = data.Nimages // batch_size
        if learning_rate is not None:
            self.sess.run(tf.assign(self.LR, learning_rate))
        for i in range(iters):
            feed_dict=data.get_vae_feed_dict(X=self.X, batch_size=batch_size, 
                                             preprocess_func = self.preprocess)
            train_scalars, _, g_step, current_loss = self.sess.run([self.scalars, self.train_step, self.global_step, self.loss],
                                                        feed_dict=feed_dict)
            self.writer.add_summary(train_scalars, g_step)
            if g_step % 25 == 0 or i == 0:
                print('iteration %d, loss: %.3e' % (g_step, current_loss))
                if current_loss < self.best_loss:
                    self.save_weights_to_checkpoint(self.temp_folder+'/model_files/best_model/model')
                    self.best_loss = current_loss
            
            if g_step % 100 == 0:
                self.save_model_to_checkpoint()
        self.save_model_to_checkpoint()
    
    @staticmethod    
    def preprocess(img):
        # centers the image/s with [-1,1] range
        # such pre-process is needed when using tanh activation on output layer
        return img * 2 - 1
    
    @staticmethod    
    def deprocess(img):
        # centers the image/s back in [0,1] range
        return img / 2 + 0.5
    
    
    def get_z(self, img):
        '''
        get a np.array with a single image (H,W,C) or a batch of images (N,H,W,C)
        returns the corresponding z represenatation from the autoencoder
        user should make sure that the input images have dimensions that match the autoencoder
        '''
        if len(img.shape) == 3:
            img = img.reshape((1,)+img.shape)
        z = self.sess.run(self.z, feed_dict={self.X:self.preprocess(img)})
        return z

    def get_image(self, img):
        '''
        get a np.array with a single image (H,W,C) or a batch of images (N,H,W,C)
        returns the corresponding output images from the autoencoder
        user should make sure that the input images have dimensions that match the autoencoder
        '''
        if len(img.shape) == 3:
            img = img.reshape((1,)+img.shape)
        out_img = self.sess.run(self.net_out, feed_dict={self.X:self.preprocess(img)})
        return self.deprocess(out_img)

    def get_img_from_z(self, z):
        '''
        get a np.array with a latent autoencoder representation
        returns the corresponding output images from the autoencoder
        user should make sure that the inputs have dimensions that match the autoencoder
        '''
        out_img = self.sess.run(self.net_out, feed_dict={self.z:z})
        return self.deprocess(out_img)
    
    def save_model_to_checkpoint(self, path=None):
        saver = tf.train.Saver()
        if path is None:
            path = self.temp_folder+'/model_files/model'
        saver.save(self.sess, path)
    
    def restore_model_from_last_checkpoint(self, path=None):
        if path is None:
            path = self.temp_folder+'/model_files/model'
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        
    def load_weights_from_checkpoint(self, path=None):
        if path is None:
            path = self.path
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(self.sess, path) 
        
    def save_weights_to_checkpoint(self, path=None):
        if path is None:
            path = self.path
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(self.sess, path)


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session