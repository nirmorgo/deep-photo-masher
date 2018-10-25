import tensorflow as tf 

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
        self.temp_folder = './tmp/'
        self.best_loss = 1e25   # Will be used to monitor the best loss and save it

        with tf.variable_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, [None,self.img_size,self.img_size,3], name='X')
            self.global_step = tf.Variable(0, name='global_step', trainable=False) 
        with tf.variable_scope('AutoEncoder'):
            net_func(self)
        with tf.variable_scope('Loss'):
            self.loss = self.loss_func()
            tf.summary.scalar('Total_Loss', self.loss)        
        with tf.variable_scope('Optimizer'):
            self.train_step = self.train_step_func()  
        self.scalars = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.temp_folder + '/tensorflow_logs/', graph=self.sess.graph)

        with self.sess.as_default():
            tf.global_variables_initializer().run()
        
#        inputs = tf.keras.
#        self.Model = tf.keras.Model(inputs=inputs, outputs=self.net_out)
    
    def loss_func(self):
        img = self.net_out
        with tf.variable_scope('inputs_shape'):
            N = tf.shape(self.X)[0]
            H = tf.shape(self.X)[1]
            W = tf.shape(self.X)[2]
            C = tf.shape(self.X)[3]
            elem_num = tf.cast(N*H*W*C, tf.float32)
        
        with tf.variable_scope('L1_loss'):
            l1_loss = tf.losses.absolute_difference(self.X, img) / elem_num
            tf.summary.scalar('L1_loss', l1_loss) 

        loss = self.c_l1 * l1_loss
        
        with tf.variable_scope('L2_loss'):
            l2_loss = tf.nn.l2_loss(self.X - img) / elem_num
            tf.summary.scalar('L2_loss', l2_loss)
        
        loss += self.c_l2 * l2_loss
        
        with tf.variable_scope('TV_loss'):
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
            tf.summary.scalar('TV_loss', tv_loss)
        
        loss += self.c_tv * tv_loss

        with tf.variable_scope('KL_div_loss'):
            KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq),1))
            KL_loss = KL_loss / elem_num
            tf.summary.scalar('KL_div_loss', KL_loss)
        
        loss += self.c_kl * KL_loss

        return loss
    
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
            feed_dict=data.get_vae_feed_dict(X=self.X,  batch_size=batch_size, img_resize=self.img_size, preload=False)
            train_scalars, _, g_step, current_loss = self.sess.run([self.scalars, self.train_step, self.global_step, self.loss],
                                                        feed_dict=feed_dict)
            self.writer.add_summary(train_scalars, g_step)
            if g_step % 25 == 0 or i == 0:
                print('iteration %d, loss: %.3f' % (g_step, current_loss))
                if current_loss < self.best_loss:
                    self.save_model_to_checkpoint(self.temp_folder+'/model_files/best_model/model')
                    self.best_loss = current_loss
            
            if g_step % 100 == 0:
                self.save_model_to_checkpoint()
        self.save_model_to_checkpoint()

    def get_z(self, img):
        '''
        get a np.array with a single image (H,W,C) or a batch of images (N,H,W,C)
        returns the corresponding z represenatation from the autoencoder
        user should make sure that the input images have dimensions that match the autoencoder
        '''
        if len(img.shape) == 3:
            img = img.reshape((1,)+img.shape)
        z = self.sess.run(self.z, feed_dict={self.X:img})
        return z

    def get_image(self, img):
        '''
        get a np.array with a single image (H,W,C) or a batch of images (N,H,W,C)
        returns the corresponding output images from the autoencoder
        user should make sure that the input images have dimensions that match the autoencoder
        '''
        if len(img.shape) == 3:
            img = img.reshape((1,)+img.shape)
        z = self.sess.run(self.net_out, feed_dict={self.X:img})
        return z

    def get_img_from_z(self, z):
        '''
        get a np.array with a latent autoencoder representation
        returns the corresponding output images from the autoencoder
        user should make sure that the inputs have dimensions that match the autoencoder
        '''
        out = self.sess.run(self.net_out, feed_dict={self.z:z})
        return out
    
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


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session