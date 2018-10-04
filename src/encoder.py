import tensorflow as tf 
import numpy as np

from src.net import build_full_conv_autoencoder as autoencoder

class AE():
    def __init__(self, **kwargs):
        self.sess = get_session()
        if kwargs is not None:
            self.params = kwargs
        self.temp_folder = './tmp/'
        self.best_loss = 1e25   # Will be used to monitor the best loss and save it

        with tf.variable_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, [None,480,480,3], name='X')
            self.global_step = tf.Variable(0, name='global_step', trainable=False) 
        with tf.variable_scope('AutoEncoder'):
            autoencoder(self)
        with tf.variable_scope('Loss'):
            self.loss = self.loss_func()
            tf.summary.scalar('Total_Loss', self.loss)        
        with tf.variable_scope('Optimizer'):
            self.train_step = self.train_step_func()  
        self.scalars = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.temp_folder + '/tensorflow_logs/', graph=self.sess.graph)

        with self.sess.as_default():
            tf.global_variables_initializer().run()
    
    def loss_func(self):
        with tf.variable_scope('L2_loss'):
            B = tf.shape(self.X)[0]
            H = tf.shape(self.X)[1]
            W = tf.shape(self.X)[2]
            C = tf.shape(self.X)[3]
            elem_num = tf.cast(B*H*W*C, tf.float32)
            loss = tf.nn.l2_loss(self.X - self.net_out) / elem_num
        return loss
    
    def train_step_func(self):
        learning_rate = self.params.get('learning_rate', 0.001)
        self.LR = tf.Variable(learning_rate, name="LR")
        return tf.train.AdamOptimizer(self.LR).minimize(self.loss, global_step=self.global_step)

    def train(self, data, iters=2000, batch_size=4, learning_rate=None, train_set_resize=480):
        print('Initialized Train!')
        #iters_per_epoch = data.Nimages // batch_size
        if learning_rate is not None:
            self.sess.run(tf.assign(self.LR, learning_rate))
        for i in range(iters):
            feed_dict=data.get_random_encoder_feed_dict(X=self.X,  batch_size=batch_size, img_resize=train_set_resize)
            train_scalars, _, g_step, current_loss = self.sess.run([self.scalars, self.train_step, self.global_step, self.loss],
                                                        feed_dict=feed_dict)
            self.writer.add_summary(train_scalars, g_step)
            if g_step % 100 == 0 or i == 0:
                print('iteration',g_step)
                if current_loss < self.best_loss:
                    self.save_model_to_checkpoint()
                    self.best_loss = current_loss
        if current_loss < self.best_loss:
                    self.save_model_to_checkpoint()
                    self.best_loss = current_loss

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