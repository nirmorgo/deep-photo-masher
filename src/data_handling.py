from os import listdir
from os.path import isfile, join
from src.image_utils import load_image
import numpy as np
import random


class AE_data():
    def __init__(self):
        self.Nimages=0
        self.images_list = []
        self.X = None
        self.idx_counter = 1e15
        self.N_preload = 20000
    
    def load_images_list_from_directory(self, folder_path):
        self.images_list = [folder_path+f for f in listdir(folder_path) if isfile(join(folder_path, f))]        
        self.Nimages = len(self.images_list)

    def preload_images(self, img_resize=480):
        '''
        preloading data to ram, to reduce IO time loss on train
        '''
        print('loading random %d images to memory' % self.N_preload)
        self.idx_counter = 0
        self.X = []
        idxs = random.sample(range(0, self.Nimages), self.N_preload)
        for idx in idxs:
            X_temp = np.array(load_image(self.images_list[idx], img_resize))
            # X_temp = np.reshape(self.X, ((1,) + self.X.shape))
            self.X.append(X_temp)
        self.X = np.array(self.X)

    def get_random_encoder_feed_dict(self, X, batch_size=1, img_resize=480, preload=False):
        '''
        loads random batch of images from list and return a tensorflow feed dictionary
        inputs:
            X - will be used a key for the dictionary. can use a tf.Placeholder or a string.
            batch_size - integer
            img_resize - integer, number of pixels for the images in the feed dict (square images)
        '''
        if not preload:
            idx = random.sample(range(0, self.Nimages), batch_size)
            Xout = np.array(load_image(self.images_list[idx[0]], img_resize))
            Xout = np.reshape(Xout, ((1,) + Xout.shape))
            if batch_size == 1:
                return {X: Xout}
            else:
                for i in idx[1:]:
                    Xtemp = np.array(load_image(self.images_list[i], img_resize))
                    Xtemp = np.reshape(Xtemp, ((1,) + Xtemp.shape))
                    Xout = np.append(Xout, Xtemp, axis=0)
                out_dict = {X: Xout}
            return out_dict
        
        else:
            if self.idx_counter >= self.N_preload - batch_size:
                self.preload_images(img_resize)
            Xout = self.X[self.idx_counter:self.idx_counter+batch_size]
            self.idx_counter += batch_size

            return {X: Xout}