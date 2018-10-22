from os import listdir
from os.path import isfile, join
from src.image_utils import load_image
import glob
import pickle
import numpy as np
import random


class AE_data():
    '''
    this data handler is designed to work with folders of unlabaled images
    '''
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

    def get_ae_feed_dict(self, X, batch_size=1, img_resize=480, preload=False):
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

class CIFAR10_data():
    def __init__(self):
        self.Nimages=0
        self.X = None
        self.labels = []
        self.encoding = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
                         'dog':5,'frog':6, 'horse':7, 'ship': 8, 'truck': 9}
        self.decoding = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
                         5:'dog',6:'frog', 7:'horse',  8:'ship', 9:'truck'}
        self.current_train_idx = 1e15

    def restart_epoch(self):
        self.rand_train_idxs = np.random.permutation(self.Nimages)
        self.current_train_idx = 0
    
    def load_batch(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin-1')
        return dict
    
    def load_data(self, folder_path, keep_classes=[0,1,2,3,4,5,6,7,8,9], **kwargs):
        file_names = glob.glob(folder_path+'data*')
        for file_name in file_names:
            data_dict = self.load_batch(file_name)
            images = data_dict["data"].reshape((10000,3,32,32))
            images = np.rollaxis(images, 1, 4)
            labels = data_dict["labels"]
            idxs_to_keep = []
            for idx, label in enumerate(labels):
                if label in keep_classes:
                    self.labels.append(label)
                    idxs_to_keep.append(idx)
            if self.X is None:
                self.X = images[idxs_to_keep] / 255.0
            else:
                self.X = np.concatenate((self.X, images[idxs_to_keep] / 255.0))
        
        self.Nimages = len(self.X)
    
    def get_ae_feed_dict(self, X, batch_size=None, **kwargs):
        '''
        gets a feed dictionary out of the data set.
        inputs: 
                X: tensorflow place holders with same dimensions as the images
                batch_size: return a randomly selected batch from dataset
        output: a feed dictionary for tensorflow session.
        '''
        np.random.seed(None)
        if self.current_train_idx + batch_size > self.Nimages:
            self.restart_epoch()
            
        idxs = self.rand_train_idxs[self.current_train_idx:self.current_train_idx + batch_size]
        X_batch = self.X[idxs]
        self.current_train_idx += batch_size
        return {X: X_batch}
