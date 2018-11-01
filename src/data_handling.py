import os
from os import listdir
from os.path import isfile, join
from src.image_utils import load_image
from glob import glob
import pickle
import numpy as np
import random


class VAE_data():
    '''
    this data handler is designed to work with folders of unlabaled images
    '''
    def __init__(self, img_resize=64):
        self.Nimages=0
        self.images_list = []
        self.X = None
        self.loaded_imgs = False
        self.img_resize=img_resize
        self.current_train_idx = 1e24
        self.epoch_counter = 1
    
    def load_images_list_from_directory(self, folder_path):
        self.images_list = [folder_path+f for f in listdir(folder_path) if isfile(join(folder_path, f))]        
        self.Nimages = len(self.images_list)
    
    def load_images_to_memory(self, Nmax=1e26, random_sample=False):
        if random_sample:
            np.random.shuffle(self.images_list)
        N = min(Nmax, self.Nimages)
        self.X = np.zeros((N, self.img_resize, self.img_resize, 3), dtype=np.float32)
        for i in range(N):
            self.X[i] = load_image(self.images_list[i], self.img_resize)
            if (i+1) % 1000 == 0:
                print('loaded %d images' % (i+1) )
        self.loaded_imgs = True
        self.Nimages = N

    def restart_epoch(self):
        print('Starting epoch %d' % self.epoch_counter)
        self.rand_train_idxs = np.random.permutation(self.Nimages)
        self.current_train_idx = 0
        self.epoch_counter += 1
    
    def get_vae_feed_dict(self, X, batch_size=4, preprocess_func=None, **kwargs):
        '''
        loads random batch of images from list and return a tensorflow feed dictionary
        inputs:
            X - will be used a key for the dictionary. can use a tf.Placeholder or a string.
            batch_size - integer
        '''
        np.random.seed(None)
        if self.current_train_idx + batch_size > self.Nimages:
            self.restart_epoch()
        
        idxs = self.rand_train_idxs[self.current_train_idx:self.current_train_idx + batch_size]
        self.current_train_idx += batch_size
        
        if self.loaded_imgs:
            X_out = self.X[idxs]
            if preprocess_func is not None:
                X_out = preprocess_func(X_out)
            return {X: X_out}
        
        X_out = np.array(load_image(self.images_list[idxs[0]], self.img_resize))
        X_out = np.reshape(X_out, ((1,) + X_out.shape))
        if batch_size > 1:
            for i in idxs[1:]:
                Xtemp = np.array(load_image(self.images_list[i], self.img_resize))
                Xtemp = np.reshape(Xtemp, ((1,) + Xtemp.shape))
                X_out = np.append(X_out, Xtemp, axis=0)
        
        if preprocess_func is not None:
            X_out = preprocess_func(X_out)
        
        return {X: X_out}
            
    
    def get_image(self,idx):
        return load_image(self.images_list[idx], self.img_resize)
    

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
        self.epoch_counter = 1

    def restart_epoch(self):
        print('Starting epoch %d' % self.epoch_counter)
        self.rand_train_idxs = np.random.permutation(self.Nimages)
        self.current_train_idx = 0
        self.epoch_counter += 1
    
    def load_batch(self, path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin-1')
        return dict
    
    def load_data(self, folder_path, keep_classes=[0,1,2,3,4,5,6,7,8,9], **kwargs):
        file_names = glob(folder_path+'data*')
        for file_name in file_names:
            data_dict = self.load_batch(file_name)
            images = data_dict["data"].reshape((10000,3,32,32))
            images = images.astype(np.float32)
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
    
    def get_vae_feed_dict(self, X, batch_size=None, **kwargs):
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


class VGGFace_data():
    def __init__(self):
        self.Nimages=0
        self.images_list = []
        self.X = None
    
    def load_images_list_from_directory(self, folder_path):
        files = []
        pattern   = "*.jpg"        
        for dir,_,_ in os.walk(folder_path):
            files.extend(glob(os.path.join(dir,pattern)))
        self.images_list = files        
        self.Nimages = len(self.images_list)
        
    def get_image(self,idx,img_resize=128):
        return load_image(self.images_list[idx], img_resize)
    
    def get_vae_feed_dict(self, X, batch_size=1, img_resize=128, **kwargs):
        '''
        loads random batch of images from list and return a tensorflow feed dictionary
        inputs:
            X - will be used a key for the dictionary. can use a tf.Placeholder or a string.
            batch_size - integer
            img_resize - integer, number of pixels for the images in the feed dict (square images)
        '''
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
