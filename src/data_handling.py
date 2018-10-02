from os import listdir
from os.path import isfile, join
from src.image_utils import load_image
import numpy as np
import random


class AE_data():
    def __init__(self):
        self.Nimages=0
        self.images_list = []
    
    def load_images_list_from_directory(self, folder_path):
        self.images_list = [folder_path+f for f in listdir(folder_path) if isfile(join(folder_path, f))]        
        self.Nimages = len(self.images_list)

    def get_random_encoder_feed_dict(self, X, y, batch_size=1, img_resize=None):
        idx = random.sample(range(0, self.Nimages), batch_size)
        Xout = np.array(load_image(self.images_list[idx[0]], img_resize))
        Xout = np.reshape(Xout, ((1,) + Xout.shape))
        if batch_size == 1:
            return {X: Xout, y: Xout}
        else:
            for i in idx[1:]:
               Xtemp = np.array(load_image(self.images_list[i], img_resize))
               Xtemp = np.reshape(Xtemp, ((1,) + Xtemp.shape))
               Xout = np.append(Xout, Xtemp, axis=0)
            out_dict = {X: Xout, y: Xout}
        return out_dict

data = AE_data()
data.load_images_list_from_directory("data/COCO_train2017/")

a = data.get_random_encoder_feed_dict('X','y',8,(320,280))
pass