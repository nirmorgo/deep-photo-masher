import numpy as np
import scipy.misc
from os.path import exists
import imageio
from skimage.transform import resize


def load_image(image_path, img_resize=480):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = imageio.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    img = preprocess_image(img, img_resize)
    img = img.astype("float32")
    return img

def preprocess_image(img, img_resize=480):
    '''
    A function that gets an image and resizing it to desired size.
    the function will crop the image to a square befor resize.
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h > img_w:
        start_point = np.random.randint(img_h-img_w) # add a little bit of randomallity to the crop
        img = img[start_point:start_point+img_w, :, :]
    elif img_w > img_h:
        start_point = np.random.randint(img_w-img_h)
        img = img[:, start_point:start_point+img_h, :]
    img = resize(img, (img_resize,img_resize), mode='constant')
    return img