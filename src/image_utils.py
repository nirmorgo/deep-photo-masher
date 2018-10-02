import numpy as np
import scipy.misc
from os.path import exists
import imageio
from skimage.transform import rotate, resize


def load_image(image_path, img_resize=None):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = imageio.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_resize is not None):
        img = preprocess_image(img, img_resize[0], img_resize[1])

    img = img.astype("float32")
    return img

def preprocess_image(img, height, width):
    '''
    A function that gets an image and resizing it to desired size.
    the function will rotate an image when needed, to minimize image distortion.
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    if img_w > img_h:
        img = rotate(img,90, resize=True)
    img = resize(img, (height,width), mode='constant', anti_aliasing=True)
    return img