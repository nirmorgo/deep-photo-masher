from src.image_utils import load_image
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from src.encoder import VAE
from src.net import build_vae_128 as net

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-size', '--image_size', default=128)
parser.add_argument('-im1', '--image1_path', default='./data/sample_imgs/sample_img_1.jpg')
parser.add_argument('-im2', '--image2_path', default='./data/sample_imgs/sample_img_2.jpg')
parser.add_argument('-model', '--model_path', default='./saved_models/celebA_128/model')
args = parser.parse_args()

img_size = int(args.image_size)
img1_path = args.image1_path
img2_path = args.image2_path
model_path = args.model_path

def slerp(z1, z2, t):
    '''
    z1, z2 - two vectors in the latent space
    t - weighted ratio of the output interpolated vector
    '''
    omega = np.arccos(np.dot(z1/norm(z1), z2.T/norm(z2)))
    sin_omega = np.sin(omega)
    z_out = np.sin((1.0-t)*omega) / sin_omega * z1 + np.sin(t*omega) / sin_omega * z2
    return z_out

def main(*args):
    img1 = load_image(img1_path, img_resize=img_size)
    img2 = load_image(img2_path, img_resize=img_size)
    
    # load trained autoencoder
    autoencoder = VAE(net, img_size=img_size, semantic_loss=False)
    autoencoder.load_weights_from_checkpoint(model_path)
    
    # plot the z space walk between the images (the "mashing")
    shape = img1.shape
    z1 = autoencoder.get_z(img1)
    z2 = autoencoder.get_z(img2)
    f = plt.figure(figsize=(14,3))
    plt.subplot(1,9,1)
    plt.imshow(img1.reshape(shape))
    plt.axis('off')
    for i, t in enumerate([0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]):    
        z3 = slerp(z1, z2, t) 
        mashed = autoencoder.get_img_from_z(z3)
        plt.subplot(1,9,i+2)
        plt.imshow(np.clip(mashed.reshape(shape), 0, 1))
        plt.axis('off')
    plt.subplot(1,9,9)
    plt.imshow(img2.reshape(shape))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()