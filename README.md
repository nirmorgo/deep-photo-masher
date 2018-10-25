# deep-photo-masher

The main goal of this project is to play around and a little bit with variational autoencoders (VAE) for educational purposes.

VAEs can be used for image encoding and for artificial images creation. the main difference between VAEs and GANs, is the fact that with VAEs we can easily extract a latent vector (z) represenatation of an existing image, and play around with it.

The application that I am trying to create here is a VAE photo masher that is doing the following steps:
* Train a VAE on a certain dataset
* Used trained net to extract latent vectors from 2 images
* Create a new arificial image from a linear combination of the 2 z vectors

hopefully, the images created from latent space arithmetics (decoded(w1 * z1 + w2 * z2)) will be more interesting that simple image combinations (w1 * img1 + w2 * img2) :)

This project is still a work in progress.

## CIFAR10 examples
![Alt text](data/readme_images/cifar10_img1.JPG "2 cats")
![Alt text](data/readme_images/cifar10_img2.JPG "a cat and a small dog")