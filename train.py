from src.data_handling import CIFAR10_data
from src.encoder import AE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning-rate', default=1e-4)
parser.add_argument('-it', '--iterations', default=500)
parser.add_argument('-b', '--batch-size', default=32)
parser.add_argument('-c_l1', '--c_l1', default=1e6)
parser.add_argument('-c_l1', '--c_l2', default=1e-2)
parser.add_argument('-c_tv', '--c_tv', default=1e-1)
parser.add_argument('-c_kl', '--c_kl', default=1e0)
parser.add_argument('-r', '--restore_last_weights', default=False)
args = parser.parse_args()

lr = float(args.learning_rate)
batch_size = int(args.batch_size)
iters = int(args.iterations)
c_l1 = float(args.c_l1)
c_l2 = float(args.c_l2)
c_tv = float(args.c_tv)
c_kl = float(args.c_kl)
restore = bool(args.restore_last_weights)

def train(*args):
    data = CIFAR10_data()
    data.load_data('data/cifar10/')

    autoencoder = AE(c_l1=c_l1, c_l2=c_l2, c_tv=c_tv, c_kl=c_kl)
    if restore:
        autoencoder.restore_model_from_last_checkpoint()
    autoencoder.train(data, batch_size=batch_size, iters=iters, learning_rate=lr)

if __name__ == "__main__":
    train()
