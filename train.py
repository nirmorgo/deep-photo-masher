from src.data_handling import AE_data
from src.encoder import AE

def train(*args):
    data = AE_data()
    data.load_images_list_from_directory('data/COCO_train2017/')

    autoencoder = AE(c_l1=1e8, c_tv=1e1, c_l2=0)
    autoencoder.restore_model_from_last_checkpoint()
    autoencoder.train(data, batch_size=8, iters=700, learning_rate=0.005)


if __name__ == "__main__":
    train()
