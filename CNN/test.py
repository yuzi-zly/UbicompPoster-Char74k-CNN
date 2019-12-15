import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2 as cv

IMG_SHAPE = 28
OUTSIZE = 2


def ABds():
    data_path = os.path.abspath(os.path.dirname(
        __file__)) + '/../combined.npz'

    data = np.load(data_path)
    train_images = data['traindata']
    test_images = data['testdata']
    train_labels = data['trainlabel']
    test_labels = data['testlabel']

    train_images, test_images = train_images/255.0, test_images/255.0

    print(train_images.shape)
    print(test_images.shape)

if __name__ == "__main__":
    ABds()


