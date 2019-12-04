import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2 as cv

IMG_SHAPE = 28
OUTSIZE = 2


def ABds():
    data_path = os.path.abspath(os.path.dirname(
        __file__)) + '/../ABds.npz'

    data = np.load(data_path)
    train_images = data['traindata']
    test_images = data['testdata']
    train_labels = data['trainlabel']
    test_labels = data['testlabel']

    train_images, test_images = train_images/255.0, test_images/255.0

    print(train_labels.shape)

def Mnist():
    data_path = os.path.abspath(os.path.dirname(
        __file__)) + '/../mnist.npz'

    data = np.load(data_path)
    train_images = data['x_train']
    test_images = data['x_test']
    train_labels = data['y_train']
    test_labels = data['y_test']

    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(train_labels[0])
    print(train_labels.shape)


if __name__ == "__main__":
    ABds()
    print('------------------')
    Mnist()

