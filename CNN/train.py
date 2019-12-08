import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers,backend
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)
IMG_SHAPE = 28
OUTSIZE = 36
Samplesize = 1100
TestS = 55

class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            32, (3,3), strides=(1,1),activation='relu',  input_shape=(IMG_SHAPE, IMG_SHAPE, 1), name='conv1'))
        model.add(layers.MaxPooling2D((2, 2), strides=2, name='pooling1'))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), strides=(1,1), activation='relu',name='conv2'))
        model.add(layers.MaxPooling2D((2, 2), strides=2,name='pooling2'))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu',name='conv3'))
        # model.add(tf.keras.layers.Dropout(0.5))

        model.add(layers.Flatten(name='f1'))
        model.add(layers.Dense(64, activation='relu',name='d1'))
        model.add(layers.Dense(OUTSIZE, activation='softmax',name='d2'))

        model.summary()

        self.model = model

class DataSource(object):
    def __init__(self):
        data_path = os.path.abspath(os.path.dirname(
            __file__)) + '/../exchar74kds.npz'

        data = np.load(data_path)
        train_images = data['traindata']
        test_images = data['testdata']
        train_labels = data['trainlabel']
        test_labels = data['testlabel']

        train_images  = train_images.reshape((36*Samplesize,28,28,1))
        test_images = test_images.reshape((36*TestS,28,28,1))

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)


        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images , self.test_labels = test_images, test_labels


class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])


        self.cnn.model.fit(self.data.train_images, self.data.train_labels,batch_size=12,
                           epochs=5, callbacks=[save_model_cb])


        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))
        self.cnn.model.save('./model/exchar74k_cnn.h5')


if __name__ == "__main__":
    app = Train()
    app.train()
