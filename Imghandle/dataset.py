import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
import random

np.set_printoptions(threshold=np.inf)
IMG_SHAPE = 28
Samplesize = 900#1100
TestS = 50#55
samplelist = ["Sample001","Sample002","Sample003","Sample004","Sample005","Sample006","Sample007","Sample008",
              "Sample009","Sample010","Sample011","Sample012","Sample013","Sample014","Sample015","Sample016",
              "Sample017","Sample018","Sample019","Sample020","Sample021","Sample022","Sample023","Sample024",
              "Sample025","Sample026","Sample027","Sample028","Sample029","Sample030","Sample031","Sample032",
              "Sample033","Sample034","Sample035","Sample000"]

Samplesize1 = 1000
TestS1 = 45
samplelist1 = ["Sample001","Sample002","Sample003","Sample004","Sample005","Sample006","Sample007","Sample008",
              "Sample009","Sample010","Sample011","Sample012","Sample013","Sample014","Sample015","Sample016",
              "Sample017","Sample018","Sample019","Sample020","Sample021","Sample022","Sample023","Sample024",
              "Sample025","Sample026","Sample027","Sample028","Sample029","Sample030","Sample031","Sample032",
              "Sample033","Sample034","Sample035","Sample036"]


def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 2.5)
    cv.bitwise_not(binary,binary)
    return binary


def gettrainimg():
    filepath = "C:/Users/JarvisZhang/Desktop/test/"
    path = os.listdir(filepath)
    cnt = 0
    imgarray = np.zeros((len(samplelist)*Samplesize,IMG_SHAPE,IMG_SHAPE))
    for eachpath in path:
        #choose
        if eachpath not in samplelist:
            continue
        child =  os.path.join("%s%s" %(filepath,eachpath))
        for filename in os.listdir(child):
            if filename in testimgname:
                continue
            img = cv.imread(os.path.join("%s%s/%s" % (filepath, eachpath, filename)))
            img = local_threshold(img)
            imgarray[cnt] = np.array(img)
            cnt = cnt + 1
            print(filename)
    return imgarray

def gettrainlabel():
    labelarray = np.zeros((len(samplelist)*Samplesize))
    cnt = 0
    for i in range(36):
       for j in range(Samplesize):
           labelarray[cnt] = i
           cnt = cnt + 1
    return labelarray


def gettestimg():
    filepath = "C:/Users/JarvisZhang/Desktop/test/"
    path = os.listdir(filepath)
    cnt = 0
    imgarray = np.zeros((TestS*len(samplelist), IMG_SHAPE, IMG_SHAPE))
    for eachpath in path:
        # choose
        if eachpath  not in samplelist:
            continue
        child = os.path.join("%s%s" % (filepath, eachpath))
        childpath = os.listdir(child)
        for index in range(TestS):
            while True:
                num = random.randint(0,Samplesize-1)
                if childpath[num] not in testimgname:
                    testimgname.append(childpath[num])
                    break
            img = cv.imread(os.path.join("%s%s/%s" % (filepath, eachpath, childpath[num])))
            img = local_threshold(img)
            imgarray[cnt] = np.array(img)
            cnt = cnt + 1
            print(childpath[num])
    return imgarray

def gettestlabel():
    labelarray = np.zeros((len(samplelist) * TestS))
    cnt = 0
    for i in range(36):
        for j in range(TestS):
            labelarray[cnt] = i
            cnt = cnt + 1
    return labelarray


def gettrainimg1():
    filepath = "C:/Users/JarvisZhang/Desktop/Img/"
    path = os.listdir(filepath)
    cnt = 0
    imgarray = np.zeros((len(samplelist1)*Samplesize1,IMG_SHAPE,IMG_SHAPE))
    for eachpath in path:
        #choose
        if eachpath not in samplelist1:
            continue
        child =  os.path.join("%s%s" %(filepath,eachpath))
        for filename in os.listdir(child):
            if filename in testimgname1:
                continue
            img = cv.imread(os.path.join("%s%s/%s" % (filepath, eachpath, filename)))
            img = local_threshold(img)
            imgarray[cnt] = np.array(img)
            cnt = cnt + 1
            print(filename)
    return imgarray

def gettrainlabel1():
    labelarray = np.zeros((len(samplelist1)*Samplesize1))
    cnt = 0
    for i in range(36):
       for j in range(Samplesize1):
           labelarray[cnt] = i
           cnt = cnt + 1
    return labelarray


def gettestimg1():
    filepath = "C:/Users/JarvisZhang/Desktop/Img/"
    path = os.listdir(filepath)
    cnt = 0
    imgarray = np.zeros((TestS1*len(samplelist1), IMG_SHAPE, IMG_SHAPE))
    for eachpath in path:
        # choose
        if eachpath  not in samplelist1:
            continue
        child = os.path.join("%s%s" % (filepath, eachpath))
        childpath = os.listdir(child)
        for index in range(TestS1):
            while True:
                num = random.randint(0,Samplesize1-1)
                if childpath[num] not in testimgname1:
                    testimgname1.append(childpath[num])
                    break
            img = cv.imread(os.path.join("%s%s/%s" % (filepath, eachpath, childpath[num])))
            img = local_threshold(img)
            imgarray[cnt] = np.array(img)
            cnt = cnt + 1
            print(childpath[num])
    return imgarray

def gettestlabel1():
    labelarray = np.zeros((len(samplelist1) * TestS1))
    cnt = 0
    for i in range(36):
        for j in range(TestS1):
            labelarray[cnt] = i
            cnt = cnt + 1
    return labelarray


if __name__ == '__main__':
    testimgname = []
    testimglist = gettestimg()
    testlabellist = gettestlabel()
    imglist = gettrainimg()
    labellist = gettrainlabel()

    testimgname1 = []
    testimglist1 = gettestimg1()
    testlabellist1 = gettestlabel1()
    imglist1 = gettrainimg1()
    labellist1 = gettrainlabel1()

    np.savez("../combined.npz",traindata = np.append(imglist,imglist1)
                                     ,trainlabel  = np.append(labellist,labellist1)
                                     ,testdata = np.append(testimglist,testimglist1)
                                     ,testlabel = np.append(testlabellist,testlabellist1))