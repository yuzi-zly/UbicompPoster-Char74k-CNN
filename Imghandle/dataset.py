import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np

np.set_printoptions(threshold=np.inf)
IMG_SHAPE = 28
samplelist = ["Sample001","Sample002","Sample003","Sample004","Sample005","Sample006","Sample007","Sample008",
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
    filepath = "C:/Users/JarvisZhang/Desktop/Img/"
    path = os.listdir(filepath)
    cnt = 0
    imgarray = np.zeros((len(samplelist)*50,IMG_SHAPE,IMG_SHAPE))
    for eachpath in path:
        #choose
        if eachpath not in samplelist:
            continue
        child =  os.path.join("%s%s" %(filepath,eachpath))
        for filename in os.listdir(child)[0:50]:
            img = cv.imread(os.path.join("%s%s/%s" % (filepath, eachpath, filename)))
            img = local_threshold(img)
            imgarray[cnt] = np.array(img)
            cnt = cnt + 1
            print(filename)
    return imgarray

def gettrainlabel():
    labelarray = np.zeros((len(samplelist)*50))
    cnt = 0
    for i in range(36):
       for j in range(50):
           labelarray[cnt] = i
           cnt = cnt + 1
    return labelarray


def gettestimg():
    filepath = "C:/Users/JarvisZhang/Desktop/Img/"
    path = os.listdir(filepath)
    cnt = 0
    imgarray = np.zeros((5*len(samplelist), IMG_SHAPE, IMG_SHAPE))
    for eachpath in path:
        # choose
        if eachpath  not in samplelist:
            continue
        child = os.path.join("%s%s" % (filepath, eachpath))
        for filename in os.listdir(child)[50:55]:
            img = cv.imread(os.path.join("%s%s/%s" % (filepath, eachpath, filename)))
            img = local_threshold(img)
            imgarray[cnt] = np.array(img)
            cnt = cnt + 1
            print(filename)
    return imgarray

def gettestlabel():
    labelarray = np.zeros((len(samplelist) * 5))
    cnt = 0
    for i in range(36):
        for j in range(5):
            labelarray[cnt] = i
            cnt = cnt + 1
    return labelarray

if __name__ == '__main__':
    imglist = gettrainimg()
    labellist = gettrainlabel()
    labellist.astype(int)
    testimglist = gettestimg()
    testlabellist = gettestlabel()
    testlabellist.astype(int)
    np.savez("../ABds.npz",traindata = imglist, trainlabel  = labellist,testdata = testimglist,testlabel = testlabellist)
