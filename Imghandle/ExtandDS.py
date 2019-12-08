import os
import cv2 as cv
import numpy as np
import threading
np.set_printoptions(threshold=np.inf)

IMG_SHAPE = 28
samplelist = ["Sample001","Sample002","Sample003","Sample004","Sample005","Sample006","Sample007","Sample008",
              "Sample009","Sample010","Sample011","Sample012","Sample013","Sample014","Sample015","Sample016",
              "Sample017","Sample018","Sample019","Sample020","Sample021","Sample022","Sample023","Sample024",
              "Sample025","Sample026","Sample027","Sample028","Sample029","Sample030","Sample031","Sample032",
              "Sample033","Sample034","Sample035","Sample036"]

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h), borderValue=(255,255,255,255))
    return rotated

def Rotateandsave(p):
    filepath = "C:/Users/JarvisZhang/Desktop/Img/" + p
    path = os.listdir(filepath)
    for eachimg in path:
        img = cv.imread(os.path.join("%s/%s" % (filepath, eachimg)))

        img10 = rotate(img, 10)
        cv.imwrite(os.path.join("%s/(10)-%s" % (filepath, eachimg)), img10)
        imgn10 = rotate(img, -10)
        cv.imwrite(os.path.join("%s/(-10)-%s" % (filepath, eachimg)), imgn10)

        img15 = rotate(img,15)
        cv.imwrite(os.path.join("%s/(15)-%s" % (filepath, eachimg)), img15)
        imgn15 = rotate(img,-15)
        cv.imwrite(os.path.join("%s/(-15)-%s" % (filepath, eachimg)), imgn15)

        img20 = rotate(img,20)
        cv.imwrite(os.path.join("%s/(20)-%s" % (filepath, eachimg)), img20)
        imgn20 = rotate(img,-20)
        cv.imwrite(os.path.join("%s/(-20)-%s" % (filepath, eachimg)), imgn20)

def RC(img,type):
    if type == 'H':
        img = cv.resize(img,(28,40),interpolation=cv.INTER_AREA)
        img = img[6:34, 0:28, 0:3]
    elif type == 'W':
        img = cv.resize(img, (40, 28), interpolation=cv.INTER_AREA)
        img = img[0:28, 6:34, 0:3]
    return img

def RCandsave(p):
    filepath = "C:/Users/JarvisZhang/Desktop/Img/" + p
    path = os.listdir(filepath)
    for eachimg in path:
        img = cv.imread(os.path.join("%s/%s" % (filepath, eachimg)))
        img1 = RC(img,'H')
        cv.imwrite(os.path.join("%s/(H)-%s" % (filepath, eachimg)), img1)
        img2 = RC(img,'W')
        cv.imwrite(os.path.join("%s/(W)-%s" % (filepath, eachimg)), img2)


if __name__ == '__main__':
    for path in samplelist:
        Rotateandsave(path)
    for path in samplelist:
        RCandsave(path)


    # img = cv.imread("C:/Users/JarvisZhang/Desktop/Img/Sample008/img008-008.png")
    # img = cv.resize(img,(28,40),interpolation=cv.INTER_AREA)
    # cv.imshow('resize',img)
    # img = img[6:34,0:28,0:3]
    # cv.imshow('after',img)
    # cv.waitKey()
    #
    # img = cv.imread("C:/Users/JarvisZhang/Desktop/Img/Sample008/img008-008.png")
    # img = cv.resize(img, (40, 28), interpolation=cv.INTER_AREA)
    # cv.imshow('resize', img)
    # img = img[0:28, 6:34, 0:3]
    # cv.imshow('after', img)
    # cv.waitKey()
