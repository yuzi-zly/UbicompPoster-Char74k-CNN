import random
import os
from PIL import Image, ImageDraw, ImageFont
import shutil

random.seed(3)
path_img = "C:/Users/JarvisZhang/Desktop/test/"

def mkdir_for_imgs():
    for i in range(36):
        if os.path.isdir(path_img + "Sample" + "%03d" % i):
            shutil.rmtree(path_img + "Sample" + "%03d" % i, ignore_errors=True)
            print(path_img + "Sample" + "%03d" % i)
            os.mkdir(path_img + "Sample" + "%03d" % i)
        else:
            print(path_img + "Sample" + "%03d" % i)
            os.mkdir(path_img + "Sample" + "%03d" % i)

switch = {
    0:  15,
    1:  16,
    2:  17,
    3:  18,
    4:  19,
    5:  20,
    6:  21,
    7:  22,
    8:  23,
    9:  24
}

# 生成单张图像
def generate_single(m,index):
    # 先绘制一个28*28的空图像
    im_50_blank = Image.new('RGBA', (28, 28), (255, 255, 255,255))
    # 创建画笔
    draw = ImageDraw.Draw(im_50_blank)

    if index >= 10:
        num = chr(index-10+65)
    else:
        num = chr(index+48)

    if int(m/10) == 0:
        font = ImageFont.truetype('times.ttf', switch[m%10])
    elif int(m/10) == 1:
        font = ImageFont.truetype('arial.ttf', switch[m%10])
    elif int(m/10) == 2:
        font = ImageFont.truetype('consola.ttf', switch[m%10])
    elif int(m/10) == 3:
        font = ImageFont.truetype('georgia.ttf', switch[m%10])
    else:
        font = ImageFont.truetype('tahoma.ttf', switch[m%10])

    w,h = draw.textsize(num,font=font)
    print(num)
    print(w,h)
    draw.text(xy=((28-w)/2,(28-h)/2), font=font, text=num, fill=(0, 0, 0, 255))

    # # 随机旋转-10-10角度
    # random_angle = random.randint(-10, 10)
    # im_50_rotated = im_50_blank.rotate(random_angle)
    #
    # # 图形扭曲参数
    # params = [1 - float(random.randint(1, 2)) / 100,
    #           0,
    #           0,
    #           0,
    #           1 - float(random.randint(1, 10)) / 100,
    #           float(random.randint(1, 2)) / 500,
    #           0.001,
    #           float(random.randint(1, 2)) / 500]
    #
    # # 创建扭曲
    # im_50_transformed = im_50_rotated.transform((50, 50), Image.PERSPECTIVE, params)

    # # 生成新的30*30空白图像
    # im_30 = im_50_transformed.crop([11, 11, 39, 39])


    return im_50_blank,num


def generate_0toZ(n):
    # 用cnt_num[0]-cnt_num[35]来计0-Z生成的个数，方便之后进行命名
    cnt_num = []
    for i in range(36):
        cnt_num.append(0)

    #n轮
    for m in range(n):
        for index in range(36):
            im,num = generate_single(m,index)
            # 取灰度
            im_gray = im.convert('1')
            cnt_num[index] = cnt_num[index] + 1
            # 输出显示路径
            print("Generate:", path_img + "Sample" + "%03d" % index + "/" + "img%03d" % index + "_" + str(cnt_num[index]) + ".png")
            # 将图像保存在指定文件夹中
            im_gray.save(path_img + "Sample" + "%03d" % index + "/" + "img%03d" % index + "_" + str(cnt_num[index]) + ".png")

    print("\n")
    # 输出分布
    print("生成的0-Z的分布：")
    for k in range(36):
        print("Sample", k if k < 10 else chr(k-10+65), ":", cnt_num[k], "in all")


if __name__ == "__main__":
    mkdir_for_imgs()
    generate_0toZ(50)