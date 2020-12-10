import cv2
import numpy as np


def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))


def load_image(file_name):  # 单张图片
    img = cv2.imread(file_name, -1)
    #if len(img.shape) == 3:        # 如果读取的是RGB图像，需要将通道维度移至第一维
    #    img = img.transpose((2,0,1))
    return img


def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 

def gray_to_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.uint8(img)

