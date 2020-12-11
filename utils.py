import os
import re
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import numpy as np


def normalize_image(image_tensor):
    assert type(image_tensor) == torch.Tensor,  '输入图像不是Tensor形式，无法正则化'
    # mean = np.array([151.29554595, 69.58749789, 179.07634817])
    mean = np.array([0.70226019, 0.27289215, 0.59331587])
    # std = np.array([38.54135229, 43.51355295, 46.54086734])
    std = np.array( [0.18251321, 0.17064138, 0.15114256])
    img = tf.normalize(tensor=image_tensor, mean=mean, std=std)
    return img


def load_image(file_name):  # 单张图片
    '''
    载入图片为PIL对象.
    '''
    assert os.path.exists(file_name), '文件不存在.'
    # img = cv2.imread(file_name, -1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.open(file_name)
    mode = img.mode

    if len(mode) == 1:
        assert mode == 'L', 'Label图像不正常:' + mode
        # to_tensor后255变为1，并且形成[1, h, w]shape的张量
        return img
    else:
        assert mode == 'RGB', '原始图像不正常:' + mode
        # 加载后进行张量标准化的工作需要后续工作进行，避免影响图片裁剪
        return img

def gray_to_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.uint8(img)

def pixel_statistics(data_dir):
    # 计算像素RGB通道均值、标准差

    images = []
    files = os.listdir(data_dir)
        
    for file in files:       # 找出图片文件
        matchObj = re.match(r'(.*)_mask(.*)', file)
        if not matchObj:
            if file.find('.png') != -1:
                image_path = os.path.join(data_dir, file)
                images.append(image_path)

    print('原图像个数：', len(images))

    pixels = []
    pixels_numpy = []
    for img_fn in images:
        img = Image.open(img_fn)
        assert img.mode == 'RGB', '非RGB图像' + str(img.mode)
        pixels_numpy.append((np.array(img)/255).reshape(-1, 3))
        img = tf.to_tensor(img)
        img = img.permute(1, 2, 0).view(-1, 3)
        
        pixels.append(img)

    pixels = torch.cat(pixels)
    pixels_numpy = np.concatenate(pixels_numpy)

    print('像素点个数(Tensor):', pixels.shape, ' 像素点个数(ndarray):', pixels_numpy.shape)
    
    diff = pixels_numpy - np.array(pixels)
    print('一范数:', np.sum(np.abs(diff.ravel())))

    # 针对误差画图
    numpy_mean = []
    torch_mean = []
    x = []
    for i in range(100, diff.shape[0], 1000000):
        x.append(i)
        pixels_ = pixels[0: i, :]
        pixels_numpy_ = pixels_numpy[0: i, :]
        numpy_mean.append(np.mean(pixels_numpy_, axis=0))
        torch_mean.append(torch.mean(pixels_, dim=0))
    numpy_mean = np.concatenate(numpy_mean).reshape((-1,3))
    torch_mean = torch.cat(torch_mean).view(-1,3)

    import matplotlib.pyplot as plt
    plt.plot(x, numpy_mean[:, 0], label='R_numpy', color='red')
    plt.plot(x, numpy_mean[:, 1], label='G_numpy', color='red')
    plt.plot(x, numpy_mean[:, 2], label='B_numpy', color='red')
    plt.plot(x, torch_mean[:, 0], label='R_torch', color='blue')
    plt.plot(x, torch_mean[:, 1], label='G_torch', color='blue')
    plt.plot(x, torch_mean[:, 2], label='B_torch', color='blue')
    plt.legend()
    plt.show()

    print('tensor mean: ', torch.mean(pixels, dim=0), 'ndarray mean: ', np.mean(pixels_numpy, axis=0))
    print('tensor std: ', torch.std(pixels, dim=0), 'ndarray std: ', np.std(pixels_numpy, axis=0))



if __name__ == "__main__":
    data_dir = './../q2_data/train/'
    pixel_statistics(data_dir)