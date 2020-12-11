import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
import random
import cv2
import utils
import matplotlib.pyplot as plt
import os
import os.path as osp
import re


def getData(data_dir):
    '''
    Get the file names of the image and mask.
    '''
    images = []
    masks = []
    files = os.listdir(data_dir)
    for file in files:
        matchObj = re.match(r'(.*)_mask(.*)', file)
        if matchObj:
            image_name = matchObj.group(1) + matchObj.group(2)
            image = osp.join(data_dir, image_name)
            mask = osp.join(data_dir, matchObj.group())
            assert image_name in files, 'mask没有对应的image'
            masks.append(mask)
            images.append(image)

    return images, masks


class MyDataset(Dataset):
    def __init__(self, img_files, gt_files, args, val=False):
        self.img_files = img_files
        self.gt_files = gt_files
        assert len(img_files) == len(gt_files), '样本与标记数目不一致.'
        self.len = len(img_files)
        self.centered = args.center_enforcement
        self.crop_size = args.crop_size
        self.crop = args.crop
        self.batch_size = args.batch_size
        self.val = val
        if val:
            self.batch_size = 1
    
    def train_transform(self, img_PIL, gt_PIL):
        """输入图像、标注图片格式转换"""

        if self.crop:
            i, j, h, w = transforms.RandomCrop.get_params(
                img_PIL, output_size=(self.crop_size, self.crop_size))  # 随机裁剪
            # print(i, j, h, w)
            img_PIL = tf.crop(img_PIL, i, j, h, w)
            gt_PIL = tf.crop(gt_PIL, i, j, h, w)

        if random.random() > 0.5:           # 随机垂直翻转
            img_PIL = tf.hflip(img_PIL)
            gt_PIL = tf.hflip(gt_PIL)
        if random.random() > 0.5:         # 随机水平翻转
            img_PIL = tf.vflip(img_PIL)
            gt_PIL = tf.vflip(gt_PIL)
        
        img = tf.to_tensor(img_PIL)
        img = utils.normalize_image(img)
        gt = tf.to_tensor(gt_PIL)
        
        return img, gt
    
    @classmethod
    def center_enforce(cls, labels_mask):
        DIST_KERNEL = 0.4          # 细胞核心区域阈值
        
        labels = np.unique(labels_mask)
        enforced_gt = np.zeros((1, labels_mask.shape[0], labels_mask.shape[1]), dtype=np.float32)
        for label in labels:
            temp_mask = np.zeros(labels_mask.shape, dtype=np.uint8)
            temp_mask[labels_mask == label] = 1           # 非细胞区域标1以求距离
        
            BLUR_KERNEL = 3
            temp_mask = cv2.medianBlur(temp_mask, BLUR_KERNEL)       # 中值模糊，去除噪声，本数据集中单个细胞的噪声较大
        
            cell_pixels = temp_mask.sum()
            if cell_pixels < 10:     # 少于10个像素点的标签抛弃
                continue
        
            dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)        # 距离变换
            dist_transform = dist_transform / np.max(dist_transform)
            enforced_gt[0] += dist_transform
        enforced_gt[0] *= (labels_mask > 0)
        enforced_gt[0] = cv2.medianBlur(enforced_gt[0], 3)       # 中值模糊，去除尖峰
        enforced_gt[0] = np.uint8(enforced_gt > DIST_KERNEL)
        
        return enforced_gt
        
    
    def __getitem__(self, index):
        raw_img_PIL = utils.load_image(self.img_files[index])
        raw_gt_PIL = utils.load_image(self.gt_files[index])

        img, mask = self.train_transform(raw_img_PIL, raw_gt_PIL)
        #if self.centered:
        #    mask = self.center_enforce(labels)
        
        if self.val:      # 注意raw_gt是2d的；raw_img通道在最后一维(h,w,3)
            return {'raw_img': np.array(raw_img_PIL), 'raw_gt': np.array(raw_gt_PIL), 'name': self.img_files[index]} 
        else:
            return {'img': img, 'gt': mask} 

    # 返回数据集长度
    def __len__(self):
        return self.len


if __name__ == "__main__":
    X_train, Y_train = getData('./../q2_data/train/')


    from config import Config
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.modelName = 'unet'
    config = Config(args)
    args = config.get_config()
    
    temp_loader_train = DataLoader(dataset=MyDataset(X_train, Y_train, args, val=False), shuffle=False)
    temp_loader_val = DataLoader(dataset=MyDataset(X_train, Y_train, args, val=True), shuffle=False)
    temp1, temp2, temp3, temp4, name = None, None, None, None, None
    for item in temp_loader_train:
        temp1 = item['img'][0]
        temp2 = item['gt'][0]
        print('img shape:', temp1.shape, '\ngt shape:', temp2.shape)
        break
    for item in temp_loader_val:
        name = item['name'][0]
        temp3 = np.array(item['raw_img'][0])
        temp4 = np.array(item['raw_gt'][0])
        print('raw_img shape:', temp3.shape, '\nraw_gt shape', temp4.shape)
        break

    temp1 = np.asarray(temp1[0])   # 仅取第1通道
    temp2 = np.asarray(temp2[0])   # 仅取第1通道
    temp3 = temp3.transpose((2,0,1))[0]

    plt.subplot(2, 2, 1)
    plt.imshow(temp1)
    print('img range: ', np.min(temp1), ',', np.max(temp1), 'shape: ', temp1.shape)

    plt.subplot(2, 2, 2)
    plt.imshow(temp2)
    print('gt range: ', np.min(temp2), ',', np.max(temp2), 'shape: ', temp2.shape)
    print('labels:', np.unique(temp2))
    print('Positive propotion:', np.sum(temp2.ravel()) / (temp2.shape[0]*temp2.shape[1]))
    
    plt.subplot(2, 2, 3)
    plt.imshow(temp3)
    print('raw_img range: ', np.min(temp3), ',', np.max(temp3), 'shape: ', temp3.shape)

    plt.subplot(2, 2, 4)
    plt.imshow(temp4)
    print('raw_gt range: ', np.min(temp4), ',', np.max(temp4), 'shape: ', temp4.shape)

    plt.title(name)
    plt.show()
    
