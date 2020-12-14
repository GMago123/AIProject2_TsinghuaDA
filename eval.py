from numpy.lib import utils
import torch
import os
import os.path as osp
import numpy as np
import cv2
import torchvision.transforms.functional as tf
from model import Model
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
import utils


def postprocessing(prob_mask, args, display=False):
    """
    进行后处理（分水岭算法），计算出预测的mask，为每个细胞核用不同序号标记
    :param: prob_mask: 2d 概率图像（0-1）[h, w]
    """
    MORPH_KERNEL = args.MORPH_KERNEL        # 操作核大小
    MORPH_ITERATIONS = args.MORPH_ITERATIONS    # CLOSE操作迭代次数
    DILATE_ITERATIONS = args.DILATE_ITERATIONS     # DILATE膨胀迭代次数
    ERODE_ITERATIONS = args.ERODE_ITERATIONS    # ERODE腐蚀迭代次数
    REAL_THRESHOLD = args.REAL_THRESHOLD    # 预测区域概率阈值
    DIST_THRESHOLD = args.DIST_THRESHOLD    # 核心区域距离阈值
    
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL),np.uint8)    # CLOSE核

    mask = np.uint8(prob_mask > REAL_THRESHOLD)   # 细胞区域
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = MORPH_ITERATIONS) # CLOSE
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = MORPH_ITERATIONS) # OPEN

    bg = cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS)   # 膨胀得到背景区域

    fg = cv2.erode(mask, kernel, iterations=ERODE_ITERATIONS)   # 通过溶解操作获取细胞核的核心区域
    # dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)  # 通过距离预测细胞核的核心区域
    # _, fg = cv2.threshold(dist_transform, DIST_THRESHOLD*dist_transform.max(), 1, 0)

    fg = np.uint8(fg)         # [0. 1.] -> [0 1]
    unsure_area = cv2.subtract(bg, fg)        # 用bg和fg勾勒出不确定区域，值为1，其余区域为0
    num, fg_with_tags = cv2.connectedComponents(fg)    # 前景的相连性分析，将不同细胞核区分开来

    # print('连通域：', np.unique(fg_with_tags))
    
    fg_with_tags += 1
    fg_with_tags[unsure_area == 1] = 0    # 不确定区域标为0. 背景加了1所以变成1
    
    mask_with_tags = cv2.watershed(utils.gray_to_bgr(mask), fg_with_tags)   # 分水岭算法
    mask_with_tags[mask_with_tags < 0] = 0      # 分界线为0
    
    if display:
        plt.subplots(figsize=(16, 8))
        plt.subplot(2,2,1)
        plt.imshow(mask)
        plt.title('Waited for segmented')
        plt.subplot(2,2,2)
        plt.imshow(unsure_area)
        plt.title('Unsure area')
        plt.subplot(2,2,3)
        plt.imshow(fg)
        plt.title('Cell kernels')
        plt.subplot(2,2,4)
        plt.imshow(mask_with_tags)
        plt.title("Segment result")
        print(np.max(np.unique(mask_with_tags)) - 1)      # 细胞个数
        plt.show()
    
    return mask_with_tags, num-1   # 注意连通域种包括一个背景区域，减1为细胞数


def feed_raw_img_unet(unet, raw_img, device):
    '''
    从一个原始图像用unet推测出语义分割区域.
    :param: raw_img: ndarray形式的PIL image，shape:[h, w, 3]
    :return: probs: Tensor形式的2d的灰度图，shape:[1, h, w]
    '''
    h, w = raw_img.shape[0], raw_img.shape[1]
    delta_h = 0 if h % 16 == 0 else 16 - h % 16
    delta_w = 0 if w % 16 == 0 else 16 - w % 16
    padding_h1 = delta_h // 2       # 制定padding方法
    padding_w1 = delta_w // 2
    padding_h2, padding_w2 = delta_h - padding_h1 , delta_w - padding_w1

    img = Image.fromarray(raw_img)            # PIL图像
    padding = (padding_w1, padding_h1, padding_w2, padding_h2)
    # print('padding', padding)
    # print('img', img.size)
    img = tf.pad(img, padding=padding)  # padding成可以被16整除的尺寸
    img = tf.to_tensor(img)       # Tensor
    img = utils.normalize_image(img)       # 正则化
    
    img = img.unsqueeze(0)   # 加入batch_size=1
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        probs = unet(img)
        probs = probs.squeeze(0).cpu()
    
    probs = tf.to_pil_image(probs)         # 转换成PIL，注意由于image采用数据类型是np.uint8，所以精度会有损失
    probs = tf.crop(probs, top=padding_h1, left=padding_w1, height=h, width=w) # 裁剪出原来对应的像素区间
    probs = tf.to_tensor(probs)
    return probs


def predict_img(net, img_fn, device, args, display=False, ensemble=False):
    '''
    给定一个图片文件名，完成细胞核的整个预测流程.
    '''
    net.eval()
    raw_img_PIL = utils.load_image(img_fn)
    raw_img = np.array(raw_img_PIL)
    h = raw_img.shape[0]
    w = raw_img.shape[1]

    if not ensemble:
        prob_mask = feed_raw_img_unet(net, raw_img=raw_img, device=device)
        prob_mask = prob_mask.squeeze().cpu().numpy()
        final_mask, num = postprocessing(prob_mask, args, display)
        raw_mask = (prob_mask > 0.5) + 0
        
    else:
        assert type(net) == list
        with torch.no_grad():
            prob_mask = np.zeros((h, w), dtype=np.float32)
            
            for net_unit in net:
                probs = feed_raw_img_unet(net_unit, raw_img=raw_img, device=device)
                prob_mask += probs.squeeze().cpu().numpy()
                
            prob_mask /= len(net)       # 平均数投票
        raw_mask = (prob_mask > 0.5) + 0
        
        final_mask, num = postprocessing(prob_mask, args, display)

    return raw_mask, final_mask, num


def eval(args):
    net = Model(args.modelName, args)

    args.device = 'cpu'

    net.to(device=args.device)
    net.load_state_dict(torch.load(args.eval, map_location=args.device))

    X_test = sorted([osp.join(args.data_dir, image) for image in os.listdir(args.data_dir) if image.find('mask.png') == -1])
    
    if not osp.exists(args.res_save_path):
        os.makedirs(args.res_save_path)

    fn_with_cell_num = []
    print("正在进行模型推断与后处理...")
    for i, fn in enumerate(X_test):
        
        raw_mask, mask, num = predict_img(net=net,
                                          img_fn=fn,
                                          device=args.device,
                                          args=args,
                                          display=args.display,
                                          ensemble=False)
        fn_with_cell_num.append(fn.split('/')[-1] + ', ' + str(num))      # 记录文件名+细胞核个数
        label = np.unique(mask)
        height, width = mask.shape[:2]
        visual_img = np.zeros((height, width, 3))
        for lab in label:
            if lab == 0:
                continue
            color = np.random.randint(low=0, high=255, size=3)
            visual_img[mask==lab, :] = color        
        if args.colored:
            cv2.imwrite(osp.join(args.res_save_path, fn.split('/')[-1].rstrip('.png') + '_colored.png'), 
                    visual_img.astype(np.uint8))
        else:
            mask = raw_mask*255
            cv2.imwrite(osp.join(args.res_save_path, 
                    'test_' + fn.split('/')[-1].rstrip('.png') + '_pred.png'), mask.astype(np.uint8))
    
    with open(osp.join(args.res_save_path, 'count_result.txt'), 'w') as f:
        for record in fn_with_cell_num:
            f.write(record + '\n')
    print("结果保存至" + args.res_save_path)


if __name__ == "__main__":
    from config import Config
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.modelName = 'unet'
    config = Config(args)
    args = config.get_config()
    
    args.eval = './../result/iou64/model/unet-2020-12-14.pth'
    args.device = 'cpu'
    args.res_save_path = './../result/iou64/final_data/'

    net = Model(args.modelName, args)
    net.to(device=args.device)
    net.load_state_dict(torch.load(args.eval, map_location=args.device))

    img_fn = './../q2_data/test/1.png'

    raw_mask, mask, num = predict_img(net=net,
                       img_fn=img_fn,
                       device=args.device,
                       args=args,
                       display=True,
                       ensemble=False)
    
    print(num)

    label = np.unique(mask)
    height, width = mask.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[mask==lab, :] = color
    
    if not osp.exists(args.res_save_path):
        os.makedirs(args.res_save_path)
        
    cv2.imwrite(osp.join(args.res_save_path, '1.png'.rstrip('.png') + '_mask.png'), visual_img.astype(np.uint8))

