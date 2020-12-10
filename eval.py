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
from utils import gray_to_bgr, load_image


def postprocessing(prob_mask, args, display=False):
    BLUR_KERNEL = args.BLUR_KERNEL          # 中值模糊核大小
    MORPH_KERNEL = args.MORPH_KERNEL        # 腐蚀膨胀核大小
    MORPH_ITERATIONS = args.MORPH_ITERATIONS    # CLOSE操作迭代次数
    DIST_THRESHOLD = args.DIST_THRESHOLD    # 寻找中心区域时，距离阈值
    BG_THRESHOLD = args.BG_THRESHOLD        # 背景区域的概率阈值
    FG_THRESHOLD = args.FG_THRESHOLD        # 前景区域（细胞核心区域）的概率阈值
    REAL_THRESHOLD = args.REAL_THRESHOLD    # 细胞区域概率阈值

    
    # mask = cv2.medianBlur(mask, BLUR_KERNEL)       # 中值模糊
    
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL),np.uint8)    # 核
    # openImage = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = MORPH_ITERATIONS)  # open操作：先腐蚀再膨胀
    # bg = cv2.dilate(mask, kernel, iterations = 3)     # 膨胀一定程度作为背景background
    
    # center_mask = cv2.morphologyEx(center_mask, cv2.MORPH_CLOSE, kernel, iterations = MORPH_ITERATIONS)  # 先膨胀再腐蚀，意图消去弱连接
    bg = np.uint8(prob_mask > BG_THRESHOLD)   # 背景区域
    
    fg = np.uint8(prob_mask > FG_THRESHOLD)   # 细胞置信度较大区域
    dist_transform = cv2.distanceTransform(fg, cv2.DIST_L2, 5)   #CV_DIST_L2（maskSize=5）的计算结果是精确的
    ret, fg = cv2.threshold(dist_transform, DIST_THRESHOLD*dist_transform.max(), 255, 0)     # 距离变换获取中心区域，即前景foreground
    fg = np.uint8(fg > 0)      # 前景区域
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations = MORPH_ITERATIONS)  # 消除弱连接
    
    unknow = cv2.subtract(bg, fg)        # 用bg和fg勾勒出不确定区域，值为1，其余区域为0
    
    ret, tagging = cv2.connectedComponents(fg)    # 前景的相连性分析，将不同细胞核区分开来
    
    tagging += 1
    tagging[unknow == 1] = 0    # 不确定区域标为0. 背景加了1所以变成1
    
    origin_mask = np.uint8(prob_mask > REAL_THRESHOLD)   # 细胞区域
    # print(np.max(prob_mask))
    
    final = cv2.watershed(gray_to_bgr(origin_mask), tagging)   # 分水岭算法
    final -= 1       # 背景为1，分界线为-1，减1加阈值，均变为0
    final[final<0] = 0
    
    if display:
        plt.subplots(figsize=(16, 8))
        plt.subplot(2,2,1)
        plt.imshow(origin_mask)
        plt.title('Waited for segmented')
        plt.subplot(2,2,2)
        plt.imshow(unknow)
        plt.title('Unsure area')
        plt.subplot(2,2,3)
        plt.imshow(fg)
        plt.title('Cell kernels')
        plt.subplot(2,2,4)
        plt.imshow(final)
        plt.title("Segment result")
        print(np.unique(final))       # 打印输出有多少个样本
        plt.show()
    
    return final


def feed_raw_img_unet(unet, raw_img, device):
    img = raw_img.cpu()/255       # 变换至0-1区间
    h, w = img.shape[1], img.shape[2]
    delta_h = 0 if h % 16 == 0 else 16 - h % 16
    delta_w = 0 if w % 16 == 0 else 16 - w % 16
    padding_h1 = delta_h // 2       # 制定padding方法
    padding_w1 = delta_w // 2
    padding_h2, padding_w2 = delta_h - padding_h1 , delta_w - padding_w1

    img = tf.to_pil_image(img)             # PIL图像
    padding = (padding_w1, padding_h1, padding_w2, padding_h2)
    # print('padding', padding)
    # print('img', img.size)
    img = tf.pad(img, padding=padding)  # padding成可以被16整除的尺寸
    img = tf.to_tensor(img)       # Tensor
    
    img = img.unsqueeze(0)   # 加入batch_size=1
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        probs = unet(img)
        probs = probs.squeeze(0).cpu()
    
    probs = tf.to_pil_image(probs)         # 转换成PIL
    probs = tf.crop(probs, top=padding_h1, left=padding_w1, height=h, width=w) # 裁剪出原来对应的像素区间
    probs = tf.to_tensor(probs)
    return probs


def predict_img(net, img_fn, device, args, display=False, ensemble=False):
    net.eval()
    raw_img = load_image(img_fn)
    raw_img = torch.Tensor(raw_img.transpose((2, 0, 1)))     # 通道置于第一维度
    h = raw_img.shape[1]
    w = raw_img.shape[2]

    if not ensemble:
        prob_mask = feed_raw_img_unet(net, raw_img=raw_img, device=device)
        prob_mask = prob_mask.squeeze().cpu().numpy()
        final_mask = postprocessing(prob_mask, args, display)
        
    else:
        assert type(net) == list
        with torch.no_grad():
            prob_mask = np.zeros((h, w), dtype=np.float32)
            
            for net_unit in net:
                probs = feed_raw_img_unet(net, raw_img=raw_img, device=device)
                prob_mask += probs.squeeze().cpu().numpy()
                
            prob_mask /= len(net)       # 平均数投票
        
        
        final_mask = postprocessing(prob_mask, args, display)

    return final_mask


def eval(args):
    net = Model(args.modelName, args)

    net.to(device=args.device)
    net.load_state_dict(torch.load(args.eval, map_location=args.device))

    X_test = sorted([osp.join(args.data_dir, image) for image in os.listdir(args.data_dir) if not image.find('mask.png')])
    
    if not osp.exists(args.res_save_path):
        os.makedirs(args.res_save_path)

    for i, fn in enumerate(X_test):
        
        mask = predict_img(net=net,
                        img_fn=fn,
                        device=args.device,
                        args=args,
                        display=args.display,
                        ensemble=False)
        cv2.imwrite(osp.join(args.res_save_path, 'mask{:0>3d}.png'.format(i)), mask.astype(np.uint8))
