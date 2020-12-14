import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

def postprocessing(prob_mask, args, display=False):
    """
    进行后处理（分水岭算法），计算出预测的mask，为每个细胞核用不同序号标记
    :param: prob_mask: 2d 概率图像（0-1）[h, w]
    """
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
    
    fg = np.uint8(prob_mask > FG_THRESHOLD)   # 细胞置信度较大区域，前景区域
    # dist_transform = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    # ret, fg = cv2.threshold(dist_transform, DIST_THRESHOLD*dist_transform.max(), 255, 0)     # 距离变换获取中心区域，即前景foreground
    fg = np.uint8(fg > 0)      # 前景区域
    # fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations = MORPH_ITERATIONS)  # 消除弱连接
    
    unsure_area = cv2.subtract(bg, fg)        # 用bg和fg勾勒出不确定区域，值为1，其余区域为0
    ret, fg_with_tags = cv2.connectedComponents(fg)    # 前景的相连性分析，将不同细胞核区分开来
    
    fg_with_tags += 1
    fg_with_tags[unsure_area == 1] = 0    # 不确定区域标为0. 背景加了1所以变成1
    
    mask = np.uint8(prob_mask > REAL_THRESHOLD)   # 细胞区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = MORPH_ITERATIONS)
    # print(np.max(prob_mask))
    
    mask_with_tags = cv2.watershed(utils.gray_to_bgr(mask), fg_with_tags)   # 分水岭算法
    # mask_with_tags -= 1       # 背景为1，分界线为-1，减1加阈值，均变为0
    mask_with_tags[mask_with_tags < 0] = 0
    
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
        print(np.max(np.unique(mask_with_tags)+1))       # 打印输出有多少个样本
        plt.show()
    
    return mask_with_tags