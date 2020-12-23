# 细胞核分割工程

## 数据说明

- 报告中(4.1)实验方法(2)提到的抽取的9对图片作为模型的测试集，包含以下图片（仅列出原图像名字）：
  - z7-4_1.png, z7-4_2.png, z16b_2.png, z19b_1.png, z44b_2.png, z56b_1.png, z59_1.png, z59_1.png, z59_1.png
  - 以上文件在实验方法2中被放在q2_data/val/文件夹下
- 实验方法(1)下数据文件夹不做改动
- AIProject2与q2_data放在同一文件夹下
- AIProject2/results/model_saves存放着最终用于预测的模型参数，注意unet基础特征图数为32
- AIProject2/results/result_saves存放着一些实验结果csv



## AIProject代码结构说明

AIProject2是工程的主文件夹，代码结构化以提供一个较为通用的细胞分割算法框架。

代码运行方式：

- 训练：

```
run.py [-h] [--modelName MODELNAME] [--loss LOSS]
              [--pretrained_path PRETRAINED_PATH]
              [--model_save_path MODEL_SAVE_PATH] [--eval EVAL]
              [--res_save_path RES_SAVE_PATH] [--data_dir DATA_DIR]
              [--test_dir TEST_DIR] [--gpu_ids GPU_IDS]
```

- modelName:  目前只支持unet'\，载入UNet模型
- loss：采用的损失函数，支持bce, focal, dice, iou损失函数
- pretrained_path（可选）：如果要在之前的某个模型参数下继续训练，指定.pth的文件路径
- model_save_path（可选，默认results/model_saves）：模型参数.pth保存路径
- eval（可选）：载入模型参数.pth的文件路径。如果此项不为空，程序将进入模型推断模式，载入这个参数指定的模型并对data_dir目录下的图片进行预测、后处理并输出预测结果，包括count_result.txt和最终的mask.png
- res_save_path（可选，默认results/result_saves）：模型评估结果表格保存路径
- test_dir（非eval模式下必须指定）：用于模型评估效果的测试集目录（图片需要同时具有原图像和mask）
- gpu_ids（可选，默认为0）：指定使用的gpu编号列表

示例执行代码在run.ipynb笔记本中。

**run.py 在main函数里指定随机数种子列表** 

除此之外，其他的一些参数在config.py中，可以指定：

- 'n_channels': 3：Unet的输入图像通道数
- 'n_classes': 1：输出图像通道数，即类别数，2分类情况下为1
- 'feature_maps': 32：Unet第一次提取的基础特征图个数，实验中可选32或64（其实多少都行）
- 'bilinear': False：上采样是否采用双线性插值
- 'crop_size': 384：随机裁剪的图像尺寸
- 'crop' : True：是否进行随机裁剪，必须为True，否则尺寸无法对齐
- 'batch_size': 5
- 'early_stop': 6：训练早停的轮数
- 'scheduler': True：是否采用scheduler减小学习率
- 'scheduler_criterion': 'f1'：scheduler最大化原则，可选f1，accuracy，recall
- 'valid' : 0.15：验证集比例
- 'optim': 'rmsprop'：优化方法，可选rmsprop，sgd，adam
- 'lr': 0.0001：学习率
- 'momentum': 0.9：动量
- 'weight_decay': 1e-8：模型正则化因子
- 'epochs': 40：最大轮数
- 'save_figure_path': './figure',     训练指标图像保存目录，可看到loss以及验证分数变化趋势
- 'center_enforcement': False：试图进行图像中心增强，告吹了，不用动
- 'display': False：是否展示后处理过程（实时显示plt图片）
- 'colored': False：是否保存彩色标记的分水岭算法结果；为False时正常保存test_XX_pred.png的mask
- 后处理参数
  - 'MORPH_KERNEL' : 3,     # 形态操作膨胀核大小
  - 'MORPH_ITERATIONS' : 3,   # 形态操作迭代次数
  - 'DILATE_ITERATIONS': 3,   # DILATE膨胀迭代次数(获取背景)
  - 'ERODE_ITERATIONS': 4,   # ERODE腐蚀迭代次数
  - 'DIST_THRESHOLD' : 0.4,   # 寻找中心区域时，距离阈值（已弃用）
  - 'REAL_THRESHOLD' : 0.5,   # 细胞区域概率阈值



## 代码说明

- config.py 配置
  - class Storage(dict)：（有外界代码参考）可以通过.操作读取字典的存储
  - class Config()：配置类，存储了参数，get_config返回一个storage，与命令行args对象的存储类似
- dataloader.py 数据载入
  - getData(data_dir)：得到训练样本的文件路径列表
  - MyDataset(Dataset): 定义的与pytorch相容的数据集，注意初始化时如果要载入的时验证集或测试集，要制定val=True，这样才不会被随机裁剪
  - 主函数是做测试用的，载入一个样本
- eval.py 模型推断
  - postprocessing(prob_mask, args, display=False) 概率图像后处理
  - feed_raw_img_unet(unet, raw_img, device) 从一个原始图像用unet推测出语义分割区域，包含padding与crop等标准化过程
  - predict_img(net, img_fn, device, args, display=False, ensemble=False) 给定单个原始图像路径，返回预测的mask，经过后处理的具标签的mask，细胞数量
  - eval(args) 模型推断主函数
  - main函数是测试代码
- loss.py 定义损失函数和优化器
- model.py 定义unet
- run.ipynb 示例命令行代码
- **run.py 执行训练程序，须在这里指定随机数种子列表** 
- train.py 模型训练
  - Train类：可执行模型训练、计算交叉验证分数、计算测试集分数
- utils.py 小工具
  - normalize_image(image_tensor) 图像标准化
  - load_image(file_name) 图像加载
  - gray_to_bgr(img) 灰度图转换为BGR图像
  - pixel_statistics(data_dir) 计算训练图像统计数据（发现像素点过多情况下numpy、torch计算出的均值不一样，画图发现可能是torch的均值机制有些毛病
  - main函数执行pixel_statistics