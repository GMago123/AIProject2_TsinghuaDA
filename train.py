import torch
import numpy as np
from tqdm import tqdm
import os
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as tf
from dataloader import MyDataset, getData
from loss import LOSS, OPTIM
from eval import feed_raw_img_unet
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image


class Train:
    def __init__(self, net, args):
        self.net = net
        self.device = args.device
        self.args = args
        self.loss = LOSS[args.loss]()
        self.optim = OPTIM[args.optim]

    def train_net(self):
        args = self.args
        X, Y = getData(args.data_dir)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=args.valid, random_state=args.seed)
        # 加载训练集和交叉验证集
        trainData = MyDataset(X_train, Y_train, args)
        valData = MyDataset(X_valid, Y_valid, args, val=True)

        train_loader = DataLoader(trainData, batch_size=trainData.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(valData, batch_size=valData.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        
        # 记录训练步数
        global_step = 0
        
        # 优化器

        optimizer = self.optim(self.net.parameters(), lr=args.lr, 
                                weight_decay=args.weight_decay, momentum=args.momentum)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
            
        val_scores = []   # 交叉验证分数记录
        loss_record = []   # 损失函数分数记录

        for epoch in range(args.epochs):
            self.net.train()

            epoch_loss = []
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
                for batch in train_loader:
                    imgs = batch['img']
                    true_masks = batch['gt']
                    assert imgs.shape[1] == self.net.model.n_channels, "Incorrect channels."            # 判断通道数是否正确

                    imgs = imgs.to(device=self.device, dtype=torch.float32)
                    mask_type = torch.float32 if self.net.model.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=self.device, dtype=mask_type)    # 转换到gpu上
                    masks_pred = self.net(imgs)
                    
                    # 损失函数权重计算
                    # w = pixel_weight(batch_gt=true_masks.cpu(), batch_labels=batch['labels'].cpu())
                    # w = torch.tensor(w).to(device=device, dtype=mask_type)
                    # 损失函数定义
                    
                    # 计算损失
                    loss = self.loss(masks_pred, true_masks)
                    epoch_loss.append(loss.item())

                    pbar.set_postfix(**{'loss(epoch)' : np.sum(epoch_loss)/len(epoch_loss)})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(self.net.parameters(), 0.1)  # 梯度裁剪
                    optimizer.step()

                    pbar.update(1)     # 每推断完1个batch进度条+1，进度条显示batch数
                    global_step += 1
            
            loss_record.append(np.sum(epoch_loss)/len(epoch_loss))   # 记录当前的loss

            # 每个epoch结束后进行交叉验证
            acc_list, f1_list, recall_list, _, losses, _ = self.call_val(val_loader)
            criterion = {'accuracy': acc_list, 'recall': recall_list, 'f1': f1_list}
            val_score_list = criterion[args.scheduler_criterion]

            avg_score = np.sum(val_score_list) / len(val_score_list)

            val_scores.append(avg_score)       # 记录验证的val_score
            
            if args.scheduler:
                scheduler.step(avg_score)
                            
            dir_checkpoint = args.model_save_path
            if args.save and (epoch + 1) % args.save_epochs == 0:  # 保存参数
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                torch.save(self.net.state_dict(), os.path.join(dir_checkpoint, f'{args.modelName}-'+\
                                        datetime.datetime.now().strftime('%Y-%m-%d')+ '-'\
                                            f'epoch{epoch + 1}.pth'))
        
        plt.subplots(figsize=(12, 6))
        plt.subplot(1,2,1)
        plt.plot(np.arange(1, len(val_scores)+1), val_scores)
        plt.title('Val_Score')
        plt.subplot(1,2,2)
        plt.plot(np.arange(1, len(loss_record)+1), loss_record)
        plt.title('Train_Loss')
        if not os.path.exists(args.save_figure_path):
            os.makedirs(args.save_figure_path)
        plt.savefig(os.path.join(args.save_figure_path, 'train.jpg'), dpi=100)


    def call_val(self, loader, net=None):         # 计算验证集上的Accuracy
        if not net:
            net = self.net
        net.eval()
        mask_type = torch.float32 if net.model.n_classes == 1 else torch.long

        accs, f1s, recalls, names, losses, preds = [], [], [], [], [], []
        
        for batch in loader:
            imgs, masks_true, img_names = batch['raw_img'], batch['raw_gt'], batch['name']

            for img_index in range(imgs.shape[0]):     # 对batch内的每一张图片进行测试
                img = np.array(imgs[img_index])      # ndarray形式的PIL图像
                mask_true = tf.to_tensor(Image.fromarray(np.array(masks_true[img_index])))  # mask转换为标准图像张量
                name = img_names[img_index]
                names.append(name)
                
                with torch.no_grad():
                    prob_pred = feed_raw_img_unet(net, img, device=self.device)
                
                # print(prob_pred.shape, mask_true.shape)
                mask_pred = (prob_pred > 0.5).float()        # mask预测结果，0.5为阈值
                preds.append(prob_pred)      # 预测结果图像（Tensor形式）

                accs.append(accuracy_score(mask_true.view(-1).cpu(), mask_pred.view(-1)))
                f1s.append(f1_score(mask_true.view(-1).cpu(), mask_pred.view(-1)))
                recalls.append(recall_score(mask_true.view(-1).cpu(), mask_pred.view(-1)))

                prob_pred = prob_pred.unsqueeze(0)
                mask_true = mask_true.unsqueeze(0)
                losses.append(self.loss(prob_pred, mask_true.cpu()).item())
            

        net.train()
        return accs, f1s, recalls, names, losses, preds


    def result(self, net, data_dir, args):        # 计算在某指定数据集上的相关指标
        print('Testing images in '+ data_dir)
        X, Y = getData(data_dir)
        dataset = MyDataset(X, Y, args, val=True)
        loader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=False, pin_memory=True)
        accs, f1s, recalls, names, losses, preds = self.call_val(loader, net=net)
        
        print('saving probability results to ' + args.res_save_path)
        for i in range(len(names)):
            preds_save_dir = os.path.join(args.res_save_path, 'prob_masks/')
            if not os.path.exists(preds_save_dir):
                os.makedirs(preds_save_dir)
            save = os.path.join(preds_save_dir, names[i].split('/')[-1].rstrip('.png') + '_prob.png')
            prob_img = tf.to_pil_image(preds[i])
            prob_img.save(save)
            
        return {'avg-acc': np.average(accs)*100, 
                'avg-recall': np.average(recalls)*100, 
                'avg-f1': np.average(f1s)*100}
