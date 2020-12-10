import torch
from torch import optim

OPTIM = {
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha:正样本损失权重占比
    """
    def __init__(self, device=torch.device('cuda'), gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = device
    
    @classmethod
    def weight_pos(self, batch_gt):               # 仅返回0.5/pos 即正例权重（进行归一化）
        assert batch_gt.shape[1] == 1        # 仅允许单通道图像
        total = batch_gt.shape[2]*batch_gt.shape[3]
        pos = batch_gt.view(batch_gt.shape[0],-1).sum(axis=1).float()/total
        positive_weight = 0.5/(pos+0.0001)
        negative_weight = 0.5/(1-pos+0.0001)
    
        return positive_weight / (positive_weight + negative_weight)
 
    def forward(self, pt, target):         # input：经过softmax后的概率
        
        # pt = torch.sigmoid(pt)
        alpha = self.alpha
        
        if not alpha:      # 如果没有alpha（相当于正例权重）则自行计算
            alpha = self.weight_pos(target)
        
        assert len(alpha)==1        # 断定batch_size=1
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt - 1e-10) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + 1e-10)

        loss = torch.mean(loss)
        # print(torch.min(pt).item(), torch.max(pt).item(), 'loss:', loss.item())
        return loss


class DiceLoss(torch.nn.Module):         # Dice损失，可以改善正负样本不平衡时的效果
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss


class DiceBCELoss(torch.nn.Module):
    def __init__(self, BCEWeight):
        super(DiceBCELoss, self).__init__()
        self.weight = BCEWeight
    def forward(self, pred, target):
        dice = DiceLoss()
        bce = torch.nn.BCELoss(weight = self.weight)
        loss = dice(pred, target) + 1e-1*bce(pred, target)
        return loss


LOSS = {
    'bce' : torch.nn.BCELoss,
    'focal' : BCEFocalLoss,
    'dice' : DiceLoss,
    'dicebce' : DiceBCELoss
}