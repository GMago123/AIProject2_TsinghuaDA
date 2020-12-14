import torch
from torch import optim

OPTIM = {
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

POSITIVE_PRIORITY = 0.1335         # 计算出来的正样本先验

class BFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha:正样本损失权重占比
    """
    def __init__(self, device=torch.device('cuda'), gamma=2, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = device
 
    def forward(self, input, target):         # input：经过softmax后的概率
        
        # pt = torch.sigmoid(pt)
        alpha = self.alpha
        
        loss = - alpha * (1 - input) ** self.gamma * target * torch.log(input - 1e-10) - \
               (1 - alpha) * input ** self.gamma * (1 - target) * torch.log(1 - input + 1e-10)

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


class IoULoss(torch.nn.Module):          # A交B/A并B
    def __init__(self):
        super(IoULoss, self).__init__()
    
    def forward(self, input, target):

        inter = input * target
        inter = inter.view(input.shape[0], input.shape[1], -1).sum(2)     # 交集大小

        union = input + target - input*target
        union = union.view(input.shape[0], input.shape[1], -1).sum(2)     # 并集大小

        iou = inter/(union + 1e-16)     # 加入平滑因子的比值
        loss = 1 - iou.mean()

        return loss


LOSS = {
    'bce' : torch.nn.BCELoss,
    'focal' : BFocalLoss,
    'dice' : DiceLoss,
    'iou': IoULoss,
}