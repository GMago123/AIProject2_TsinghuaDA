from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class DoubleConv(nn.Module):       # 两次卷积单元，第一次卷积会扩大通道数
    """(conv + BatchNorm + ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.conv_kernel = 3   # 卷积核大小
        self.padding = (self.conv_kernel-1) // 2
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=self.conv_kernel, padding=self.padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=self.conv_kernel, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Contracting(nn.Module):           # 收缩路径单元
    """2*2 maxpool 降采样 + double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Expansive(nn.Module):             # 扩张路径单元
    """反卷积 Upscaling + double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.upconv_kernel = 2          # 上卷积核大小

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=self.upconv_kernel, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)            # 与扩张路径对应单元拼接
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, basic_channels=64, bilinear=False):
        """bilinear: 上采样方法采用双线性插值，否则采用反卷积"""
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.basic_channels = basic_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, basic_channels)
        self.down1 = Contracting(basic_channels, 2*basic_channels)
        self.down2 = Contracting(2*basic_channels, 4*basic_channels)
        self.down3 = Contracting(4*basic_channels, 8*basic_channels)
        factor = 2 if bilinear else 1
        self.down4 = Contracting(8*basic_channels, 16*basic_channels // factor)
        self.up1 = Expansive(16*basic_channels, 8*basic_channels // factor, bilinear)
        self.up2 = Expansive(8*basic_channels, 4*basic_channels // factor, bilinear)
        self.up3 = Expansive(4*basic_channels, 2*basic_channels // factor, bilinear)
        self.up4 = Expansive(2*basic_channels, basic_channels, bilinear)
        self.outc = OutConv(basic_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        log_logits = self.outc(x)
        
        if self.n_classes > 1:
            logits = F.softmax(log_logits, dim=1)
        else:
            logits = torch.sigmoid(log_logits)
        
        return logits


class Model(nn.Module):
    def __init__(self, modelName, args):
        super(Model, self).__init__() 
        self.args = args
        self.modelName = modelName
        self.MODELS = {
            'unet': self.__unet,
        }
        self.model = self.MODELS[self.modelName]()


    def __unet(self):
        return UNet(n_channels=self.args.n_channels,
                    n_classes=self.args.n_classes,
                    basic_channels=self.args.feature_maps,
                    bilinear=self.args.bilinear)

        
    def forward(self, x):
        assert self.model, '模型未activate'
        return self.model(x)

