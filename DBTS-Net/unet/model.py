import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inSize, outSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(outSize),
            nn.ReLU(inplace=True),
            nn.Conv2d(outSize, outSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(outSize),
            nn.ReLU(inplace=True)
        )
    #     Sequential===============================
    def forward(self, x):
        return self.conv(x)



class Down(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inSize, outSize))

    def forward(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, inSize, outSize):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inSize, outSize, 1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = DoubleConv(inSize, outSize)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SGVP(nn.Module):
    """Steger Guided Vascular Prior (SGVP) module for vessel segmentation"""
    def __init__(self, nChannel, nClass):
        super(SGVP, self).__init__()
        # 修改为接受多通道输入
        self.inc = DoubleConv(nChannel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, nClass)
        
        # 添加血管特征注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),  # 将2通道特征图融合为1通道注意力图
            nn.Sigmoid()
        )

    def forward(self, x):
        # x是2通道输入: [batch, 2, height, width]
        
        # 生成注意力权重，聚焦于血管区域
        attention_weight = self.attention(x)
        
        # 将注意力应用到输入
        enhanced_input = x[:, 0:1, :, :] * attention_weight + x[:, 0:1, :, :]
        
        # 构建增强后的输入，保留原始信息
        enhanced_x = torch.cat([enhanced_input, x[:, 1:2, :, :]], dim=1)
        
        # 标准UNet前向传播
        x1 = self.inc(enhanced_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

# 保留原始UNetWithSteger类以保持向后兼容性
class UNetWithSteger(SGVP):
    def __init__(self, nChannel, nClass):
        super().__init__(nChannel, nClass)

# 保留原始UNet类以保持兼容性
class UNet(nn.Module):
    def __init__(self, nChannel, nClass):
        super(UNet, self).__init__()
        self.inc = DoubleConv(nChannel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, nClass)

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
        return self.outc(x)
