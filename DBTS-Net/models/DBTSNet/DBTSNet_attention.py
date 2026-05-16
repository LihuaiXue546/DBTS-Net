import torch
import torch.nn as nn
from ..deeplabv3.deeplabv3_model import DeepLabV3Plus
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from unet.model import SGVP

class DBTSNet(DeepLabV3Plus):
    """DBTS-Net: Dual-Branch Thrombus Segmentation Network with Vascular Attention"""
    
    def __init__(
        self,
        encoder_name: str = "resnext101_32x4d",
        encoder_weights: str = "imagenet",
        encoder_depth: int = 5,
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: str = None,
        upsampling: int = 4,
        aux_params: dict = None,
    ):
        # 初始化父类DeepLabV3Plus
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )
        
        # 1. 初始化SGVP用于血管分割
        # 注意：SGVP期望输入是2通道，但我们这里使用3通道RGB图像
        # 我们需要修改SGVP或者创建一个适配器
        self.vessel_unet = SGVP(nChannel=2, nClass=1)
        
        # 2. 添加输入适配器，将3通道RGB图像转换为2通道输入
        self.input_adapter = nn.Conv2d(3, 2, kernel_size=1)
        
        # 3. 添加血管注意力模块
        # 这个模块将血管特征图转换为与decoder输出匹配的注意力权重
        self.vessel_attention = nn.Sequential(
            nn.Conv2d(1, decoder_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 1. 生成血管特征图 (SGVP branch)
        # 将输入转换为2通道
        adapted_input = self.input_adapter(x)
        vessel_features = self.vessel_unet(adapted_input)
        
        # 2. 获取原始DeepLabV3Plus的特征
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(*encoder_features)
        
        # 3. 调整血管特征图的大小以匹配decoder输出
        vessel_features_upsampled = nn.functional.interpolate(
            vessel_features,
            size=decoder_output.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 4. 生成血管注意力权重
        vessel_weights = self.vessel_attention(vessel_features_upsampled)
        
        # 5. 将血管注意力权重应用到decoder输出
        # 这里使用乘法操作，增强血管区域的特征
        weighted_decoder_output = decoder_output * vessel_weights
        
        # 6. 生成最终分割掩码
        masks = self.segmentation_head(weighted_decoder_output)
        
        if self.classification_head is not None:
            labels = self.classification_head(encoder_features[-1])
            return masks, labels
        
        return masks



