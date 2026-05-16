import torch
import torch.nn as nn
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ..base import SegmentationModel, SegmentationHead, ClassificationHead
from ..base import get_encoder
from ..deeplabv3.deeplabv3_model import DeepLabV3Plus
from ..deeplabv3.deeplabv3_decoder import DeepLabV3PlusDecoder
from unet.model import SGVP

class DBTSNet(SegmentationModel):
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
        vessel_unet_channels: int = 1,
    ):
        super().__init__()
        
        # 1. Initialize SGVP for vessel segmentation
        self.vessel_unet = SGVP(nChannel=in_channels, nClass=vessel_unet_channels)
        
        # 2. Initialize DeepLabV3+  for thrombus segmentation
        self.stsnet = DeepLabV3Plus(
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
        
        # 3. Add vessel attention fusion module
        self.vessel_attention_fusion = nn.Sequential(
            nn.Conv2d(decoder_channels + vessel_unet_channels, decoder_channels, kernel_size=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4. Replace the segmentation head to accept fused features
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
    
    def forward(self, x):
        # 1. Generate vessel features using SGVP
        vessel_features = self.vessel_unet(x)
        
        # 2. Get STSNet features
        stsnet_features = self.stsnet.encoder(x)
        decoder_output = self.stsnet.decoder(*stsnet_features)
        
        # 3. Upsample vessel features to match decoder output size
        vessel_features_upsampled = nn.functional.interpolate(
            vessel_features,
            size=decoder_output.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 4. Concatenate and fuse vessel features with STSNet decoder output
        fused_features = torch.cat([decoder_output, vessel_features_upsampled], dim=1)
        fused_features = self.vessel_attention_fusion(fused_features)
        
        # 5. Generate final segmentation mask
        masks = self.segmentation_head(fused_features)
        
        if self.stsnet.classification_head is not None:
            labels = self.stsnet.classification_head(stsnet_features[-1])
            return masks, labels
        
        return masks



# 另外，我们可以修改注意力模块，直接将血管特征作为注意力权重
class VesselAttentionModule(nn.Module):
    """Vessel Attention Module: Uses vessel features to weight the original features"""
    
    def __init__(self, in_channels):
        super().__init__()
        # 1x1 convolution to adjust vessel features channel-wise
        self.vessel_adjust = nn.Conv2d(1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, vessel_features):
        # Upsample vessel features to match input size
        vessel_features_upsampled = nn.functional.interpolate(
            vessel_features,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Adjust vessel features to match input channels
        vessel_weights = self.vessel_adjust(vessel_features_upsampled)
        vessel_weights = self.sigmoid(vessel_weights)
        
        # Apply vessel attention to input features
        weighted_features = x * vessel_weights
        
        return weighted_features

