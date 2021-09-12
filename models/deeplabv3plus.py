import torch
import torch.nn as nn
import torch.nn.functional as F

from models.xception import Xception
from models.resnet import Resnet101


class ASPPblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ImagePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.adaptive_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.aspp_block_1 = ASPPblock(in_channels, out_channels, kernel_size=1, padding=0)
        self.aspp_block_2 = ASPPblock(in_channels, out_channels, dilation=6, padding=6)
        self.aspp_block_3 = ASPPblock(in_channels, out_channels, dilation=12, padding=12)
        self.aspp_block_4 = ASPPblock(in_channels, out_channels, dilation=18, padding=18)

        self.image_pooling = ImagePooling(in_channels, out_channels)

        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.conv_1 = nn.Conv2d(out_channels, out_channels, 1)  # added
        self.bn_1 = nn.BatchNorm2d(out_channels)  # added

    def forward(self, x):
        x1 = self.aspp_block_1(x)
        x2 = self.aspp_block_2(x)
        x3 = self.aspp_block_3(x)
        x4 = self.aspp_block_4(x)
        x5 = self.image_pooling(x)

        x5 = F.interpolate(x5, x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, in_channel):
        super().__init__()

        self.input_size = input_size

        self.conv_1 = nn.Conv2d(in_channel, 64, kernel_size=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv_2 = nn.Conv2d(320, 256, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(256)

        self.out_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv_1(low_level_feature)
        low_level_feature = self.bn_1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)

        x = F.interpolate(x, low_level_feature.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((low_level_feature, x), dim=1)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.out_conv(x)
        x = F.interpolate(x, self.input_size, mode='bilinear', align_corners=True)
        return x


class DeepLabV3plus(nn.Module):
    def __init__(self, backbone='xception', input_size=(320, 240)):
        super().__init__()

        in_channel = 256
        if backbone == 'resnet101':
            self.backbone = Resnet101()
        if backbone == 'resnet152':
            self.backbone = Resnet152()
        if backbone == 'xception':
            self.backbone = Xception()
            in_channel = 128

        self.aspp = ASPP(2048, 256)
        self.decoder = Decoder(input_size, in_channel=in_channel)

    def forward(self, x):
        x, low_level_feature = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        return x
