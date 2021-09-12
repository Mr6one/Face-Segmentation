import torch
import torch.nn as nn


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.stride = stride
        if stride == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
            self.bn_0 = nn.BatchNorm2d(out_channels)

        self.sep_conv_1 = SeparableConv(in_channels, out_channels)
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.sep_conv_2 = SeparableConv(out_channels, out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.sep_conv_3 = SeparableConv(out_channels, out_channels, stride=stride)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.stride == 2:
            skip = self.bn_0(self.conv(x))
        else:
            skip = x

        x = self.relu(self.bn_1(self.sep_conv_1(x)))
        x = self.relu(self.bn_2(self.sep_conv_2(x)))
        x = self.relu(self.bn_3(self.sep_conv_3(x)))

        x += skip

        return x


class EntryFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)

        self.sep_conv_block_1 = SepConvBlock(64, 128)
        self.sep_conv_block_2 = SepConvBlock(128, 256)
        self.sep_conv_block_3 = SepConvBlock(256, 728)

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))

        x = self.sep_conv_block_1(x)
        low_level_feature = x
        x = self.sep_conv_block_2(x)
        x = self.sep_conv_block_3(x)
        return x, low_level_feature


class MiddleFlow(nn.Module):
    def __init__(self):
        super().__init__()

        flow = []
        for i in range(16):
            flow.append(SepConvBlock(728, 728, stride=1))

        self.conv = nn.Sequential(*flow)

    def forward(self, x):
        x = self.conv(x)
        return x


class ExitFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.sep_conv_block = SepConvBlock(728, 1024)
        self.sep_conv_1 = SeparableConv(1024, 1536)
        self.sep_conv_2 = SeparableConv(1536, 1536)
        self.sep_conv_3 = SeparableConv(1536, 2048)

    def forward(self, x):
        x = self.sep_conv_block(x)
        x = self.sep_conv_1(x)
        x = self.sep_conv_2(x)
        x = self.sep_conv_3(x)
        return x


class Xception(nn.Module):
    def __init__(self):
        super().__init__()

        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()

    def forward(self, x):
        x, low_level_feature = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, low_level_feature
