import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, stride=1, dilation=1, padding=1,
                 skip_transform=False):
        super().__init__()

        self.skip_transform = skip_transform
        if skip_transform:
            self.conv_0 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation)
            self.bn_0 = nn.BatchNorm2d(out_channels)

        self.conv_1 = nn.Conv2d(in_channels, middle_channels, 1, dilation=dilation)
        self.bn_1 = nn.BatchNorm2d(middle_channels)

        self.conv_2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=padding, stride=stride, dilation=dilation)
        self.bn_2 = nn.BatchNorm2d(middle_channels)

        self.conv_3 = nn.Conv2d(middle_channels, out_channels, 1, dilation=dilation)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        skip = x

        if self.skip_transform:
            skip = self.conv_0(skip)
            skip = self.bn_0(skip)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.bn_3(x)

        x += skip

        x = self.relu(x)
        return x


class Resnet(nn.Module):
    def __init__(self, block_sizes):
        super().__init__()

        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        layer_1 = []
        layer_1.append(Bottleneck(64, 64, 256, skip_transform=True))
        for i in range(1, block_sizes[0]):
            layer_1.append(Bottleneck(256, 64, 256))

        self.layer_1 = nn.Sequential(*layer_1)

        layer_2 = []
        layer_2.append(Bottleneck(256, 128, 512, stride=2, skip_transform=True))
        for i in range(1, block_sizes[1]):
            layer_2.append(Bottleneck(512, 128, 512))

        self.layer_2 = nn.Sequential(*layer_2)

        layer_3 = []
        layer_3.append(Bottleneck(512, 256, 1024, stride=2, skip_transform=True))
        for i in range(1, block_sizes[2]):
            layer_3.append(Bottleneck(1024, 256, 1024))

        self.layer_3 = nn.Sequential(*layer_3)

        layer_4 = []
        layer_4.append(Bottleneck(1024, 512, 2048, stride=2, dilation=2, padding=2, skip_transform=True))
        for i in range(1, block_sizes[3]):
            layer_4.append(Bottleneck(2048, 512, 2048, dilation=2, padding=2))

        self.layer_4 = nn.Sequential(*layer_4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer_1(x)
        low_level_feature = x

        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x, low_level_feature


class Resnet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = Resnet([3, 4, 23, 3])

    def forward(self, x):
        x, low_level_feature = self.resnet(x)
        return x, low_level_feature


class Resnet152(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = Resnet([3, 8, 36, 3])

    def forward(self, x):
        x, low_level_feature = self.resnet(x)
        return x, low_level_feature
