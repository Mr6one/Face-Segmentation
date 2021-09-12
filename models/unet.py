import torch
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = Convolution(2 * out_channels, out_channels)

    def forward(self, x, previous_layer_x):
        x = self.up_conv(x)
        x = torch.cat((previous_layer_x, x), dim=1)
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.conv = Convolution(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.max_pool(x))


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = Convolution(3, 64)
        self.downsample_1 = DownSample(64, 128)
        self.downsample_2 = DownSample(128, 256)
        self.downsample_3 = DownSample(256, 512)
        self.downsample_4 = DownSample(512, 1024)

        self.upsample_1 = UpSample(1024, 512)
        self.upsample_2 = UpSample(512, 256)
        self.upsample_3 = UpSample(256, 128)
        self.upsample_4 = UpSample(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        output_1 = self.conv(x)
        output_2 = self.downsample_1(output_1)
        output_3 = self.downsample_2(output_2)
        output_4 = self.downsample_3(output_3)
        x = self.downsample_4(output_4)

        x = self.upsample_1(x, output_4)
        x = self.upsample_2(x, output_3)
        x = self.upsample_3(x, output_2)
        x = self.upsample_4(x, output_1)

        x = self.out_conv(x)
        return x
