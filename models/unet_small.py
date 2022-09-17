from torch import nn
import torch
from torchvision import transforms as t


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.double_conv(x)


class DownScaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpScaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        h, w = x1.size(-2), x1.size(-1)
        x2 = t.CenterCrop((h, w))(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(self, config):
        super(UNetSmall, self).__init__()
        self.in_channels = config.in_channels
        self.num_classes = config.num_classes

        self.convs1 = DoubleConv(self.in_channels, 64)
        self.down1 = DownScaling(64, 128)
        self.down2 = DownScaling(128, 256)
        self.up3 = UpScaling(256, 128)
        self.up4 = UpScaling(128, 64)
        self.conv = LastConv(64, self.num_classes)

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up3(x3, x2)
        x5 = self.up4(x4, x1)
        out = self.conv(x5)
        out = torch.sigmoid(out)
        return out


def get_model(cfg):
    model = UNetSmall(cfg)
    return model.cuda()
