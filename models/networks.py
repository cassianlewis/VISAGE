import torch
import torch.nn as nn
import torch.nn.parallel
from models.blurpool import BlurPool, BlurTranspose


def normal_init(m: nn.Module, mean: float, std: float) -> None:
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class MaxBlurPool(nn.Module):
    """Max pooling layer with blurring."""
    def __init__(self, in_channels: int):
        super(MaxBlurPool, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=2, stride=1)
        self.blur = BlurPool(in_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.max(x)
        x2 = self.blur(x1)
        return x2


class DownLayer(nn.Module):
    """Downsampling layer with convolutional and blurring operations."""
    def __init__(self, in_channels: int, out_channels: int, batch_norm1: bool = True, batch_norm2: bool = True):
        super(DownLayer, self).__init__()
        self.maxblur = MaxBlurPool(in_channels)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm1 else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm2 else nn.Identity(),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.maxblur(x)
        x2 = self.down(x1)
        return x2


class UpLayer(nn.Module):
    """Upsampling layer with convolutional and blurring operations."""
    def __init__(self, in_channels: int, out_channels: int, batch_norm1: bool = True, batch_norm2: bool = True):
        super(UpLayer, self).__init__()
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.blur = BlurTranspose(in_channels, stride=2)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels) if batch_norm1 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm2 else nn.Identity(),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.convo(x)
        x2 = self.blur(x1)
        x3 = self.up(x2)
        return x3



class ConvBlurPool(nn.Module):
    """Convolutional layer with blurring and pooling. For use in the discriminator network."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, filt_size: int = 4):
        super(ConvBlurPool, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.blur_pool = BlurPool(out_channels, filt_size=filt_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.blur_pool(x)
        return x



class Generator(nn.Module):
    """Generator network with U-Net architecture."""
    def __init__(self):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024, batch_norm2=False)

        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512 + 512, 256)
        self.up3 = UpLayer(256 + 256, 128)
        self.up4 = UpLayer(128 + 128, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.initial(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u1 = torch.cat((u1, d3), dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat((u2, d2), dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat((u3, d1), dim=1)
        u4 = self.up4(u3)

        out = self.tanh(self.final(u4))
        return out



class Discriminator(nn.Module):
    """PatchGAN discriminator with a receptive field of 70x70"""
    def __init__(self, in_channels: int = 4, base_filters: int = 64):
        super(Discriminator, self).__init__()
        self.layer1 = ConvBlurPool(in_channels, base_filters, kernel_size=4, stride=2)
        self.layer2 = ConvBlurPool(base_filters, base_filters * 2, kernel_size=4, stride=2)
        self.layer3 = ConvBlurPool(base_filters * 2, base_filters * 4, kernel_size=4, stride=2)
        self.final = nn.Conv2d(base_filters * 4, 1, kernel_size=4, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.sigmoid(self.final(x3))
        return x4
