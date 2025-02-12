import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 32)  # Initial convolution block
        self.down1 = Down(32, 64)  # Downscaling layers
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)
        self.up5 = Up(64, 32, 32)

        # Output layer
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

        # Optional: Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # Decoder path with skip connections
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        # Output layer
        logits = self.outc(x)

        return nn.Sigmoid()(logits)  # Activation (e.g., Sigmoid) should be applied outside the model

class DoubleConv(nn.Module):
    """(Convolution -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels_decoder, in_channels_encoder, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels_decoder, in_channels_decoder // 2, kernel_size=2, stride=2
            )
        # DoubleConv with concatenated channels
        self.conv = DoubleConv(in_channels_decoder + in_channels_encoder, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust dimensions by padding if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=1)
    print(model)
    x = torch.randn((32, 1, 256, 256))
    print(model(x).shape)  # Expected output: torch.Size([1, 1, 388, 388])