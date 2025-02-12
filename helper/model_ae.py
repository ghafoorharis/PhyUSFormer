import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Autoencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.enc1 = DoubleConv(n_channels, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)
        self.enc4 = Down(128, 256)
        self.enc5 = Down(256, 512)

        # Decoder: Match input channels with encoder output channels
        self.dec1 = Up(512, None, 256, bilinear=False)  # No skip connections
        self.dec2 = Up(256, None, 128, bilinear=False)
        self.dec3 = Up(128, None, 64, bilinear=False)
        self.dec4 = Up(64, None, 32, bilinear=False)

        # Output layer
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoder path
        x = self.dec1(x5, None)  # No skip connection
        x = self.dec2(x, None)
        x = self.dec3(x, None)
        x = self.dec4(x, None)

        # Output layer
        logits = self.outc(x)

        return nn.Sigmoid()(logits)  # Activation applied outside the model


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
                in_channels_decoder, out_channels, kernel_size=2, stride=2
            )
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        return self.conv(x1)  # No concatenation, as this is a simple autoencoder

if __name__ == "__main__":
    model = Autoencoder(n_channels=1, n_classes=1)
    x = torch.randn((32, 1, 256, 256))  # Batch size 32, 1 channel, 256x256
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
