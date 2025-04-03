import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    U-Net Generator architecture with skip connections for image-to-image translation tasks.
    """
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = self.conv2Relu(1, 32, kernel_size=3, stride=1)
        self.enc2 = self.conv2Relu(32, 64, kernel_size=3, stride=2)
        self.enc3 = self.conv2Relu(64, 128, kernel_size=3, stride=2)
        self.enc4 = self.conv2Relu(128, 256, kernel_size=3, stride=2)
        self.enc5 = self.conv2Relu(256, 512, kernel_size=3, stride=2)

        # Decoder
        self.dec1 = self.deconv2Relu(512, 256)
        self.dec2 = self.deconv2Relu(256 + 256, 128)
        self.dec3 = self.deconv2Relu(128 + 128, 64)
        self.dec4 = self.deconv2Relu(64 + 64, 32)
        self.dec5 = nn.Sequential(nn.Conv2d(32 + 32, 1, kernel_size=3, padding=1), nn.Tanh())

    def conv2Relu(self, in_c, out_c, kernel_size=3, stride=1):
        """
        Convolution -> BatchNorm -> LeakyReLU
        """
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        return nn.Sequential(*layers)

    def deconv2Relu(self, in_c, out_c, kernel_size=3):
        """
        Transposed Convolution -> BatchNorm -> ReLU
        """
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoding with skip connections
        out = self.dec1(x5)
        out = self.dec2(torch.cat((out, x4), dim=1))
        out = self.dec3(torch.cat((out, x3), dim=1))
        out = self.dec4(torch.cat((out, x2), dim=1))
        out = self.dec5(torch.cat((out, x1), dim=1))
        return out

class Discriminator(nn.Module):
    """
    Discriminator architecture for adversarial training in image-to-image translation tasks.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = self.conv2relu(2, 16, kernel_size=5, stride=1)
        self.layer2 = self.conv2relu(16, 32, kernel_size=3, stride=2)
        self.layer3 = self.conv2relu(32, 64, kernel_size=3, stride=2)
        self.layer4 = self.conv2relu(64, 128, kernel_size=3, stride=2)
        self.layer5 = self.conv2relu(128, 256, kernel_size=3, stride=2)
        self.layer6 = nn.Conv2d(256, 1, kernel_size=1)

    def conv2relu(self, in_c, out_c, kernel_size=3, stride=1):
        """
        Convolution -> BatchNorm -> LeakyReLU
        """
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x, x1):
        x = torch.cat((x, x1), dim=1)
        out = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return self.layer6(out)

class ArtNet(nn.Module):
    """
    ArtNet architecture for conditional adversarial tasks.
    """
    def __init__(self, n_in=2, n_out=64):
        super(ArtNet, self).__init__()
        self.kernel_size = 4
        self.padding = 1

        self.model = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.LeakyReLU(0.2, True),
            self._get_layer(n_out, 2*n_out, 2),
            self._get_layer(2*n_out, 4*n_out, 2),
            self._get_layer(4*n_out, 8*n_out, 1),
            nn.Conv2d(8 * n_out, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        )

    def forward(self, input_1, input_2):
        input = torch.cat([input_1, input_2], 1)
        return self.model(input)

    def _get_layer(self, n_input_channels, n_output_channels, stride):
        """
        Helper method to create a convolutional block with batch normalization and LeakyReLU.
        """
        return nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm2d(n_output_channels),
            nn.LeakyReLU(0.2, True)
        )

class EncodeModule(nn.Module):
    """
    Encoder module for Pavel's Generator with optional batch normalization.
    """
    def __init__(self, in_c, out_c, batchnorm=True):
        super(EncodeModule, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv', nn.Conv2d(in_c, out_c, 4, stride=2, padding=1))
        if batchnorm:
            self.layers.add_module('bn', nn.BatchNorm2d(out_c))
        self.layers.add_module('relu', nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.layers(x)

class DecodeModule(nn.Module):
    """
    Decoder module for Pavel's Generator with optional batch normalization and dropout.
    """
    def __init__(self, in_c, out_c, batchnorm=True, dropout=False):
        super(DecodeModule, self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, stride=2)
        self.layers = nn.Sequential()
        if batchnorm:
            self.layers.add_module('bn', nn.BatchNorm2d(out_c*2))
        if dropout:
            self.layers.add_module('do', nn.Dropout2d(p=0.5, inplace=True))
        self.layers.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dw = x2.size(2) - x1.size(2)
        dh = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.layers(x)

class PavelNet(nn.Module):
    """
    PavelNet: A deeper U-Net architecture with 8 encoder and decoder layers.
    Originally from: https://github.com/sigtot/pix2pix-model/blob/master/unet
    """
    def __init__(self, in_c=1, out_c=1):
        super(PavelNet, self).__init__()
        # Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        self.e1 = EncodeModule(in_c, 64, batchnorm=False)  # 256 -> 128
        self.e2 = EncodeModule(64, 128)  # 128 -> 64
        self.e3 = EncodeModule(128, 256)  # 64 -> 32
        self.e4 = EncodeModule(256, 512)  # 32 -> 16
        self.e5 = EncodeModule(512, 512)  # 16 -> 8
        self.e6 = EncodeModule(512, 512)  # 8 -> 4
        self.e7 = EncodeModule(512, 512)  # 4 -> 2
        self.e8 = EncodeModule(512, 512, batchnorm=False)  # 2 -> 1

        # Decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        self.d1 = DecodeModule(512, 512, dropout=True)
        self.d2 = DecodeModule(1024, 512, dropout=True)
        self.d3 = DecodeModule(1024, 512, dropout=True)
        self.d4 = DecodeModule(1024, 512)
        self.d5 = DecodeModule(1024, 256)
        self.d6 = DecodeModule(512, 128)
        self.d7 = DecodeModule(256, 64)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_c, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder path
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6)
        x8 = self.e8(x7)

        # Decoder path with skip connections
        y = self.d1(x8, x7)
        y = self.d2(y, x6)
        y = self.d3(y, x5)
        y = self.d4(y, x4)
        y = self.d5(y, x3)
        y = self.d6(y, x2)
        y = self.d7(y, x1)
        return self.out(y)
    
if __name__=="__main__":
    # Simple test to verify the models
    G = PavelNet()
    D = ArtNet()
    print(G)
    print(D)
    x = torch.randn(32, 1, 256, 256)
    print(f"Generator output shape: {G(x).shape}")
    print(f"Discriminator output shape: {D(x,x).shape}")
