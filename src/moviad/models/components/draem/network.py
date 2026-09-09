"""Sub-networks of the DRAEM model.

DRAEM is composed of two networks trained jointly:

* a *reconstructive* sub-network, an encoder-decoder that is trained to restore
  the anomaly-free appearance of a synthetically corrupted image;
* a *discriminative* sub-network, a U-Net that receives the channel-wise
  concatenation of the corrupted image and of its reconstruction and outputs the
  anomaly segmentation logits.

Reference:
    "DRAEM - A discriminatively trained reconstruction embedding for surface
    anomaly detection", Zavrtanik et al., ICCV 2021.
    https://arxiv.org/abs/2108.07610
"""

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Bilinear upsampling followed by a 3x3 convolution."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class EncoderReconstructive(nn.Module):
    """Encoder of the reconstructive sub-network (4 downsampling stages)."""

    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        self.block1 = _conv_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = _conv_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = _conv_block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = _conv_block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = _conv_block(base_width * 8, base_width * 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        b5 = self.block5(self.mp4(b4))
        return b5


class DecoderReconstructive(nn.Module):
    """Decoder of the reconstructive sub-network (no skip connections)."""

    def __init__(self, base_width: int, out_channels: int):
        super().__init__()
        self.up1 = _up_block(base_width * 8, base_width * 8)
        self.db1 = _conv_block(base_width * 8, base_width * 8)
        self.up2 = _up_block(base_width * 8, base_width * 4)
        self.db2 = _conv_block(base_width * 4, base_width * 4)
        self.up3 = _up_block(base_width * 4, base_width * 2)
        self.db3 = _conv_block(base_width * 2, base_width * 2)
        self.up4 = _up_block(base_width * 2, base_width)
        self.db4 = _conv_block(base_width, base_width)
        self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5: torch.Tensor) -> torch.Tensor:
        db1 = self.db1(self.up1(b5))
        db2 = self.db2(self.up2(db1))
        db3 = self.db3(self.up3(db2))
        db4 = self.db4(self.up4(db3))
        return self.fin_out(db4)


class ReconstructiveSubNetwork(nn.Module):
    """Encoder-decoder restoring the anomaly-free appearance of an image.

    Args:
        in_channels (int): number of channels of the input image.
        out_channels (int): number of channels of the reconstructed image.
        base_width (int): number of channels of the first convolutional stage.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 128):
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class EncoderDiscriminative(nn.Module):
    """Encoder of the discriminative U-Net (5 downsampling stages)."""

    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        self.block1 = _conv_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = _conv_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = _conv_block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = _conv_block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = _conv_block(base_width * 8, base_width * 8)
        self.mp5 = nn.MaxPool2d(2)
        self.block6 = _conv_block(base_width * 8, base_width * 8)

    def forward(self, x: torch.Tensor):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        b5 = self.block5(self.mp4(b4))
        b6 = self.block6(self.mp5(b5))
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    """Decoder of the discriminative U-Net (with skip connections)."""

    def __init__(self, base_width: int, out_channels: int):
        super().__init__()
        self.up_b = _up_block(base_width * 8, base_width * 8)
        self.db_b = _conv_block(base_width * (8 + 8), base_width * 8)
        self.up1 = _up_block(base_width * 8, base_width * 4)
        self.db1 = _conv_block(base_width * (4 + 8), base_width * 4)
        self.up2 = _up_block(base_width * 4, base_width * 2)
        self.db2 = _conv_block(base_width * (2 + 4), base_width * 2)
        self.up3 = _up_block(base_width * 2, base_width)
        self.db3 = _conv_block(base_width * (1 + 2), base_width)
        self.up4 = _up_block(base_width, base_width)
        self.db4 = _conv_block(base_width * 2, base_width)
        self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b1, b2, b3, b4, b5, b6) -> torch.Tensor:
        db_b = self.db_b(torch.cat((self.up_b(b6), b5), dim=1))
        db1 = self.db1(torch.cat((self.up1(db_b), b4), dim=1))
        db2 = self.db2(torch.cat((self.up2(db1), b3), dim=1))
        db3 = self.db3(torch.cat((self.up3(db2), b2), dim=1))
        db4 = self.db4(torch.cat((self.up4(db3), b1), dim=1))
        return self.fin_out(db4)


class DiscriminativeSubNetwork(nn.Module):
    """U-Net predicting the anomaly segmentation logits.

    Args:
        in_channels (int): channels of the input, i.e. the concatenation of the
            (possibly corrupted) image and of its reconstruction.
        out_channels (int): number of segmentation logits, 2 in DRAEM (normal
            and anomalous).
        base_width (int): number of channels of the first convolutional stage.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 2, base_width: int = 64):
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_segment(*self.encoder_segment(x))
