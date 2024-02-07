import torch
from torch import nn


class DepthwiseSeparableConvolution(nn.Module):
    """
    Depthwise Separable Convolution (DSC) block.

    Consists of a depthwise convolution (DC) followed by
    a pointwise convolution (PC).
    DC applies a single filter to each input channel, while PC combines the
    outputs using 1x1 convolutions.

    Args:
        inchannels (int): Number of input channels.
        outchannels (int): Number of output channels.
        nstride (int, optional): Stride for the
        depthwise convolution. Default: 1.
    """

    def __init__(self, inchannels: int, outchannels: int,
                 nstride: int = 1) -> None:
        # Depthwise Convolution
        super().__init__()
        self.DC = nn.Conv2d(in_channels=inchannels,
                            out_channels=inchannels,
                            groups=inchannels,
                            kernel_size=3,
                            stride=nstride, padding=1, bias=False)

        self.PC = nn.Conv2d(in_channels=inchannels,
                            out_channels=outchannels, kernel_size=1)

        self.BN1 = nn.BatchNorm2d(num_features=inchannels)
        self.BN2 = nn.BatchNorm2d(num_features=outchannels)
        self.RELU = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.DC(x)
        out = self.BN1(out)
        out = self.RELU(out)
        out = self.PC(out)
        out = self.BN2(out)
        return out
