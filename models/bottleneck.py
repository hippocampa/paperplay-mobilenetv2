import torch
from torch import nn
from .dsc import DepthwiseSeparableConvolution as DSC


class BottleneckBlock(nn.Module):
    """
    Bottleneck block, a core component of MobileNetV2 architecture.

    Reduces computations while maintaining representational power through
    linear bottlenecks and inverted residuals.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        nstrides (int): Number of strides for the first 1x1 convolution.
        exp_factor (int): Expansion factor for the number of channels in the
        intermediate layers.
    """

    def __init__(self, in_c: int, out_c: int,
                 exp_factor:int,
                 nstrides: int=1) -> None:
        super().__init__()
        expanded_ch = in_c*exp_factor
        self.nstrides = nstrides
        # TODO: Use ordered dicts
        self.blayers = nn.Sequential(
            nn.Conv2d(in_channels=in_c,
                      out_channels=expanded_ch,
                      stride=self.nstrides, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_c * exp_factor),
            nn.ReLU6(inplace=True),
            DSC(inchannels=expanded_ch,
                outchannels=expanded_ch),
            nn.Conv2d(in_channels=expanded_ch,
                      out_channels=out_c, kernel_size=1),
        )
        self.use_skip_connection = self.nstrides == 1 and in_c == out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        out = self.blayers(x)
        if self.use_skip_connection:
            out+=x
        return out
