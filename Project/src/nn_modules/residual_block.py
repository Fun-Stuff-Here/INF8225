import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        output_channels: int,
        is_plain=False,
        stride=2,
        device="cpu",
    ):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.is_plain = is_plain
        self.num_channels = num_channels
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            device=self.device,
        )
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, device=self.device
        )

        self.conv3 = None
        if self.num_channels != self.output_channels and not self.is_plain:
            self.conv3 = nn.Conv2d(
                num_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                device=self.device,
            )

        self.conv4 = nn.Conv2d(
            num_channels,
            output_channels,
            padding=0,
            kernel_size=1,
            stride=stride,
            device=self.device,
        )
        self.bn = nn.BatchNorm2d(num_channels)
        self.reLu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Init residual
        if not self.is_plain:
            y = x
            if self.conv3 is not None:
                y = self.conv3(x)
        # Conv block 1
        x = self.conv1(x)
        x = self.bn(x)
        x = self.reLu(x)
        # Conv block 2
        x = self.conv2(x)
        x = self.bn(x)
        if self.is_plain:
            x = self.conv4(x)
            return self.reLu(x)

        # Add residual to output of stacked
        if y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1], :, :]
        x += y

        # Last conv block
        x = self.conv4(x)

        return self.reLu(x)
