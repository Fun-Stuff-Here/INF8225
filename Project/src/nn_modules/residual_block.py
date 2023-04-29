import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        output_channels: int,
        is_plain=False,
        device="cpu",
        dropout=0.5,
    ):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.is_plain = is_plain
        self.num_channels = num_channels
        self.output_channels = output_channels
        self.is_identity = num_channels == output_channels

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=output_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2) if not self.is_identity else (1, 1),
            device=self.device,
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
            device=self.device,
        )

        self.conv3 = None
        if not self.is_identity and not self.is_plain:
            self.conv3 = nn.Conv2d(
                in_channels=num_channels,
                out_channels=output_channels,
                kernel_size=(1, 1),
                stride=(2, 2),
                padding=(0, 0),
                device=self.device,
            )

        self.batch_normalization = nn.BatchNorm2d(
            num_features=output_channels,
            device=self.device,
        )
        self.reLu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Init residual
        y = x.clone()
        # Conv block 1
        x = self.conv1(x)
        x = self.batch_normalization(x)
        x = self.reLu(x)
        # Conv block 2
        x = self.conv2(x)
        x = self.batch_normalization(x)
        x = self.dropout(x)
        if not self.is_plain:
            if self.conv3 is not None:
                y = self.conv3(y)
            x = x + y

        return self.reLu(x)
