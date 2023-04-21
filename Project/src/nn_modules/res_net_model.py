import torch
import torch.nn as nn
from .residual_block import ResidualBlock


class ResNetModel(nn.Module):
    def __init__(self, *arg, **kwargs):
        super(ResNetModel, self).__init__()
        self.device = kwargs["device"]
        residual_blocks_kwargs = arg[0]
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=1, device=self.device
        )
        self.batch_norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.reLu = nn.ReLU()
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    **residual_block_kwargs,
                    device=self.device,
                    is_plain=kwargs["is_plain"]
                )
                for residual_block_kwargs in residual_blocks_kwargs
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.Flatten()

        self.fc = nn.Linear(
            in_features=self.residual_blocks[-1].output_channels,
            out_features=kwargs["nb_classes"],
            device=self.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.reLu(x)
        x = self.max_pool(x)
        x = self.residual_blocks(x)
        x = self.avg_pool(x)
        x = self.gap(x)
        x = self.fc(x)
        return x
