import torch
import torch.nn as nn
from .residual_block import ResidualBlock


class ResNetModel(nn.Module):
    def __init__(self, *arg, **kwargs):
        super(ResNetModel, self).__init__()
        self.device = kwargs["device"]
        residual_blocks_kwargs = arg[0]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            device=self.device,
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=16,
            device=self.device,
        )
        self.reLu = nn.ReLU()
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    **residual_block_kwargs,
                    device=self.device,
                    is_plain=kwargs["is_plain"],
                )
                for residual_block_kwargs in residual_blocks_kwargs
            ]
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(4, 4),
            stride=(1, 1),
        )

        self.fc = nn.Linear(
            in_features=self.residual_blocks[-1].output_channels,
            out_features=kwargs["nb_classes"],
            device=self.device,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.reLu(x)
        x = self.residual_blocks(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
