from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision as torch_vision
import torchvision.transforms as transforms
import torch.optim as optimizer
from nn_modules.res_net_model import ResNetModel


@dataclass
class Config:
    # General parameters
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    betas: tuple = (0.9, 0.99)
    clip: float = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Others
    loss = nn.CrossEntropyLoss()
    seed: int = 0
    log_every: int = 50
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # training sets and loader
    train_loader = torch.utils.data.DataLoader(
        torch_vision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    validation_loader = torch.utils.data.DataLoader(
        torch_vision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = torch.utils.data.DataLoader(
        torch_vision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    def __init__(self, *args, **kwargs) -> None:
        self.residual_blocks = args[0]
        self.model = ResNetModel(*args, **kwargs, device=self.device)
        self.model.to(self.device)
        self.optimizer = optimizer.Adam(
            self.model.parameters(), lr=self.lr, betas=self.betas
        )
        torch.manual_seed(self.seed)
