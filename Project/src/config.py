from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torchvision as torch_vision
import torchvision.transforms as transforms
import torch.optim as optimizer
from nn_modules.res_net_model import ResNetModel


@dataclass
class Config:
    # General parameters
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    betas: tuple = (0.9, 0.99)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Others
    loss = nn.CrossEntropyLoss()
    seed: int = 0
    log_every: int = 50
    classes: tuple[str] = (
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
        self.is_plain = kwargs["is_plain"]
        self.optimizer = optimizer.Adam(
            self.model.parameters(), lr=self.lr, betas=self.betas
        )
        self.n_layers = len(self.residual_blocks) * 2 + 2
        torch.manual_seed(self.seed)

    @property
    def values(self) -> dict:
        config_dict = asdict(self)
        config_dict["n_layers"] = self.n_layers
        config_dict["model"] = str(self.model)
        config_dict["residual_blocks"] = [str(block) for block in self.residual_blocks]
        config_dict["is_plain"] = self.is_plain
        return config_dict
