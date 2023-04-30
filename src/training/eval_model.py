from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..config import Config
from .metric import Metric


def eval_model(model: nn.Module, data_loader: DataLoader, config: Config) -> float:
    """
    Evaluate the model on the given dataloader.
    @return: the accuracy of the model on the given dataloader.
    """
    device = config.device
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
