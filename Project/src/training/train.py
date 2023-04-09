from ..config import Config
import torch
import torch.nn as nn
import wandb
from collections import defaultdict
from dataclasses import asdict
import numpy as np
from .metric import Metric
from .eval_model import eval_model
from ..logging.logging import print_logs


def train_model(model: nn.Module, config: Config) -> None:
    """Train the model."""
    train_loader, val_loader = config.train_loader, config.validation_loader
    optimizer = config.optimizer
    clip = config.clip
    device = config.device
    epochs = config.epochs

    print(f"Starting training for {epochs} epochs, using {device}.")
    for e in range(epochs):
        print(f"\nEpoch {e+1}")

        model.to(device)
        model.train()
        logs = defaultdict(list)
        train_accuracy = 0.0

        logs["validation_accuracy"] = [eval_model(model, val_loader, config)]
        for batch_id, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            train_accuracy += (torch.argmax(outputs, axis=1) == labels).float().mean()
            loss = config.loss(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            metrics = Metric(
                loss=loss.item(),
                training_accuracy=train_accuracy.item() / (batch_id + 1),
            )

            for name, value in asdict(metrics).items():
                logs[name].append(value)

            if batch_id % config.log_every == 0:
                for name, value in logs.items():
                    logs[name] = np.mean(value)

                train_logs = {f"{m}": v for m, v in logs.items()}
                wandb.log(train_logs)
                logs = defaultdict(list)

        if len(logs) != 0:
            for name, value in logs.items():
                logs[name] = np.mean(value)
            train_logs = {f"{m}": v for m, v in logs.items()}
        else:
            logs = {m.split(" - ")[1]: v for m, v in train_logs.items()}

        print_logs("Train", logs)
