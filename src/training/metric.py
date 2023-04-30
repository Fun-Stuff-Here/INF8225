from dataclasses import dataclass


@dataclass
class Metric:
    loss: float
    training_accuracy: float
