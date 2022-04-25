from omegaconf import DictConfig

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Network(Protocol):
    def __init__(self, hparams: DictConfig, in_channels: int, n_classes: int):
        ...
