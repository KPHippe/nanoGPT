import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from nanogpt.model import GPTConfig, GPT


from dataclasses import dataclass


@dataclass
class LightningGPTConfig(GPTConfig):
    weight_decay: float = 0.0
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.95)
    device_type: str = "cuda"


# Setup nanoGPT as lightning model
class LightningNanoGPT(LightningModule):
    def __init__(self, config: LightningGPTConfig, verbose=True) -> None:
        super().__init__()
        self.config = config
        self.verbose = verbose

        self.model = GPT(config)

    def training_step(self, batch, batch_idx, **kwargs):
        idx, targets = batch
        _, loss = self.model(idx, targets=targets)
        self.log("train_loss", loss, prog_bar=self.verbose)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        idx, targets = batch
        _, loss = self.model(idx, targets=targets)
        self.log("val_loss", loss, prog_bar=self.verbose)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=self.config.betas,
            device_type=self.config.device_type,
        )
