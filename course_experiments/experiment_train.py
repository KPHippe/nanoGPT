import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything

from experiment_model import LightningNanoGPT, LightningGPTConfig

# TODO: setup profiling + experiment logging (probably through callbacks?)
# TODO: setup LR scheduler

seed_everything(42)

# Optimizations
torch.set_float32_matmul_precision("high")


# Define model with defaults from char shakespeare model
vocab_size = 65
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
compile = False  # Doesn't work too well with PL it appears?

weight_decay = 1e-1
learning_rate = 1e-3
betas = (0.9, 0.99)
device_type = "cuda"
device = "cuda"

dset_root = "../data"
dataset = "shakespeare_char"
batch_size = 64

max_steps = 5000

config = LightningGPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=betas,
    device_type=device_type,
)

model = LightningNanoGPT(config)
if compile:
    print("Compiling model")
    model.model = torch.compile(model.model)

# Dataset
class CharDataset(Dataset):
    def __init__(self, split: str = "train", block_size: int = 256) -> None:
        super().__init__()
        self.split = split
        self.data = np.memmap(
            os.path.join(dset_root, dataset, f"{self.split}.bin"),
            dtype=np.uint16,
            mode="r",
        )

    def __len__(self) -> int:
        return max_steps * batch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Roughly equivalent to original training, instead of getting a full batch,
        # we get a single example of size `block_size`
        ix = np.random.randint(len(self.data) - block_size)

        x = torch.tensor(
            self.data[ix : ix + block_size].astype(np.int16), dtype=torch.long
        )
        y = torch.tensor(
            self.data[ix + 1 : ix + block_size + 1].astype(np.int16), dtype=torch.long
        )

        return x, y


train_dset = CharDataset(split="train")
val_dset = CharDataset(split="val")

train_dloader = DataLoader(
    train_dset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)
val_dloader = DataLoader(
    val_dset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)


# Define trainer and train
trainer = Trainer(
    max_steps=max_steps,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    val_check_interval=100,
    check_val_every_n_epoch=None,
    limit_val_batches=15,
    num_sanity_val_steps=0,  # Disable validation on startup
)
trainer.fit(model, train_dataloaders=train_dloader, val_dataloaders=val_dloader)
