import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.profilers import PyTorchProfiler

from experiment_model import LightningNanoGPT, LightningGPTConfig


seed_everything(42)

# Optimizations
torch.set_float32_matmul_precision("high")

"""
Order
------
1. precision (fp32, bf16) hold optimizer constant (adamW)
2. optimizer (adamw, SGD) hold precision constant (bf16)
3. DS (ds2, ds3) hold precision constant (bf16) # compares to adamW and bf16

"""

# Experiment optimizations
precision = None  # default is None, option is 'bf16'
optimizer = "adamw"  # default is Adamw, option is 'SGD', ('cpuadam', 'fusedadam' needed for ds2, ds3 respectively)
acceleratation_strategy = None  # default is None, option is 'ds2', 'ds3'


optims = f"{'fp32' if not precision else 'bf16'}-{optimizer}-{acceleratation_strategy if acceleratation_strategy else 'noDS'}"


# Define model with defaults from char shakespeare model
vocab_size = 65
block_size = 256
dropout = 0.2
compile = False  # Doesn't work too well with PL it appears?

# # Small
# n_layer = 6
# n_head = 6
# n_embd = 384
# model_name = "small"

# Med
# n_layer = 8
# n_head = 8
# n_embd = 512
# model_name = "med"

# Large
n_layer = 12
n_head = 12
n_embd = 768
model_name = "large"


weight_decay = 1e-1
learning_rate = 1e-3
betas = (0.9, 0.99)
device_type = "cuda"
device = "cuda"

dset_root = "../data"
dataset = "shakespeare_char"
batch_size = 64

max_steps = 100
dirpath = f"../experiment-logs/{model_name}-{optims}"
filename = "pt-profile"

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
    optimizer=optimizer,
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

profiler = PyTorchProfiler(dirpath=dirpath, filename=filename, profile_memory=True)

if acceleratation_strategy and acceleratation_strategy.lower() == "ds2":
    strategy = "deepspeed_stage_2_offload"
elif acceleratation_strategy and acceleratation_strategy.lower() == "ds3":
    strategy = "deepspeed_stage_3"
else:
    strategy = "auto"

trainer = Trainer(
    profiler=profiler,
    strategy=strategy,
    max_steps=max_steps,
    log_every_n_steps=10,
    val_check_interval=None,
    check_val_every_n_epoch=None,
    num_sanity_val_steps=0,  # Disable validation on startup
    precision=precision,
    benchmark=True,
)
# Easy way to disable validation is not provide a val_dataloader
trainer.fit(model, train_dataloaders=train_dloader)

# Save cuda memory summary to file
with open(os.path.join(dirpath, f"{filename}-memory-summary.txt"), "w") as f:
    f.write(torch.cuda.memory_summary())
print(torch.cuda.memory_summary())
