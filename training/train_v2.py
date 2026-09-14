from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LunarSRDataset
from model_v2 import AstroNexLunarSRV2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AstroNex")

TRAIN_LR = (
    PROJECT_ROOT
    / "data"
    / "v2_dataset"
    / "train"
    / "LR"
)

TRAIN_HR = (
    PROJECT_ROOT
    / "data"
    / "v2_dataset"
    / "train"
    / "HR"
)

VAL_LR = (
    PROJECT_ROOT
    / "data"
    / "v2_dataset"
    / "validation"
    / "LR"
)

VAL_HR = (
    PROJECT_ROOT
    / "data"
    / "v2_dataset"
    / "validation"
    / "HR"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lunar_sr_v2_best.pth"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")


BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 0.0001

SCHEDULER_PATIENCE = 5   # will use below

NUM_WORKERS = 0


# ============================================================
# START
# ============================================================

print("=" * 70)
print("AstroNex Lunar Super Resolution V2 Training")
print("=" * 70)

print("\nDevice:", DEVICE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# DATASETS
# ============================================================

print("\nLoading V2 datasets...")

train_dataset = LunarSRDataset(
    TRAIN_LR,
    TRAIN_HR
)

val_dataset = LunarSRDataset(
    VAL_LR,
    VAL_HR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

print("\nCreating V2 model...")

model = AstroNexLunarSRV2(scale=4)

model = model.to(DEVICE)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", parameter_count)


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


print("\nStarting V2 training...\n")


for epoch in range(1, EPOCHS + 1):

    start_time = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for batch_index, (lr, hr) in enumerate(
        train_loader,
        start=1
    ):

        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        optimizer.zero_grad()

        output = model(lr)

        loss = criterion(
            output,
            hr
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if (
            batch_index % 25 == 0
            or batch_index == len(train_loader)
        ):

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| Batch {batch_index}/{len(train_loader)} "
                f"| Loss: {loss.item():.6f}"
            )

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            output = model(lr)

            loss = criterion(
                output,
                hr
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scale": 4,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            MODEL_PATH
        )

        status = "BEST MODEL SAVED"

    else:

        status = ""

    # --------------------------------------------------------
    # Epoch report
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Time: {elapsed:.1f} seconds"
    )

    print(
        f"Train Loss: {train_loss:.6f}"
    )

    print(
        f"Val Loss:   {val_loss:.6f}"
    )

    print(
        f"Best Val:   {best_val_loss:.6f}"
    )

    print(status)

    scheduler.step(val_loss)

    print("-" * 70)
    print()


# ============================================================
# FINISH
# ============================================================

print("=" * 70)
print("ASTRONEX V2 TRAINING COMPLETE")
print("=" * 70)

print("\nBest validation loss:")
print(best_val_loss)

print("\nModel saved at:")
print(MODEL_PATH)