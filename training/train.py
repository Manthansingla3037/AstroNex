from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LunarSRDataset
from model import AstroNexLunarSR


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\AstroNex")

TRAIN_LR = PROJECT_ROOT / "data" / "processed" / "train" / "LR"
TRAIN_HR = PROJECT_ROOT / "data" / "processed" / "train" / "HR"

VAL_LR = PROJECT_ROOT / "data" / "processed" / "validation" / "LR"
VAL_HR = PROJECT_ROOT / "data" / "processed" / "validation" / "HR"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "lunar_sr_best.pth"

# CPU-friendly settings
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cpu")

# Windows + CPU: keep workers at 0 for reliability
NUM_WORKERS = 0


# ============================================================
# DATASET
# ============================================================

print("=" * 70)
print("AstroNex Lunar Super Resolution Training")
print("=" * 70)

print("\nDevice:", DEVICE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)

print("\nLoading datasets...")

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

print("\nCreating model...")

model = AstroNexLunarSR(scale=4)
model = model.to(DEVICE)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", parameter_count)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for batch_index, (lr, hr) in enumerate(train_loader, start=1):

        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        optimizer.zero_grad()

        output = model(lr)

        loss = criterion(output, hr)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if batch_index % 25 == 0 or batch_index == len(train_loader):

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| Batch {batch_index}/{len(train_loader)} "
                f"| Loss: {loss.item():.6f}"
            )

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            output = model(lr)

            loss = criterion(output, hr)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # --------------------------------------------------------
    # SAVE BEST MODEL
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
            BEST_MODEL_PATH
        )

        saved_message = " ← BEST MODEL SAVED"

    else:

        saved_message = ""

    elapsed = time.time() - epoch_start

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS} completed "
        f"in {elapsed:.1f} seconds"
    )

    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")
    print(f"Best Val:   {best_val_loss:.6f}")

    print("-" * 70)
    print(saved_message)
    print()


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nBest validation loss:", best_val_loss)

print("\nModel saved at:")
print(BEST_MODEL_PATH)

print("\nNext step:")
print("Evaluate the trained model using PSNR / SSIM / RMSE.")