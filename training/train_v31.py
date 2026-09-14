from pathlib import Path
import argparse
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_v31 import LunarSRV31Dataset
from model_v3 import AstroNexLunarSRV3


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def mse_to_psnr(mse):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def validate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    mse_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            pred = model(lr)

            loss = charbonnier_loss(pred, hr).item()
            mse = torch.mean((pred - hr) ** 2).item()

            bs = lr.size(0)
            loss_sum += loss * bs
            mse_sum += mse * bs
            n += bs

    avg_loss = loss_sum / max(n, 1)
    avg_mse = mse_sum / max(n, 1)
    return avg_loss, mse_to_psnr(avg_mse)


def main():
    parser = argparse.ArgumentParser(
        description="AstroNex V3.1: 96 train / 16 val / 16 final test"
    )
    parser.add_argument(
        "--data",
        default=r"C:\AstroNex\external_datasets\lunar_sr_archive\archive",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default=r"C:\AstroNex\models\lunar_sr_v31_best.pth",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # CPU-first: automatically uses CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("ASTRONEX V3.1 TRAINING")
    print("=" * 64)
    print("Device:", device)

    train_ds = LunarSRV31Dataset(
        args.data,
        split="train",
        lr_patch_size=64,
        scale=8,
        patches_per_image=args.patches_per_image,
        augment=True,
    )

    val_ds = LunarSRV31Dataset(
        args.data,
        split="val",
        lr_patch_size=64,
        scale=8,
        patches_per_image=1,
        augment=False,
    )

    final_test_ds = LunarSRV31Dataset(
        args.data,
        split="test",
        lr_patch_size=64,
        scale=8,
        patches_per_image=1,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Train pairs :", len(train_ds.pairs))
    print("Val pairs   :", len(val_ds.pairs))
    print("Final tests :", len(final_test_ds.pairs))
    print("Train samples/epoch:", len(train_ds))

    model = AstroNexLunarSRV3(
        scale=8,
        channels=32,
        num_blocks=6,
    ).to(device)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_val_psnr = -float("inf")

    print("\nFinal test images are NEVER used for checkpoint selection.")
    print("Training starts...\n")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()

        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(lr_img)
            loss = charbonnier_loss(pred, hr_img)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * lr_img.size(0)
            seen += lr_img.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, val_psnr = validate(model, val_loader, device)

        scheduler.step(val_loss)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.4f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scale": 8,
                    "channels": 32,
                    "num_blocks": 6,
                    "best_val_psnr": best_val_psnr,
                    "split": "96 train / 16 validation / 16 final test",
                },
                checkpoint,
            )

            print("  -> BEST MODEL SAVED")

    print("\n" + "=" * 64)
    print("V3.1 TRAINING COMPLETE")
    print("=" * 64)
    print(f"Best validation PSNR: {best_val_psnr:.4f} dB")
    print("Checkpoint:", checkpoint)
    print(
        "\nThe 16-image final test set was not used during training "
        "or checkpoint selection."
    )


if __name__ == "__main__":
    main()
