from pathlib import Path
import argparse, math, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_v3 import LunarSRV3Dataset
from model_v3 import AstroNexLunarSRV3

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def charbonnier(pred, target, eps=1e-3):
    d = pred - target
    return torch.mean(torch.sqrt(d*d + eps*eps))

def psnr(mse):
    return float("inf") if mse <= 0 else 10*math.log10(1.0/mse)

def evaluate(model, loader, device):
    model.eval(); loss_sum = mse_sum = n = 0
    with torch.no_grad():
        for b in loader:
            lr, hr = b["lr"].to(device), b["hr"].to(device)
            pred = model(lr)
            l = charbonnier(pred, hr).item()
            m = torch.mean((pred-hr)**2).item()
            bs = lr.size(0)
            loss_sum += l*bs; mse_sum += m*bs; n += bs
    return loss_sum/max(n,1), psnr(mse_sum/max(n,1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\AstroNex\external_datasets\lunar_sr_archive\archive")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--patches-per-image", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=r"C:\AstroNex\models\lunar_sr_v3_best.pth")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = LunarSRV3Dataset(args.data, "train", 64, 8, args.patches_per_image, True)
    test_ds = LunarSRV3Dataset(args.data, "test", 64, 8, 1, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = AstroNexLunarSRV3().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best = -float("inf")
    ckpt = Path(args.checkpoint); ckpt.parent.mkdir(parents=True, exist_ok=True)

    print("Train pairs:", len(train_ds.pairs), "samples/epoch:", len(train_ds))
    print("Test pairs:", len(test_ds.pairs))

    for epoch in range(1, args.epochs+1):
        model.train(); total = seen = 0
        for b in train_loader:
            lr, hr = b["lr"].to(device), b["hr"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(lr)
            loss = charbonnier(pred, hr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()*lr.size(0); seen += lr.size(0)

        train_loss = total/max(seen,1)
        val_loss, val_psnr = evaluate(model, test_loader, device)
        sched.step(val_loss)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.6f} | test_patch_loss={val_loss:.6f} | test_patch_PSNR={val_psnr:.3f} dB | lr={opt.param_groups[0]['lr']:.2e}")

        if val_psnr > best:
            best = val_psnr
            torch.save({
                "model_state_dict": model.state_dict(),
                "scale": 8, "channels": 32, "num_blocks": 6,
                "best_test_patch_psnr": best,
            }, ckpt)
            print("Saved:", ckpt)

    print("Training complete. Best test-patch PSNR:", round(best,3), "dB")
    print("Checkpoint:", ckpt)

if __name__ == "__main__":
    main()
