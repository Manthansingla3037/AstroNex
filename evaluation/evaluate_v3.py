from pathlib import Path
import math
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "training"))

from model_v3 import AstroNexLunarSRV3


DATA_ROOT = Path(
    r"C:\AstroNex\external_datasets\lunar_sr_archive\archive"
)
CHECKPOINT = Path(
    r"C:\AstroNex\models\lunar_sr_v3_best.pth"
)
OUTPUT_DIR = Path(
    r"C:\AstroNex\evaluation\results_v3"
)

SCALE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gray(path):
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return arr


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def ssim(pred, target):
    return float(
        structural_similarity(
            target,
            pred,
            data_range=1.0
        )
    )


def bicubic_upscale(lr):
    x = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        y = F.interpolate(
            x,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False
        )
    return y.squeeze().numpy()


def model_upscale(model, lr):
    x = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        y = model(x)
    return y.squeeze().cpu().numpy()


def save_image(arr, path):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_lr = DATA_ROOT / "test" / "images_lr"
    test_hr = DATA_ROOT / "test" / "images_hr"

    if not test_lr.exists() or not test_hr.exists():
        raise FileNotFoundError(
            "Expected test/images_lr and test/images_hr were not found."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = AstroNexLunarSRV3(
        scale=8,
        channels=32,
        num_blocks=6
    ).to(DEVICE)

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    rows = []

    lr_files = sorted(test_lr.glob("*.png"))

    print("Device:", DEVICE)
    print("Test images:", len(lr_files))
    print()

    for lr_path in lr_files:
        hr_path = test_hr / lr_path.name

        if not hr_path.exists():
            print("Skipping missing HR:", lr_path.name)
            continue

        lr = load_gray(lr_path)
        hr = load_gray(hr_path)

        if hr.shape != (lr.shape[0] * SCALE, lr.shape[1] * SCALE):
            print("Skipping dimension mismatch:", lr_path.name)
            continue

        bic = np.clip(bicubic_upscale(lr), 0, 1)
        pred = np.clip(model_upscale(model, lr), 0, 1)

        bic_psnr = psnr(bic, hr)
        bic_ssim = ssim(bic, hr)
        bic_rmse = rmse(bic, hr)

        v3_psnr = psnr(pred, hr)
        v3_ssim = ssim(pred, hr)
        v3_rmse = rmse(pred, hr)

        rows.append({
            "image": lr_path.name,
            "bicubic_psnr": bic_psnr,
            "v3_psnr": v3_psnr,
            "psnr_gain": v3_psnr - bic_psnr,
            "bicubic_ssim": bic_ssim,
            "v3_ssim": v3_ssim,
            "ssim_gain": v3_ssim - bic_ssim,
            "bicubic_rmse": bic_rmse,
            "v3_rmse": v3_rmse,
            "rmse_change": v3_rmse - bic_rmse,
        })

        save_image(bic, OUTPUT_DIR / f"{lr_path.stem}_bicubic.png")
        save_image(pred, OUTPUT_DIR / f"{lr_path.stem}_v3.png")
        save_image(hr, OUTPUT_DIR / f"{lr_path.stem}_hr.png")

        print(
            f"{lr_path.name}: "
            f"Bicubic PSNR={bic_psnr:.3f}, "
            f"V3 PSNR={v3_psnr:.3f}, "
            f"gain={v3_psnr - bic_psnr:+.3f}"
        )

    if not rows:
        raise RuntimeError("No valid test pairs were evaluated.")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    print("\n==============================")
    print("AstroNex V3 TEST EVALUATION")
    print("==============================")
    print(f"Images evaluated: {len(rows)}")

    print("\nPSNR")
    print(f"Bicubic : {mean('bicubic_psnr'):.4f} dB")
    print(f"V3      : {mean('v3_psnr'):.4f} dB")
    print(f"Gain    : {mean('psnr_gain'):+.4f} dB")

    print("\nSSIM")
    print(f"Bicubic : {mean('bicubic_ssim'):.4f}")
    print(f"V3      : {mean('v3_ssim'):.4f}")
    print(f"Gain    : {mean('ssim_gain'):+.4f}")

    print("\nRMSE")
    print(f"Bicubic : {mean('bicubic_rmse'):.6f}")
    print(f"V3      : {mean('v3_rmse'):.6f}")
    print(f"Change  : {mean('rmse_change'):+.6f}")

    csv_path = OUTPUT_DIR / "v3_test_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUTPUT_DIR / "v3_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AstroNex V3 Test Evaluation\n")
        f.write("===========================\n")
        f.write(f"Images evaluated: {len(rows)}\n\n")
        f.write(
            f"Bicubic PSNR: {mean('bicubic_psnr'):.4f} dB\n"
        )
        f.write(
            f"V3 PSNR: {mean('v3_psnr'):.4f} dB\n"
        )
        f.write(
            f"PSNR gain: {mean('psnr_gain'):+.4f} dB\n\n"
        )
        f.write(
            f"Bicubic SSIM: {mean('bicubic_ssim'):.4f}\n"
        )
        f.write(
            f"V3 SSIM: {mean('v3_ssim'):.4f}\n"
        )
        f.write(
            f"SSIM gain: {mean('ssim_gain'):+.4f}\n\n"
        )
        f.write(
            f"Bicubic RMSE: {mean('bicubic_rmse'):.6f}\n"
        )
        f.write(
            f"V3 RMSE: {mean('v3_rmse'):.6f}\n"
        )
        f.write(
            f"RMSE change: {mean('rmse_change'):+.6f}\n"
        )

    print("\nSaved:")
    print(csv_path)
    print(summary_path)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()