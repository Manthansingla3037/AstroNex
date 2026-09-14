from pathlib import Path
import csv
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.append(str(Path(__file__).resolve().parents[1] / "training"))

from dataset_v31 import LunarSRV31Dataset
from model_v3 import AstroNexLunarSRV3


DATA_ROOT = Path(
    r"C:\AstroNex\external_datasets\lunar_sr_archive\archive"
)
CHECKPOINT = Path(
    r"C:\AstroNex\models\lunar_sr_v31_best.pth"
)
OUTPUT_DIR = Path(
    r"C:\AstroNex\evaluation\results_v31"
)

SCALE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gray(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


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


def bicubic(lr):
    x = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        y = F.interpolate(
            x,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False
        )
    return y.squeeze().numpy()


def model_predict(model, lr):
    x = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        y = model(x)
    return y.squeeze().cpu().numpy()


def save_png(arr, path):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = AstroNexLunarSRV3(
        scale=8,
        channels=32,
        num_blocks=6,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Read only the final 16 images.
    dataset = LunarSRV31Dataset(
        DATA_ROOT,
        split="test",
        lr_patch_size=64,
        scale=8,
        patches_per_image=1,
        augment=False,
    )

    lr_dir = DATA_ROOT / "test" / "images_lr"
    hr_dir = DATA_ROOT / "test" / "images_hr"

    rows = []

    print("=" * 64)
    print("ASTRONEX V3.1 FINAL UNSEEN TEST")
    print("=" * 64)
    print("Device:", DEVICE)
    print("Final test pairs:", len(dataset.pairs))
    print("These images were not used for checkpoint selection.\n")

    for lr_path, hr_path in dataset.pairs:
        lr = load_gray(lr_path)
        hr = load_gray(hr_path)

        bic = np.clip(bicubic(lr), 0, 1)
        pred = np.clip(model_predict(model, lr), 0, 1)

        if pred.shape != hr.shape:
            raise ValueError(
                f"{lr_path.name}: prediction {pred.shape} != HR {hr.shape}"
            )

        b_psnr = psnr(bic, hr)
        v_psnr = psnr(pred, hr)

        b_ssim = ssim(bic, hr)
        v_ssim = ssim(pred, hr)

        b_rmse = rmse(bic, hr)
        v_rmse = rmse(pred, hr)

        rows.append({
            "image": lr_path.name,
            "bicubic_psnr": b_psnr,
            "v31_psnr": v_psnr,
            "psnr_gain": v_psnr - b_psnr,
            "bicubic_ssim": b_ssim,
            "v31_ssim": v_ssim,
            "ssim_gain": v_ssim - b_ssim,
            "bicubic_rmse": b_rmse,
            "v31_rmse": v_rmse,
            "rmse_change": v_rmse - b_rmse,
        })

        save_png(
            bic,
            OUTPUT_DIR / f"{lr_path.stem}_bicubic.png"
        )
        save_png(
            pred,
            OUTPUT_DIR / f"{lr_path.stem}_v31.png"
        )
        save_png(
            hr,
            OUTPUT_DIR / f"{lr_path.stem}_hr.png"
        )

        print(
            f"{lr_path.name}: "
            f"Bicubic={b_psnr:.3f} dB | "
            f"V3.1={v_psnr:.3f} dB | "
            f"Gain={v_psnr - b_psnr:+.3f} dB"
        )

    def avg(key):
        return float(np.mean([r[key] for r in rows]))

    print("\n" + "=" * 64)
    print("FINAL V3.1 RESULTS")
    print("=" * 64)
    print(f"Images evaluated: {len(rows)}")

    print("\nPSNR")
    print(f"Bicubic : {avg('bicubic_psnr'):.4f} dB")
    print(f"V3.1    : {avg('v31_psnr'):.4f} dB")
    print(f"Gain    : {avg('psnr_gain'):+.4f} dB")

    print("\nSSIM")
    print(f"Bicubic : {avg('bicubic_ssim'):.4f}")
    print(f"V3.1    : {avg('v31_ssim'):.4f}")
    print(f"Gain    : {avg('ssim_gain'):+.4f}")

    print("\nRMSE")
    print(f"Bicubic : {avg('bicubic_rmse'):.6f}")
    print(f"V3.1    : {avg('v31_rmse'):.6f}")
    print(f"Change  : {avg('rmse_change'):+.6f}")

    csv_path = OUTPUT_DIR / "v31_final_test_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUTPUT_DIR / "v31_final_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AstroNex V3.1 Final Unseen Test\n")
        f.write("==============================\n")
        f.write(f"Images evaluated: {len(rows)}\n\n")
        f.write(f"Bicubic PSNR: {avg('bicubic_psnr'):.4f} dB\n")
        f.write(f"V3.1 PSNR: {avg('v31_psnr'):.4f} dB\n")
        f.write(f"PSNR gain: {avg('psnr_gain'):+.4f} dB\n\n")
        f.write(f"Bicubic SSIM: {avg('bicubic_ssim'):.4f}\n")
        f.write(f"V3.1 SSIM: {avg('v31_ssim'):.4f}\n")
        f.write(f"SSIM gain: {avg('ssim_gain'):+.4f}\n\n")
        f.write(f"Bicubic RMSE: {avg('bicubic_rmse'):.6f}\n")
        f.write(f"V3.1 RMSE: {avg('v31_rmse'):.6f}\n")
        f.write(f"RMSE change: {avg('rmse_change'):+.6f}\n")

    print("\nSaved:")
    print(csv_path)
    print(summary_path)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
