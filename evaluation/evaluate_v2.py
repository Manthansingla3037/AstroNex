import sys
sys.path.append(r"C:\AstroNex")

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from training.model_v2 import AstroNexLunarSRV2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AstroNex")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lunar_sr_v2_best.pth"
)

TEST_LR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_v2"
    / "test"
    / "LR"
)

TEST_HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_v2"
    / "test"
    / "HR"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("ASTRONEX LUNAR SUPER RESOLUTION V2 EVALUATION")
print("=" * 70)

print("\nLoading trained V2 model...")

model = AstroNexLunarSRV2(scale=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(DEVICE)
model.eval()

print("V2 model loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def psnr(reference, prediction):

    mse = np.mean(
        (reference - prediction) ** 2
    )

    if mse == 0:
        return float("inf")

    return float(
        10 * np.log10(
            1.0 / mse
        )
    )


def rmse(reference, prediction):

    return float(
        np.sqrt(
            np.mean(
                (reference - prediction) ** 2
            )
        )
    )


# ============================================================
# TEST FILES
# ============================================================

test_files = sorted(
    TEST_LR_DIR.glob("*.png")
)

print("\nTest images:", len(test_files))

if not test_files:

    raise RuntimeError(
        "No test images found."
    )


# ============================================================
# RESULT ARRAYS
# ============================================================

bicubic_psnr = []
bicubic_ssim = []
bicubic_rmse = []

v2_psnr = []
v2_ssim = []
v2_rmse = []


# ============================================================
# PROCESS TEST SET
# ============================================================

print("\nEvaluating test set...\n")

comparison_saved = False


for index, lr_path in enumerate(test_files):

    hr_path = TEST_HR_DIR / lr_path.name

    if not hr_path.exists():
        continue

    # --------------------------------------------------------
    # Load LR
    # --------------------------------------------------------

    lr_image = Image.open(
        lr_path
    ).convert("L")

    lr = np.asarray(
        lr_image,
        dtype=np.float32
    ) / 255.0


    # --------------------------------------------------------
    # Load HR
    # --------------------------------------------------------

    hr_image = Image.open(
        hr_path
    ).convert("L")

    hr = np.asarray(
        hr_image,
        dtype=np.float32
    ) / 255.0


    # --------------------------------------------------------
    # Bicubic
    # --------------------------------------------------------

    bicubic_image = lr_image.resize(
        hr_image.size,
        Image.Resampling.BICUBIC
    )

    bicubic = np.asarray(
        bicubic_image,
        dtype=np.float32
    ) / 255.0


    # --------------------------------------------------------
    # AstroNex V2
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        lr
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        prediction = model(
            tensor.to(DEVICE)
        )

    prediction = (
        prediction
        .squeeze()
        .numpy()
    )

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Bicubic metrics
    # --------------------------------------------------------

    b_psnr = psnr(
        hr,
        bicubic
    )

    b_ssim = ssim(
        hr,
        bicubic,
        data_range=1.0
    )

    b_rmse = rmse(
        hr,
        bicubic
    )


    # --------------------------------------------------------
    # V2 metrics
    # --------------------------------------------------------

    a_psnr = psnr(
        hr,
        prediction
    )

    a_ssim = ssim(
        hr,
        prediction,
        data_range=1.0
    )

    a_rmse = rmse(
        hr,
        prediction
    )


    bicubic_psnr.append(b_psnr)
    bicubic_ssim.append(b_ssim)
    bicubic_rmse.append(b_rmse)

    v2_psnr.append(a_psnr)
    v2_ssim.append(a_ssim)
    v2_rmse.append(a_rmse)


    # --------------------------------------------------------
    # Save first comparison
    # --------------------------------------------------------

    if not comparison_saved:

        ai_image = Image.fromarray(
            np.uint8(
                prediction * 255
            )
        )

        # Make LR visible at HR size
        lr_big = lr_image.resize(
            hr_image.size,
            Image.Resampling.NEAREST
        )

        width = hr_image.width

        comparison = Image.new(
            "L",
            (
                width * 4,
                hr_image.height
            )
        )

        comparison.paste(
            lr_big,
            (0, 0)
        )

        comparison.paste(
            bicubic_image,
            (width, 0)
        )

        comparison.paste(
            ai_image,
            (width * 2, 0)
        )

        comparison.paste(
            hr_image,
            (width * 3, 0)
        )

        comparison.save(
            OUTPUT_DIR
            / "comparison_v2.png"
        )

        ai_image.save(
            OUTPUT_DIR
            / "ai_output_v2.png"
        )

        comparison_saved = True


    if (index + 1) % 20 == 0:

        print(
            f"Evaluated "
            f"{index + 1}/{len(test_files)}"
        )


# ============================================================
# AVERAGES
# ============================================================

b_psnr_avg = np.mean(bicubic_psnr)
b_ssim_avg = np.mean(bicubic_ssim)
b_rmse_avg = np.mean(bicubic_rmse)

a_psnr_avg = np.mean(v2_psnr)
a_ssim_avg = np.mean(v2_ssim)
a_rmse_avg = np.mean(v2_rmse)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL V2 EVALUATION")
print("=" * 70)


print("\nBICUBIC BASELINE")
print("----------------")
print(
    f"PSNR : {b_psnr_avg:.4f} dB"
)

print(
    f"SSIM : {b_ssim_avg:.4f}"
)

print(
    f"RMSE : {b_rmse_avg:.6f}"
)


print("\nASTRONEX V2")
print("----------------")
print(
    f"PSNR : {a_psnr_avg:.4f} dB"
)

print(
    f"SSIM : {a_ssim_avg:.4f}"
)

print(
    f"RMSE : {a_rmse_avg:.6f}"
)


print("\nV2 RELATIVE TO BICUBIC")
print("----------------------")

print(
    f"PSNR difference : "
    f"{a_psnr_avg - b_psnr_avg:.4f} dB"
)

print(
    f"SSIM difference : "
    f"{a_ssim_avg - b_ssim_avg:.4f}"
)

print(
    f"RMSE difference : "
    f"{a_rmse_avg - b_rmse_avg:.6f}"
)


print("\nVISUAL RESULTS:")
print(OUTPUT_DIR)

print(
    "\n" + "=" * 70
)

print(
    "V2 EVALUATION COMPLETE"
)

print(
    "=" * 70
)