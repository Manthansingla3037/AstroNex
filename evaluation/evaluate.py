import sys
sys.path.append(r"C:\AstroNex")

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from training.model import AstroNexLunarSR


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AstroNex")

MODEL_PATH = PROJECT_ROOT / "models" / "lunar_sr_best.pth"

TEST_LR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "LR"
)

TEST_HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "HR"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("AstroNex Lunar Super Resolution Evaluation")
print("=" * 70)

print("\nLoading trained model...")

model = AstroNexLunarSR(scale=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support both our checkpoint format and normal state_dict
if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_rmse(reference, prediction):
    reference = reference.astype(np.float32)
    prediction = prediction.astype(np.float32)

    return float(
        np.sqrt(
            np.mean(
                (reference - prediction) ** 2
            )
        )
    )


def calculate_psnr(reference, prediction):
    reference = reference.astype(np.float32)
    prediction = prediction.astype(np.float32)

    mse = np.mean(
        (reference - prediction) ** 2
    )

    if mse == 0:
        return float("inf")

    max_value = 1.0

    return float(
        10 * np.log10(
            (max_value ** 2) / mse
        )
    )


# ============================================================
# FILES
# ============================================================

test_files = sorted(
    TEST_LR_DIR.glob("*.png")
)

if not test_files:
    raise RuntimeError(
        f"No test images found in {TEST_LR_DIR}"
    )

print("\nTest images:", len(test_files))


# ============================================================
# ACCUMULATORS
# ============================================================

bicubic_psnr_values = []
bicubic_ssim_values = []
bicubic_rmse_values = []

ai_psnr_values = []
ai_ssim_values = []
ai_rmse_values = []


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating...\n")

first_result_saved = False

for index, lr_path in enumerate(test_files):

    hr_path = TEST_HR_DIR / lr_path.name

    if not hr_path.exists():
        print(
            f"Skipping {lr_path.name}: "
            "HR pair missing."
        )
        continue

    # --------------------------------------------------------
    # Read LR
    # --------------------------------------------------------

    lr_image = Image.open(
        lr_path
    ).convert("L")

    lr = np.asarray(
        lr_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Read HR
    # --------------------------------------------------------

    hr_image = Image.open(
        hr_path
    ).convert("L")

    hr = np.asarray(
        hr_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Bicubic baseline
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
    # AstroNex AI
    # --------------------------------------------------------

    input_tensor = torch.from_numpy(
        lr
    ).float()

    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():

        prediction = model(
            input_tensor.to(DEVICE)
        )

    prediction = prediction.squeeze().numpy()

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Metrics - Bicubic
    # --------------------------------------------------------

    bicubic_psnr = calculate_psnr(
        hr,
        bicubic
    )

    bicubic_rmse = calculate_rmse(
        hr,
        bicubic
    )

    bicubic_ssim = ssim(
        hr,
        bicubic,
        data_range=1.0
    )

    # --------------------------------------------------------
    # Metrics - AI
    # --------------------------------------------------------

    ai_psnr = calculate_psnr(
        hr,
        prediction
    )

    ai_rmse = calculate_rmse(
        hr,
        prediction
    )

    ai_ssim = ssim(
        hr,
        prediction,
        data_range=1.0
    )

    bicubic_psnr_values.append(
        bicubic_psnr
    )

    bicubic_ssim_values.append(
        bicubic_ssim
    )

    bicubic_rmse_values.append(
        bicubic_rmse
    )

    ai_psnr_values.append(
        ai_psnr
    )

    ai_ssim_values.append(
        ai_ssim
    )

    ai_rmse_values.append(
        ai_rmse
    )

    # --------------------------------------------------------
    # Save first visual comparison
    # --------------------------------------------------------

    if not first_result_saved:

        lr_big = lr_image.resize(
            hr_image.size,
            Image.Resampling.NEAREST
        )

        comparison_width = (
            hr_image.width * 4
        )

        comparison = Image.new(
            "L",
            (
                comparison_width,
                hr_image.height
            )
        )

        comparison.paste(
            lr_big,
            (0, 0)
        )

        comparison.paste(
            bicubic_image,
            (hr_image.width, 0)
        )

        ai_image = Image.fromarray(
            np.uint8(
                prediction * 255
            )
        )

        comparison.paste(
            ai_image,
            (hr_image.width * 2, 0)
        )

        comparison.paste(
            hr_image,
            (hr_image.width * 3, 0)
        )

        comparison.save(
            OUTPUT_DIR
            / "comparison.png"
        )

        # Save individual AI output
        ai_image.save(
            OUTPUT_DIR
            / "ai_output.png"
        )

        first_result_saved = True

    if (index + 1) % 20 == 0:
        print(
            f"Evaluated "
            f"{index + 1}/{len(test_files)}"
        )


# ============================================================
# AVERAGE RESULTS
# ============================================================

bicubic_psnr_avg = np.mean(
    bicubic_psnr_values
)

bicubic_ssim_avg = np.mean(
    bicubic_ssim_values
)

bicubic_rmse_avg = np.mean(
    bicubic_rmse_values
)

ai_psnr_avg = np.mean(
    ai_psnr_values
)

ai_ssim_avg = np.mean(
    ai_ssim_values
)

ai_rmse_avg = np.mean(
    ai_rmse_values
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print("\nBICUBIC BASELINE")
print("----------------")
print(
    f"PSNR : {bicubic_psnr_avg:.4f} dB"
)
print(
    f"SSIM : {bicubic_ssim_avg:.4f}"
)
print(
    f"RMSE : {bicubic_rmse_avg:.6f}"
)

print("\nASTRONEX AI")
print("----------------")
print(
    f"PSNR : {ai_psnr_avg:.4f} dB"
)
print(
    f"SSIM : {ai_ssim_avg:.4f}"
)
print(
    f"RMSE : {ai_rmse_avg:.6f}"
)

print("\nIMPROVEMENT")
print("----------------")

print(
    f"PSNR improvement : "
    f"{ai_psnr_avg - bicubic_psnr_avg:.4f} dB"
)

print(
    f"SSIM improvement : "
    f"{ai_ssim_avg - bicubic_ssim_avg:.4f}"
)

print(
    f"RMSE reduction : "
    f"{bicubic_rmse_avg - ai_rmse_avg:.6f}"
)

print("\nVisual results saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)