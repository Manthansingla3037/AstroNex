from pathlib import Path
import numpy as np
import cv2
from PIL import Image


# ============================================================
# AstroNex Super Resolution Dataset V2
# ============================================================

PROJECT_ROOT = Path(r"C:\AstroNex")

IMG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "OHRC"
    / "ohrc.img"
)

BASE_OUTPUT = PROJECT_ROOT / "data" / "processed_v2"


ROWS = 101075
COLS = 12000

DTYPE = np.uint8

SCALE = 4

PATCH_SIZE = 256
STRIDE = 256

MAX_TRAIN = 1000
MAX_VAL = 200
MAX_TEST = 200


# ============================================================
# Create directories
# ============================================================

splits = [
    "train",
    "validation",
    "test"
]

for split in splits:

    (BASE_OUTPUT / split / "HR").mkdir(
        parents=True,
        exist_ok=True
    )

    (BASE_OUTPUT / split / "LR").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Check file
# ============================================================

print("=" * 70)
print("AstroNex SR Dataset V2")
print("=" * 70)

if not IMG_PATH.exists():

    print("\nERROR: OHRC image not found.")
    print(IMG_PATH)

    raise SystemExit(1)


expected_size = (
    ROWS
    * COLS
    * np.dtype(DTYPE).itemsize
)

actual_size = IMG_PATH.stat().st_size

print("\nOHRC file:", IMG_PATH)

print("Actual size  :", actual_size)
print("Expected size:", expected_size)

if actual_size != expected_size:

    print("\nERROR: File size mismatch.")
    raise SystemExit(1)


# ============================================================
# Memory map
# ============================================================

print("\nOpening OHRC using memory mapping...")

image = np.memmap(
    IMG_PATH,
    dtype=DTYPE,
    mode="r",
    shape=(ROWS, COLS)
)

print("Image shape:", image.shape)


# ============================================================
# Degradation function
# ============================================================

def create_lr(hr):

    # Convert to float
    hr_float = hr.astype(np.float32)

    # ----------------------------------------------
    # Slight Gaussian blur
    # ----------------------------------------------

    blurred = cv2.GaussianBlur(
        hr_float,
        (5, 5),
        0.8
    )

    # ----------------------------------------------
    # Downsample
    # ----------------------------------------------

    lr_size = PATCH_SIZE // SCALE

    lr = cv2.resize(
        blurred,
        (lr_size, lr_size),
        interpolation=cv2.INTER_AREA
    )

    # ----------------------------------------------
    # Add very small sensor-like noise
    # ----------------------------------------------

    noise = np.random.normal(
        loc=0,
        scale=1.0,
        size=lr.shape
    )

    lr = lr + noise

    # ----------------------------------------------
    # Clip
    # ----------------------------------------------

    lr = np.clip(
        lr,
        0,
        255
    )

    return lr.astype(np.uint8)


# ============================================================
# Generate split
# ============================================================

def generate_split(
    name,
    y_start,
    y_end,
    max_patches
):

    hr_folder = (
        BASE_OUTPUT
        / name
        / "HR"
    )

    lr_folder = (
        BASE_OUTPUT
        / name
        / "LR"
    )

    count = 0

    print(f"\n--- Creating {name} ---")

    last_y = y_end - PATCH_SIZE
    last_x = COLS - PATCH_SIZE

    for y in range(
        y_start,
        last_y + 1,
        STRIDE
    ):

        if count >= max_patches:
            break

        for x in range(
            0,
            last_x + 1,
            STRIDE
        ):

            if count >= max_patches:
                break

            # Read small crop only
            hr = np.array(
                image[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]
            )

            mean_value = float(hr.mean())
            std_value = float(hr.std())

            # Skip almost empty patches
            if mean_value < 5:
                continue

            if std_value < 3:
                continue

            # Create degraded LR
            lr = create_lr(hr)

            # Save
            filename = (
                f"{name}_{count:05d}.png"
            )

            Image.fromarray(hr).save(
                hr_folder / filename
            )

            Image.fromarray(lr).save(
                lr_folder / filename
            )

            count += 1

            if count % 100 == 0:

                print(
                    f"Created "
                    f"{count}/{max_patches}"
                )

    print(
        f"{name} complete: {count} patches"
    )


# ============================================================
# Spatial split
# ============================================================

train_end = int(ROWS * 0.70)
val_end = int(ROWS * 0.85)

generate_split(
    "train",
    0,
    train_end,
    MAX_TRAIN
)

generate_split(
    "validation",
    train_end,
    val_end,
    MAX_VAL
)

generate_split(
    "test",
    val_end,
    ROWS,
    MAX_TEST
)


print("\n" + "=" * 70)
print("V2 DATASET COMPLETE")
print("=" * 70)

print("\nLocation:")
print(BASE_OUTPUT)

print("\nHR:", PATCH_SIZE, "x", PATCH_SIZE)

print(
    "LR:",
    PATCH_SIZE // SCALE,
    "x",
    PATCH_SIZE // SCALE
)

print("\nScale:", SCALE, "x")