from pathlib import Path
import numpy as np
from PIL import Image

# ============================================================
# AstroNex - OHRC Training Dataset Generator
# ============================================================

# Project paths
PROJECT_ROOT = Path(r"C:\AstroNex")

IMG_PATH = PROJECT_ROOT / "data" / "raw" / "OHRC" / "ohrc.img"

TRAIN_HR = PROJECT_ROOT / "data" / "processed" / "train" / "HR"
TRAIN_LR = PROJECT_ROOT / "data" / "processed" / "train" / "LR"

VAL_HR = PROJECT_ROOT / "data" / "processed" / "validation" / "HR"
VAL_LR = PROJECT_ROOT / "data" / "processed" / "validation" / "LR"

TEST_HR = PROJECT_ROOT / "data" / "processed" / "test" / "HR"
TEST_LR = PROJECT_ROOT / "data" / "processed" / "test" / "LR"


# ============================================================
# OHRC metadata
# ============================================================

ROWS = 101075
COLS = 12000

DTYPE = np.uint8

# Super-resolution scale
SCALE = 4

# High-resolution patch size
PATCH_SIZE = 256

# Distance between extracted patches
STRIDE = 256

# Number of patches for this first experiment
MAX_TRAIN = 600
MAX_VAL = 100
MAX_TEST = 100


# ============================================================
# Create output directories
# ============================================================

for folder in [
    TRAIN_HR, TRAIN_LR,
    VAL_HR, VAL_LR,
    TEST_HR, TEST_LR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# Basic validation
# ============================================================

print("=" * 60)
print("AstroNex OHRC Dataset Generator")
print("=" * 60)

if not IMG_PATH.exists():
    print("\nERROR: OHRC file not found:")
    print(IMG_PATH)
    raise SystemExit(1)

print("\nOHRC file:")
print(IMG_PATH)

file_size = IMG_PATH.stat().st_size
expected_size = ROWS * COLS * np.dtype(DTYPE).itemsize

print("\nActual size  :", file_size)
print("Expected size:", expected_size)

if file_size != expected_size:
    print("\nERROR: File size does not match expected metadata.")
    raise SystemExit(1)

print("\nFile size verified.")


# ============================================================
# Memory map the OHRC image
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
# Patch extraction function
# ============================================================

def generate_split(
    split_name,
    y_start,
    y_end,
    hr_folder,
    lr_folder,
    max_patches
):
    print(f"\n--- Creating {split_name} dataset ---")

    count = 0

    # Leave enough space for a complete patch
    last_y = y_end - PATCH_SIZE
    last_x = COLS - PATCH_SIZE

    for y in range(y_start, last_y + 1, STRIDE):

        if count >= max_patches:
            break

        for x in range(0, last_x + 1, STRIDE):

            if count >= max_patches:
                break

            # Read ONLY this small portion from the 1.2 GB file
            hr = np.array(
                image[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]
            )

            # ------------------------------------------------
            # Remove patches that contain almost no information
            # ------------------------------------------------

            mean_value = float(hr.mean())
            std_value = float(hr.std())

            if mean_value < 5:
                continue

            if std_value < 3:
                continue

            # ------------------------------------------------
            # Create LR image
            # ------------------------------------------------

            lr_size = PATCH_SIZE // SCALE

            hr_image = Image.fromarray(hr)

            lr_image = hr_image.resize(
                (lr_size, lr_size),
                Image.Resampling.BICUBIC
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            filename = f"{split_name.lower()}_{count:05d}.png"

            hr_path = hr_folder / filename
            lr_path = lr_folder / filename

            hr_image.save(hr_path)
            lr_image.save(lr_path)

            count += 1

            if count % 50 == 0:
                print(f"Created {count}/{max_patches} patches")

    print(f"{split_name} complete: {count} patches")

    return count


# ============================================================
# Split image spatially
# ============================================================

# We use different vertical regions for train/validation/test.
#
# This is better than randomly mixing patches from exactly the
# same area.

train_end = int(ROWS * 0.70)
val_end = int(ROWS * 0.85)

train_count = generate_split(
    "train",
    0,
    train_end,
    TRAIN_HR,
    TRAIN_LR,
    MAX_TRAIN
)

val_count = generate_split(
    "validation",
    train_end,
    val_end,
    VAL_HR,
    VAL_LR,
    MAX_VAL
)

test_count = generate_split(
    "test",
    val_end,
    ROWS,
    TEST_HR,
    TEST_LR,
    MAX_TEST
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)

print(f"\nTraining patches   : {train_count}")
print(f"Validation patches : {val_count}")
print(f"Test patches       : {test_count}")

print("\nDataset location:")
print(PROJECT_ROOT / "data" / "processed")

print("\nEach pair contains:")
print("HR:", PATCH_SIZE, "x", PATCH_SIZE)
print("LR:", PATCH_SIZE // SCALE, "x", PATCH_SIZE // SCALE)

print("\nScale factor:", SCALE, "x")

print("\nSUCCESS!")