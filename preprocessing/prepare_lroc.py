"""
AstroNex - LRO NAC .IMG to training patches converter
Run: python preprocessing\prepare_lroc.py
"""

import os
import glob
import random
import numpy as np
from PIL import Image

# ============================================================
# PATHS
# ============================================================

INPUT_DIR       = r"C:\AstroNex\data\raw\ohrc"

OUTPUT_TRAIN_HR = r"C:\AstroNex\data\v2_dataset\train\HR"
OUTPUT_TRAIN_LR = r"C:\AstroNex\data\v2_dataset\train\LR"
OUTPUT_VAL_HR   = r"C:\AstroNex\data\v2_dataset\validation\HR"
OUTPUT_VAL_LR   = r"C:\AstroNex\data\v2_dataset\validation\LR"

# ============================================================
# SETTINGS
# ============================================================

TRAIN_SPLIT       = 0.85   # 85% train, 15% val
PATCH_SIZE        = 256    # HR patch size (pixels)
SCALE             = 4      # LR = 64x64
PATCHES_PER_IMAGE = 40

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(OUTPUT_TRAIN_HR, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_LR, exist_ok=True)
os.makedirs(OUTPUT_VAL_HR,   exist_ok=True)
os.makedirs(OUTPUT_VAL_LR,   exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def read_img_file(path):
    """Read NASA PDS3 .IMG file as numpy array"""

    # Try rasterio first (handles PDS3 natively)
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            mn, mx = data.min(), data.max()
            if mx > mn:
                data = (data - mn) / (mx - mn) * 255.0
            return data.astype(np.uint8)
    except Exception as e:
        print(f"  rasterio failed: {e}, trying raw read...")

    # Fallback: raw binary read (works for most LRO NAC EDRs)
    with open(path, 'rb') as f:
        raw = f.read()

    width = 5000
    skip = 1024 * 1024  # skip 1MB PDS label area
    img_bytes = raw[skip:]
    height = len(img_bytes) // width

    if height < 100:
        skip = 0
        img_bytes = raw
        height = len(img_bytes) // width

    arr = np.frombuffer(img_bytes[:height * width], dtype=np.uint8)
    return arr.reshape(height, width)


def extract_patches(arr, patch_size, n_patches):
    """Extract random good-quality patches from image array"""
    h, w = arr.shape
    patches = []
    attempts = 0

    while len(patches) < n_patches and attempts < n_patches * 10:
        attempts += 1

        if h < patch_size or w < patch_size:
            break

        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        patch = arr[y:y + patch_size, x:x + patch_size]

        # Skip mostly-black shadow patches
        if patch.mean() < 10 or patch.mean() > 245:
            continue

        # Skip featureless flat patches
        if patch.std() < 5:
            continue

        patches.append(patch)

    return patches


def save_pair(patch, idx, hr_dir, lr_dir):
    """Save one HR patch and its synthetic LR version"""
    hr_img = Image.fromarray(patch, mode='L')
    hr_img.save(os.path.join(hr_dir, f"patch_{idx:05d}.png"))

    lr_size = PATCH_SIZE // SCALE  # 64
    lr_arr = np.array(
        hr_img.resize((lr_size, lr_size), Image.BICUBIC)
    ).astype(np.float32)

    # Add slight sensor noise
    lr_arr += np.random.normal(0, 2, lr_arr.shape)
    lr_arr = np.clip(lr_arr, 0, 255).astype(np.uint8)

    Image.fromarray(lr_arr, mode='L').save(
        os.path.join(lr_dir, f"patch_{idx:05d}.png")
    )

# ============================================================
# MAIN
# ============================================================

img_files = (
    glob.glob(os.path.join(INPUT_DIR, "*.IMG")) +
    glob.glob(os.path.join(INPUT_DIR, "*.img"))
)

print(f"Found {len(img_files)} .IMG files in {INPUT_DIR}")

if len(img_files) == 0:
    print("ERROR: No .IMG files found. Check INPUT_DIR path.")
    exit(1)

# Collect all patches from all images
all_patches = []

for img_path in img_files:
    fname = os.path.basename(img_path)
    print(f"\nProcessing {fname}...")

    try:
        arr = read_img_file(img_path)
        print(f"  Loaded: shape={arr.shape}, mean={arr.mean():.1f}, std={arr.std():.1f}")
    except Exception as e:
        print(f"  ERROR loading: {e} — skipping this file")
        continue

    patches = extract_patches(arr, PATCH_SIZE, PATCHES_PER_IMAGE)
    print(f"  Extracted {len(patches)} patches")
    all_patches.extend(patches)

print(f"\nTotal patches collected: {len(all_patches)}")

if len(all_patches) == 0:
    print("ERROR: No patches extracted. Check your .IMG files.")
    exit(1)

# Shuffle and split
random.shuffle(all_patches)
split_idx     = int(len(all_patches) * TRAIN_SPLIT)
train_patches = all_patches[:split_idx]
val_patches   = all_patches[split_idx:]

print(f"Train patches : {len(train_patches)}")
print(f"Val patches   : {len(val_patches)}")

# Save train patches
print("\nSaving train patches...")
for i, patch in enumerate(train_patches):
    save_pair(patch, i, OUTPUT_TRAIN_HR, OUTPUT_TRAIN_LR)
    if (i + 1) % 50 == 0:
        print(f"  Saved {i + 1}/{len(train_patches)}")

# Save validation patches
print("\nSaving validation patches...")
for i, patch in enumerate(val_patches):
    save_pair(patch, i, OUTPUT_VAL_HR, OUTPUT_VAL_LR)
    if (i + 1) % 20 == 0:
        print(f"  Saved {i + 1}/{len(val_patches)}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Train HR : {OUTPUT_TRAIN_HR}")
print(f"Train LR : {OUTPUT_TRAIN_LR}")
print(f"Val HR   : {OUTPUT_VAL_HR}")
print(f"Val LR   : {OUTPUT_VAL_LR}")
print(f"\nNext step: python training\\train_v2.py")