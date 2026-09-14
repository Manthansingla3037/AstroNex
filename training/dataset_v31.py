from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LunarSRV31Dataset(Dataset):
    """
    AstroNex V3.1 dataset with a clean 96/16/16 split.

    Existing archive:
        train/images_lr + train/images_hr = 96 paired images
        test/images_lr  + test/images_hr  = 32 paired images

    V3.1 uses:
        96 existing train pairs -> TRAIN
        first 16 sorted test pairs -> VALIDATION
        last 16 sorted test pairs  -> FINAL TEST

    LR: 128x128, HR: 1024x1024, scale=8.
    Training returns aligned random 64x64 -> 512x512 patches.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        lr_patch_size=64,
        scale=8,
        patches_per_image=16,
        augment=False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.lr_patch_size = int(lr_patch_size)
        self.scale = int(scale)
        self.patches_per_image = int(patches_per_image)
        self.augment = bool(augment)

        if self.scale != 8:
            raise ValueError("V3.1 is configured for 8x SR.")

        if split == "train":
            split_dir = self.root_dir / "train"
        elif split in ("val", "test"):
            split_dir = self.root_dir / "test"
        else:
            raise ValueError("split must be train, val, or test")

        lr_dir = split_dir / "images_lr"
        hr_dir = split_dir / "images_hr"

        if not lr_dir.exists():
            raise FileNotFoundError(f"Missing LR directory: {lr_dir}")
        if not hr_dir.exists():
            raise FileNotFoundError(f"Missing HR directory: {hr_dir}")

        hr_by_name = {p.name: p for p in hr_dir.glob("*.png")}
        all_pairs = [
            (p, hr_by_name[p.name])
            for p in sorted(lr_dir.glob("*.png"))
            if p.name in hr_by_name
        ]

        if split == "train":
            self.pairs = all_pairs
        else:
            if len(all_pairs) < 32:
                raise RuntimeError(
                    f"Expected at least 32 test pairs; found {len(all_pairs)}"
                )
            midpoint = len(all_pairs) // 2
            if split == "val":
                self.pairs = all_pairs[:midpoint]
            else:
                self.pairs = all_pairs[midpoint:]

        if not self.pairs:
            raise RuntimeError(f"No pairs found for split={split}")

        self.length = (
            len(self.pairs) * self.patches_per_image
            if split == "train"
            else len(self.pairs)
        )

    @staticmethod
    def _load_gray(path):
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return arr

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pair_index = (
            index // self.patches_per_image
            if self.split == "train"
            else index
        )

        lr_path, hr_path = self.pairs[pair_index]
        lr = self._load_gray(lr_path)
        hr = self._load_gray(hr_path)

        h, w = lr.shape
        expected_hr = (h * self.scale, w * self.scale)
        if hr.shape != expected_hr:
            raise ValueError(
                f"{lr_path.name}: LR={lr.shape}, HR={hr.shape}, "
                f"expected HR={expected_hr}"
            )

        if self.split == "train":
            ps = self.lr_patch_size
            if h < ps or w < ps:
                raise ValueError(
                    f"{lr_path.name} is smaller than the {ps}x{ps} LR patch."
                )

            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)

            lr = lr[y:y + ps, x:x + ps]

            hy = y * self.scale
            hx = x * self.scale
            hs = ps * self.scale
            hr = hr[hy:hy + hs, hx:hx + hs]

            if self.augment and random.random() < 0.5:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()

            if self.augment and random.random() < 0.5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()

            if self.augment:
                k = random.randint(0, 3)
                if k:
                    lr = np.rot90(lr, k).copy()
                    hr = np.rot90(hr, k).copy()

        else:
            # Deterministic center crop for validation/test checkpoint selection.
            ps = self.lr_patch_size
            y = max((h - ps) // 2, 0)
            x = max((w - ps) // 2, 0)

            lr = lr[y:y + ps, x:x + ps]

            hy = y * self.scale
            hx = x * self.scale
            hs = ps * self.scale
            hr = hr[hy:hy + hs, hx:hx + hs]

        return {
            "lr": torch.from_numpy(lr).unsqueeze(0).float(),
            "hr": torch.from_numpy(hr).unsqueeze(0).float(),
            "name": lr_path.name,
        }


if __name__ == "__main__":
    root = r"C:\AstroNex\external_datasets\lunar_sr_archive\archive"

    for split in ("train", "val", "test"):
        ds = LunarSRV31Dataset(
            root,
            split=split,
            patches_per_image=1,
            augment=False,
        )
        sample = ds[0]
        print(
            f"{split:5s}: pairs={len(ds.pairs):2d}, "
            f"LR={tuple(sample['lr'].shape)}, "
            f"HR={tuple(sample['hr'].shape)}, "
            f"first={sample['name']}"
        )
