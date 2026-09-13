from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class LunarSRV3Dataset(Dataset):
    def __init__(self, root_dir, split="train", lr_patch_size=64,
                 scale=8, patches_per_image=16, augment=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.lr_patch_size = int(lr_patch_size)
        self.scale = int(scale)
        self.hr_patch_size = self.lr_patch_size * self.scale
        self.patches_per_image = int(patches_per_image)
        self.augment = bool(augment)

        lr_dir = self.root_dir / split / "images_lr"
        hr_dir = self.root_dir / split / "images_hr"
        if not lr_dir.exists():
            raise FileNotFoundError(f"LR directory not found: {lr_dir}")
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")

        hr_by_name = {p.name: p for p in hr_dir.glob("*.png")}
        self.pairs = [(p, hr_by_name[p.name])
                      for p in sorted(lr_dir.glob("*.png"))
                      if p.name in hr_by_name]
        if not self.pairs:
            raise RuntimeError("No LR/HR PNG pairs found.")

        self.length = len(self.pairs) * self.patches_per_image if split == "train" else len(self.pairs)

    @staticmethod
    def _load(path):
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pair_index = index // self.patches_per_image if self.split == "train" else index
        lr_path, hr_path = self.pairs[pair_index]
        lr, hr = self._load(lr_path), self._load(hr_path)

        h, w = lr.shape
        expected = (h * self.scale, w * self.scale)
        if hr.shape != expected:
            raise ValueError(f"{lr_path.name}: LR={lr.shape}, HR={hr.shape}, expected HR={expected}")

        if self.split == "train":
            ps = self.lr_patch_size
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
            lr = lr[y:y+ps, x:x+ps]
            hy, hx, hs = y*self.scale, x*self.scale, ps*self.scale
            hr = hr[hy:hy+hs, hx:hx+hs]

            if self.augment and random.random() < 0.5:
                lr, hr = np.fliplr(lr).copy(), np.fliplr(hr).copy()
            if self.augment and random.random() < 0.5:
                lr, hr = np.flipud(lr).copy(), np.flipud(hr).copy()
            if self.augment:
                k = random.randint(0, 3)
                if k:
                    lr, hr = np.rot90(lr, k).copy(), np.rot90(hr, k).copy()
        else:
            ps = min(self.lr_patch_size, h, w)
            y, x = max((h-ps)//2, 0), max((w-ps)//2, 0)
            lr = lr[y:y+ps, x:x+ps]
            hy, hx, hs = y*self.scale, x*self.scale, ps*self.scale
            hr = hr[hy:hy+hs, hx:hx+hs]

        return {"lr": torch.from_numpy(lr).unsqueeze(0).float(),
                "hr": torch.from_numpy(hr).unsqueeze(0).float(),
                "name": lr_path.name}

if __name__ == "__main__":
    root = r"C:\AstroNex\external_datasets\lunar_sr_archive\archive"
    ds = LunarSRV3Dataset(root, "train", patches_per_image=1)
    s = ds[0]
    print("pairs:", len(ds.pairs))
    print("LR:", tuple(s["lr"].shape), "HR:", tuple(s["hr"].shape), s["name"])
