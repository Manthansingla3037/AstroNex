from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class LunarSRDataset(Dataset):
    """
    Loads matching LR and HR lunar image patches.

    LR: 64 x 64
    HR: 256 x 256
    """

    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.files = sorted(
            [
                f for f in self.lr_dir.iterdir()
                if f.suffix.lower() == ".png"
            ]
        )

        if not self.files:
            raise RuntimeError(
                f"No PNG images found in {self.lr_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        lr_path = self.files[index]
        hr_path = self.hr_dir / lr_path.name

        if not hr_path.exists():
            raise FileNotFoundError(
                f"Missing HR pair for {lr_path.name}"
            )

        # Load grayscale images
        lr = Image.open(lr_path).convert("L")
        hr = Image.open(hr_path).convert("L")

        # Convert [0,255] -> [0,1]
        lr = torch.from_numpy(
            __import__("numpy").array(lr)
        ).float() / 255.0

        hr = torch.from_numpy(
            __import__("numpy").array(hr)
        ).float() / 255.0

        # Add channel dimension
        # [H,W] -> [1,H,W]
        lr = lr.unsqueeze(0)
        hr = hr.unsqueeze(0)

        return lr, hr


if __name__ == "__main__":

    dataset = LunarSRDataset(
        r"C:\AstroNex\data\processed\train\LR",
        r"C:\AstroNex\data\processed\train\HR"
    )

    print("Dataset loaded successfully.")
    print("Number of training pairs:", len(dataset))

    lr, hr = dataset[0]

    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)

    print("LR range:", lr.min().item(), "to", lr.max().item())
    print("HR range:", hr.min().item(), "to", hr.max().item())