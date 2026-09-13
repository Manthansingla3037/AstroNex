import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.block(x)

class AstroNexLunarSRV3(nn.Module):
    """CPU-friendly 8x grayscale lunar SR model using bicubic + learned residual."""
    def __init__(self, scale=8, channels=32, num_blocks=6):
        super().__init__()
        if scale != 8:
            raise ValueError("This model is configured for 8x SR.")
        self.head = nn.Conv2d(1, channels, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)],
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1),
        )

    def forward(self, x):
        base = F.interpolate(x, scale_factor=8, mode="bicubic", align_corners=False)
        feat = self.head(x)
        feat = self.body(feat) + feat
        residual = self.tail(feat)
        residual = F.interpolate(residual, scale_factor=8, mode="bilinear", align_corners=False)
        return torch.clamp(base + residual, 0.0, 1.0)

if __name__ == "__main__":
    model = AstroNexLunarSRV3()
    x = torch.rand(1,1,64,64)
    with torch.no_grad():
        y = model(x)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
