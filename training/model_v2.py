import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=48):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class AstroNexLunarSRV2(nn.Module):
    """
    Residual Super Resolution model.

    Input:
        1 x 64 x 64

    Output:
        1 x 256 x 256

    The model uses Bicubic upscaling as a base image
    and learns the residual high-frequency details.
    """

    def __init__(self, scale=4):
        super().__init__()

        self.scale = scale

        # Feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Deeper residual feature learning
        self.body = nn.Sequential(
            ResidualBlock(48),
            ResidualBlock(48),
            ResidualBlock(48),
            ResidualBlock(48),
            ResidualBlock(48),
            ResidualBlock(48)
        )

        self.body_conv = nn.Conv2d(
            48,
            48,
            3,
            padding=1
        )

        # Learn residual information at LR resolution
        self.residual_head = nn.Conv2d(
            48,
            1,
            3,
            padding=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = self.head(x)

        residual = self.body(features)

        residual = self.body_conv(residual)

        residual = features + residual

        # ----------------------------------------------------
        # Predict missing details
        # ----------------------------------------------------

        detail = self.residual_head(residual)

        # ----------------------------------------------------
        # Upscale learned details
        # ----------------------------------------------------

        detail = F.interpolate(
            detail,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False
        )

        # Keep the predicted correction relatively small.
        detail = 0.25 * torch.tanh(detail)

        # ----------------------------------------------------
        # Bicubic + AI residual
        # ----------------------------------------------------

        output = bicubic + detail

        # Valid image range
        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AstroNex Lunar Super Resolution V2")
    print("=" * 70)

    model = AstroNexLunarSRV2(scale=4)

    test_input = torch.rand(
        1,
        1,
        64,
        64
    )

    print("\nInput:")
    print(test_input.shape)

    with torch.no_grad():
        output = model(test_input)

    print("\nOutput:")
    print(output.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", parameters)

    print("\nOutput range:")
    print(
        output.min().item(),
        "to",
        output.max().item()
    )

    print("\n" + "=" * 70)
    print("V2 MODEL TEST SUCCESSFUL")
    print("=" * 70)