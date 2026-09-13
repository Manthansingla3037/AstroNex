import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Small residual block for lunar image feature extraction.
    """

    def __init__(self, channels=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class AstroNexLunarSR(nn.Module):
    """
    Lightweight 4x Super Resolution model.

    Input:
        1 x 64 x 64

    Output:
        1 x 256 x 256
    """

    def __init__(self, scale=4):
        super().__init__()

        self.scale = scale

        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # Residual feature learning
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        # Feature reconstruction
        self.body_conv = nn.Conv2d(
            32,
            32,
            kernel_size=3,
            padding=1
        )

        # 4x upscaling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                32,
                32 * scale * scale,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True)
        )

        # Final image reconstruction
        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Extract features
        features = self.head(x)

        # Residual learning
        residual = self.residual_blocks(features)

        # Reconstruct features
        residual = self.body_conv(residual)

        # Global residual connection
        features = features + residual

        # Upscale
        features = self.upsample(features)

        # Generate final grayscale HR image
        output = self.output(features)

        # Keep pixel values in [0, 1]
        output = torch.sigmoid(output)

        return output


# ============================================================
# Simple model test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AstroNex Lunar Super Resolution Model Test")
    print("=" * 60)

    device = torch.device("cpu")

    model = AstroNexLunarSR(scale=4)
    model = model.to(device)

    # Simulate one LR lunar image
    test_input = torch.rand(
        1,       # batch
        1,       # grayscale channel
        64,      # height
        64       # width
    ).to(device)

    print("\nInput shape:")
    print(test_input.shape)

    with torch.no_grad():
        output = model(test_input)

    print("\nOutput shape:")
    print(output.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    print("\nOutput range:")
    print(
        output.min().item(),
        "to",
        output.max().item()
    )

    print("\n" + "=" * 60)
    print("MODEL TEST SUCCESSFUL")
    print("=" * 60)