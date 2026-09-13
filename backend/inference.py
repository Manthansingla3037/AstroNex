import sys

sys.path.append(r"C:\AstroNex")

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.model_v2 import AstroNexLunarSRV2


PROJECT_ROOT = Path(r"C:\AstroNex")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lunar_sr_v2_best.pth"
)

DEVICE = torch.device("cpu")

SCALE = 4

PATCH_SIZE = 64


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = AstroNexLunarSRV2(scale=SCALE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# ============================================================
# SUPER RESOLUTION
# ============================================================

def super_resolve(input_path: str, output_path: str):

    input_path = Path(input_path)
    output_path = Path(output_path)

    image = Image.open(
        input_path
    ).convert("L")

    original_width, original_height = image.size

    # Keep original low-resolution image
    original_lr = image.copy()

    # Make dimensions compatible with model
    new_width = max(
        PATCH_SIZE,
        (original_width // PATCH_SIZE) * PATCH_SIZE
    )

    new_height = max(
        PATCH_SIZE,
        (original_height // PATCH_SIZE) * PATCH_SIZE
    )

    if (
        new_width != original_width
        or new_height != original_height
    ):
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BICUBIC
        )

    image_array = (
        np.asarray(
            image,
            dtype=np.float32
        ) / 255.0
    )

    height, width = image_array.shape

    output_height = height * SCALE
    output_width = width * SCALE

    output = np.zeros(
        (output_height, output_width),
        dtype=np.float32
    )

    # ========================================================
    # AI inference patch by patch
    # ========================================================

    with torch.no_grad():

        for y in range(
            0,
            height,
            PATCH_SIZE
        ):

            for x in range(
                0,
                width,
                PATCH_SIZE
            ):

                patch = image_array[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                if patch.shape != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    continue

                tensor = torch.from_numpy(
                    patch
                ).float()

                tensor = tensor.unsqueeze(0)
                tensor = tensor.unsqueeze(0)

                prediction = model(
                    tensor.to(DEVICE)
                )

                prediction = (
                    prediction
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                output_y = y * SCALE
                output_x = x * SCALE

                output[
                    output_y:
                    output_y + PATCH_SIZE * SCALE,
                    output_x:
                    output_x + PATCH_SIZE * SCALE
                ] = prediction

    # ========================================================
    # AI image
    # ========================================================

    output = np.clip(
        output,
        0.0,
        1.0
    )

    ai_uint8 = np.uint8(
        output * 255.0
    )

    ai_image = Image.fromarray(
        ai_uint8
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ai_image.save(
        output_path
    )

    # ========================================================
    # Bicubic baseline
    # ========================================================

    bicubic_image = original_lr.resize(
        (output_width, output_height),
        Image.Resampling.BICUBIC
    )

    bicubic_path = output_path.with_name(
        output_path.stem + "_bicubic.png"
    )

    bicubic_image.save(
        bicubic_path
    )

    # ========================================================
    # Pixelated version for visual demonstration
    # ========================================================

    pixelated_image = original_lr.resize(
        (output_width, output_height),
        Image.Resampling.NEAREST
    )

    pixelated_path = output_path.with_name(
        output_path.stem + "_pixelated.png"
    )

    pixelated_image.save(
        pixelated_path
    )

    return {
        "input_width": original_width,
        "input_height": original_height,

        "output_width": ai_image.width,
        "output_height": ai_image.height,

        "output_path": str(output_path),

        "bicubic_path": str(bicubic_path),

        "pixelated_path": str(pixelated_path)
    }










def create_mineral_demo(
    input_path: str,
    output_path: str
):
    """
    Creates an illustrative false-color visualization
    from an IIRS-style NumPy hyperspectral cube.

    This is NOT a validated mineral classifier.
    """

    cube = np.load(
        input_path,
        allow_pickle=False
    )

    if cube.ndim != 3:
        raise ValueError(
            "Expected a 3D hyperspectral cube."
        )

    # --------------------------------------------------------
    # Support either:
    # (bands, height, width)
    # or
    # (height, width, bands)
    # --------------------------------------------------------

    if cube.shape[0] < cube.shape[-1]:
        bands, height, width = cube.shape

        band_cube = cube

    else:
        height, width, bands = cube.shape

        band_cube = np.transpose(
            cube,
            (2, 0, 1)
        )

    if bands < 3:
        raise ValueError(
            "Hyperspectral cube needs at least 3 bands."
        )

    # --------------------------------------------------------
    # Select three bands for visualization
    # --------------------------------------------------------

    b1 = band_cube[0].astype(np.float32)
    b2 = band_cube[bands // 2].astype(np.float32)
    b3 = band_cube[-1].astype(np.float32)

    def normalize_band(band):

        band_min = np.nanpercentile(
            band,
            2
        )

        band_max = np.nanpercentile(
            band,
            98
        )

        if band_max <= band_min:
            return np.zeros_like(
                band,
                dtype=np.float32
            )

        result = (
            band - band_min
        ) / (
            band_max - band_min
        )

        return np.clip(
            result,
            0,
            1
        )

    r = normalize_band(b1)
    g = normalize_band(b2)
    b = normalize_band(b3)

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    rgb = np.uint8(
        rgb * 255
    )

    # --------------------------------------------------------
    # Enhance visualization
    # --------------------------------------------------------

    rgb = cv2.GaussianBlur(
        rgb,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    result = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        result
    )

    return {
        "width": width,
        "height": height,
        "bands": bands,
        "output_path": str(output_path)
    }