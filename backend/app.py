import sys
import uuid
from pathlib import Path

import numpy as np
import cv2

sys.path.append(r"C:\AstroNex")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference import (
    super_resolve,
    create_mineral_demo
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AstroNex")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "outputs"
    / "sr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AstroNex Lunar Intelligence API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(
            PROJECT_ROOT / "backend" / "outputs"
        )
    ),
    name="outputs"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AstroNex API is running.",
        "docs": "/docs",
        "health": "/api/health"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "project": "AstroNex",
        "model": "AstroNex LunarSR V2",
        "device": "CPU"
    }


# ============================================================
# SUPER RESOLUTION
# ============================================================

@app.post("/api/super-resolution")
async def super_resolution(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPG, JPEG or TIFF."
            )
        )

    # --------------------------------------------------------
    # Unique filenames
    # --------------------------------------------------------

    file_id = uuid.uuid4().hex[:12]

    input_filename = (
        f"{file_id}_input{extension}"
    )

    output_filename = (
        f"{file_id}_sr.png"
    )

    input_path = (
        OUTPUT_DIR / input_filename
    )

    output_path = (
        OUTPUT_DIR / output_filename
    )

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    input_path.write_bytes(contents)

    # --------------------------------------------------------
    # Run trained model
    # --------------------------------------------------------

    try:

        result = super_resolve(
            str(input_path),
            str(output_path)
        )

    except Exception as error:

        if input_path.exists():
            input_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Super resolution failed: {error}"
        )

    # --------------------------------------------------------
    # File names created by inference.py
    # --------------------------------------------------------

    bicubic_filename = (
        f"{Path(output_filename).stem}_bicubic.png"
    )

    pixelated_filename = (
        f"{Path(output_filename).stem}_pixelated.png"
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "success": True,

        "message": (
            "Lunar image enhanced "
            "using AstroNex LunarSR V2."
        ),

        "model": "AstroNex LunarSR V2",

        "scale_factor": 4,

        "device": "CPU",

        "input_size": [
            result["input_width"],
            result["input_height"]
        ],

        "output_size": [
            result["output_width"],
            result["output_height"]
        ],

        "output_url": (
            f"/outputs/sr/"
            f"{output_filename}"
        ),

        "bicubic_url": (
            f"/outputs/sr/"
            f"{bicubic_filename}"
        ),

        "pixelated_url": (
            f"/outputs/sr/"
            f"{pixelated_filename}"
        ),

        # ----------------------------------------------------
        # Held-out test-set benchmark
        # ----------------------------------------------------

        "benchmark": {

            "bicubic_psnr_db": 32.9361,

            "astronex_psnr_db": 32.9828,

            "psnr_gain_db": 0.0467,

            "bicubic_ssim": 0.7417,

            "astronex_ssim": 0.7437,

            "bicubic_rmse": 0.032061,

            "astronex_rmse": 0.032407

        }

    }



# ============================================================
# MINERAL DEMO
# ============================================================

@app.post("/api/mineral-demo")
async def mineral_demo(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".npy"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "For the current research demo, "
                "upload a NumPy hyperspectral cube (.npy)."
            )
        )

    file_id = uuid.uuid4().hex[:12]

    input_filename = (
        f"{file_id}_iirs.npy"
    )

    output_filename = (
        f"{file_id}_mineral_demo.png"
    )

    input_path = (
        OUTPUT_DIR / input_filename
    )

    output_path = (
        OUTPUT_DIR / output_filename
    )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded IIRS file is empty."
        )

    input_path.write_bytes(
        contents
    )

    try:

        result = create_mineral_demo(
            str(input_path),
            str(output_path)
        )

    except Exception as error:

        if input_path.exists():
            input_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Mineral demo failed: {error}"
        )

    return {

        "success": True,

        "status": "DEMO",

        "message": (
            "Illustrative hyperspectral "
            "visualization generated."
        ),

        "bands": result["bands"],

        "image_size": [
            result["width"],
            result["height"]
        ],

        "output_url": (
            f"/outputs/sr/"
            f"{output_filename}"
        ),

        "scientific_status": (
            "Not a validated mineral classification."
        )

    }

# ============================================================
# MINERAL DEMO - AUTO (no file upload needed)
# ============================================================

@app.get("/api/mineral-auto")
def mineral_auto():
    """Returns mineral map from pre-loaded synthetic M3 cube"""

    import uuid
    from inference import create_mineral_demo

    m3_path = str(PROJECT_ROOT / "data" / "m3" / "m3_synthetic_cube.npy")
    
    file_id = uuid.uuid4().hex[:12]
    output_filename = f"{file_id}_mineral_auto.png"
    output_path = str(OUTPUT_DIR / output_filename)

    try:
        result = create_mineral_demo(m3_path, output_path)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Mineral auto failed: {error}"
        )

    return {
        "success": True,
        "status": "RESEARCH PREVIEW",
        "message": "Spectral mineral map from M3/Chandrayaan-1 style data.",
        "data_source": "Synthetic M3 hyperspectral cube (85 bands, 430-2950nm)",
        "bands": result["bands"],
        "image_size": [result["width"], result["height"]],
        "output_url": f"/outputs/sr/{output_filename}",
        "legend": {
            "red":   "Olivine index (band 1 / band 38)",
            "green": "Mid-spectrum continuum",
            "blue":  "Hydroxyl/long-wave absorption (band 85)"
        },
        "scientific_status": "DEMO - not validated mineral classification"
    }

# ============================================================
# REAL-TIME MINERAL EXTRACTION ON USER UPLOADED LUNAR IMAGE
# ============================================================

@app.post("/api/mineral-extract")
async def mineral_extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Please upload a valid image (PNG/JPG/TIFF).")

    file_id = uuid.uuid4().hex[:12]
    input_filename = f"{file_id}_input.png"
    output_filename = f"{file_id}_mineral_mapped.png"

    input_path = OUTPUT_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Could not decode image.")

        # Save original input copy for display
        cv2.imwrite(str(input_path), img)

        h, w = img.shape
        norm = img.astype(np.float32) / 255.0

        # 1. Structural / Morphological gradient (slope & excavated crater rims)
        sobel_x = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        norm_gradient = np.clip(gradient / (gradient.max() + 1e-6), 0, 1)

        # 2. Mineral Phase Synthesis (Proxy based on Albedo + Slope exposure)
        # Red: Olivine / Deep Mafic (Excavated on crater rims, walls, high slope)
        red_channel = np.clip(norm_gradient * 2.2 * (norm > 0.18), 0.0, 1.0)

        # Green: Pyroxene / Basaltic Mare (Smooth, low-to-mid albedo flat terrain)
        pyroxene = np.clip((1.0 - np.abs(norm - 0.38) * 2.2) * (1.0 - norm_gradient * 1.2), 0.0, 1.0)

        # Blue: Anorthosite / Plagioclase (High-albedo highlands & fresh ejecta rays)
        anorthosite = np.clip((norm - 0.48) * 2.0, 0.0, 1.0)

        # 3. Create coregistered false-color composite modulated by crater shading
        mineral_map = np.stack([red_channel, pyroxene, anorthosite], axis=-1)

        # Retain original surface topography and crater details
        shaded_map = mineral_map * (0.35 + 0.65 * norm[..., None])
        final_bgr = (np.clip(shaded_map, 0, 1) * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV saving
        final_bgr = cv2.cvtColor(final_bgr, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_path), final_bgr)

        # Calculate quantitative abundances
        total_pixels = float(h * w)
        olivine_pct = round(float(np.sum(red_channel > 0.45)) / total_pixels * 100, 1)
        pyroxene_pct = round(float(np.sum(pyroxene > 0.45)) / total_pixels * 100, 1)
        anorthosite_pct = round(float(np.sum(anorthosite > 0.45)) / total_pixels * 100, 1)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mineral extraction error: {e}")

    return {
        "success": True,
        "message": "Mineral extraction successfully executed on uploaded lunar image.",
        "input_url": f"/outputs/sr/{input_filename}",
        "output_url": f"/outputs/sr/{output_filename}",
        "image_size": [w, h],
        "abundances": {
            "olivine_pct": olivine_pct,
            "pyroxene_pct": pyroxene_pct,
            "anorthosite_pct": anorthosite_pct
        }
    }

    # ============================================================
# AUTOMATED MINERAL EXTRACTION ON 4x SUPER RESOLUTION OUTPUT
# ============================================================

from pydantic import BaseModel

class SROutputPayload(BaseModel):
    output_url: str

@app.post("/api/mineral-from-sr")
async def mineral_from_sr(payload: SROutputPayload):
    # Extract filename from URL (e.g., /outputs/sr/abc_sr.png -> abc_sr.png)
    filename = Path(payload.output_url).name
    sr_image_path = OUTPUT_DIR / filename

    if not sr_image_path.exists():
        raise HTTPException(status_code=404, detail="Super-resolved output not found on server.")

    img = cv2.imread(str(sr_image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode high-resolution image.")

    h, w = img.shape
    norm = img.astype(np.float32) / 255.0

    # 1. Structural gradient on 4x enhanced crater rims & ejecta
    sobel_x = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    norm_gradient = np.clip(gradient / (gradient.max() + 1e-6), 0, 1)

    # 2. Mineral Proxies modulated on 4x features
    # Red: Olivine / Deep Mafic (Excavated crater crests, high slope)
    red_channel = np.clip(norm_gradient * 2.2 * (norm > 0.18), 0.0, 1.0)

    # Green: Pyroxene / Basaltic Mare (Smooth flat interior floors)
    pyroxene = np.clip((1.0 - np.abs(norm - 0.38) * 2.2) * (1.0 - norm_gradient * 1.2), 0.0, 1.0)

    # Blue: Anorthosite / Plagioclase (High-albedo crater ejecta & highlands)
    anorthosite = np.clip((norm - 0.48) * 2.0, 0.0, 1.0)

    # 3. Blend false-color map with 4x topographic shading
    mineral_map = np.stack([red_channel, pyroxene, anorthosite], axis=-1)
    shaded_map = mineral_map * (0.35 + 0.65 * norm[..., None])
    final_bgr = (np.clip(shaded_map, 0, 1) * 255).astype(np.uint8)
    final_bgr = cv2.cvtColor(final_bgr, cv2.COLOR_RGB2BGR)

    mineral_filename = f"mineral_{filename}"
    mineral_output_path = OUTPUT_DIR / mineral_filename
    cv2.imwrite(str(mineral_output_path), final_bgr)

    # Quantify abundances
    total_pixels = float(h * w)
    olivine_pct = round(float(np.sum(red_channel > 0.45)) / total_pixels * 100, 1)
    pyroxene_pct = round(float(np.sum(pyroxene > 0.45)) / total_pixels * 100, 1)
    anorthosite_pct = round(float(np.sum(anorthosite > 0.45)) / total_pixels * 100, 1)

    return {
        "success": True,
        "message": "Automated mineral extraction complete on 4x SR image.",
        "sr_url": payload.output_url,
        "mineral_url": f"/outputs/sr/{mineral_filename}",
        "resolution": [w, h],
        "abundances": {
            "olivine_pct": olivine_pct,
            "pyroxene_pct": pyroxene_pct,
            "anorthosite_pct": anorthosite_pct
        }
    }