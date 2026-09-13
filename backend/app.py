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

            "bicubic_psnr_db": 35.3105,

            "astronex_psnr_db": 36.5302,

            "psnr_gain_db": 1.2197,

            "bicubic_ssim": 0.8911,

            "astronex_ssim": 0.9075,

            "bicubic_rmse": 0.017328,

            "astronex_rmse": 0.015083

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