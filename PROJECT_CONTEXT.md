# AstroNex Project Context

## Project Identity

**Project:** AstroNex – AI-Powered Lunar Surface Intelligence

**Use case:** Smart India Hackathon / Space Technology

AstroNex is a software platform for lunar image enhancement and spectral/mineral research using Chandrayaan-2 data.

## Problem

Lunar observations from different sensors have different spatial and spectral characteristics.

The project aims to:
- improve usable spatial detail of lower-resolution lunar imagery,
- preserve scientific structure instead of merely enlarging pixels,
- exploit high-resolution OHRC imagery for reference/validation,
- use IIRS hyperspectral information for spectral analysis,
- eventually fuse spatial and spectral information.

## Chandrayaan-2 Data Roles

### TMC-2
Role:
- relatively lower-resolution lunar imaging input,
- target source for future real cross-sensor super-resolution training,
- approximately 5 m class spatial sampling in the project context.

### OHRC
Role:
- very high-resolution optical reference/target,
- used for validation and, when spatially co-located, paired training,
- approximately 0.25 m class spatial sampling in the project context.

A real TMC-2 → OHRC training pair must have a verified common footprint and registration.

### IIRS
Role:
- hyperspectral spectral-information source,
- intended for mineral/composition or hydration-related analysis when compatible coverage and processing are available,
- not the high-resolution spatial target for the SR model.

## Current OHRC Prototype Dataset

The project has used real OHRC imagery to create training/validation/test patches.

Current V2 dataset-generation concept:
1. extract 256×256 OHRC HR patches,
2. apply Gaussian blur,
3. downsample to 64×64,
4. apply small synthetic noise,
5. use the degraded 64×64 image as LR input,
6. retain the 256×256 image as HR target.

This is a controlled synthetic-degradation experiment.

## Current SR Model

File:

```text
training/model_v2.py
```

Class:

```text
AstroNexLunarSRV2
```

Architecture:
- 48 channels,
- 6 residual blocks,
- 4× reconstruction,
- bicubic base,
- learned residual correction.

Training:

```text
training/train_v2.py
```

Evaluation:

```text
evaluation/evaluate_v2.py
```

Checkpoint:

```text
models/lunar_sr_v2_best.pth
```

## Historical V1

There is also an older model:

```text
models/lunar_sr_best.pth
```

The V1 experiment performed worse than bicubic on its held-out test set and should be treated as a historical baseline, not the preferred model.

## V2 Results

Synthetic-degradation held-out test set:

```text
                 Bicubic      AstroNex V2
PSNR             35.3105      36.5302 dB
SSIM              0.8911       0.9075
RMSE              0.017328     0.015083
```

Observed V2 change relative to bicubic:

```text
PSNR improvement ≈ +1.2197 dB
SSIM improvement ≈ +0.0164
RMSE reduction  ≈ 0.002244
```

These values do not establish performance on real TMC-2 → OHRC cross-sensor reconstruction.

## Real Data Compatibility Lessons

Product metadata must be checked before creating training pairs.

A visually overlapping MapBrowse selection is not sufficient evidence of actual product overlap.

The valid approach is:
1. inspect the product XML,
2. verify sensor,
3. verify observation region,
4. verify footprint coordinates,
5. verify temporal/spatial compatibility,
6. register the imagery,
7. only then create paired patches.

## Backend

Main files:

```text
backend/app.py
backend/inference.py
```

The backend:
- exposes FastAPI endpoints,
- performs tiled super-resolution inference,
- provides bicubic and pixelated comparisons,
- serves output images,
- exposes benchmark information,
- includes a mineral demo endpoint/function.

The mineral demo is not a validated IIRS mineral classifier.

## Frontend

Current frontend:

```text
frontend_new/
```

Technology:
- React
- Vite
- CSS

Current intended dashboard:
- Super Resolution module,
- 64×64 input visualization,
- 4× AI reconstruction,
- fixed-size comparison slider,
- PSNR/SSIM/RMSE benchmark,
- Mineral Extraction research preview,
- realistic sample lunar imagery,
- false-color mineral-style demonstration,
- explicit DEMO / RESEARCH PREVIEW labeling where appropriate.

The older `frontend/` directory is not the active frontend.

## Reproducibility

### Windows setup

```powershell
cd C:\AstroNex
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### Generate V2 dataset

```powershell
python preprocessing\create_dataset_v2.py
```

### Train V2

```powershell
python training\train_v2.py
```

### Evaluate

```powershell
python evaluation\evaluate_v2.py
```

### Run backend

```powershell
cd backend
python -m uvicorn app:app --reload
```

### Run frontend

In another PowerShell:

```powershell
cd C:\AstroNex\frontend_new
npm install
npm run dev
```

## Repository Data Policy

The repository should contain:
- source code,
- scripts,
- documentation,
- small demo assets,
- trained model checkpoints when reasonably sized,
- evaluation images/results.

It should not contain:
- `.venv`,
- `node_modules`,
- Python cache,
- large raw lunar datasets,
- multi-gigabyte `.img` products,
- large hyperspectral cubes,
- generated training data that can be reproduced from source data.

## Future Research Direction

The intended final research pipeline is:

```text
                TMC-2
                  ↓
             preprocessing
                  ↓
          co-location / registration
                  ↓
          paired TMC-2 → OHRC patches
                  ↓
            Cross-sensor SR
                  ↓
          enhanced lunar image
                  ↓
          ┌──── validation ────┐
          ↓                    ↓
        OHRC                  metrics

IIRS hyperspectral data
          ↓
 spectral preprocessing
          ↓
 spectral representation
          ↓
 spatial-spectral fusion
          ↓
 potential composition/mineral products
          ↓
 confidence / uncertainty
```

The final system should be evaluated on unseen lunar regions and should document dataset splits, registration quality, model configuration, metrics, and limitations.

## Important Scientific Boundaries

AstroNex should distinguish:
- image enhancement from factual recovery,
- synthetic degradation from real sensor degradation,
- research demonstrations from validated mineral mapping,
- visual similarity from scientifically registered correspondence,
- model confidence from physical truth.

A future model should not be promoted as scientifically stronger solely because it produces sharper-looking images. Quantitative validation and correct cross-sensor pairing are essential.
