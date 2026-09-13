# AstroNex AI Development Instructions

## Purpose

AstroNex is an AI-powered lunar surface intelligence project for Smart India Hackathon and future research development.

The long-term objective is to:
- enhance lunar imagery using super-resolution,
- use Chandrayaan-2 TMC-2 and OHRC observations,
- use Chandrayaan-2 IIRS hyperspectral data for spectral analysis,
- combine compatible spatial and spectral information,
- generate potential mineral/composition or hydration-related maps only when the data support those claims,
- report confidence/uncertainty and quantitative validation.

## Repository Structure

```text
AstroNex/
├── backend/          # FastAPI API and inference
├── frontend_new/     # Current React/Vite dashboard
├── preprocessing/    # Dataset generation and preprocessing
├── training/         # Dataset loader, model definitions, training scripts
├── models/           # Trained model checkpoints
├── evaluation/       # Evaluation scripts and results
├── data/             # Dataset instructions and local data
├── README.md
├── AI_INSTRUCTIONS.md
└── PROJECT_CONTEXT.md
```

`frontend_new/` is the current frontend. The older `frontend/` directory is not the active frontend.

## Current Prototype

The current prototype includes:
- OHRC-based image super-resolution experiments,
- a synthetic-degradation training pipeline,
- AstroNex Lunar SR V2,
- FastAPI inference,
- a React/Vite dashboard,
- PSNR/SSIM/RMSE evaluation,
- a research-preview mineral/spectral demonstration.

## Current Super-Resolution Model

Main model file:

```text
training/model_v2.py
```

Model class:

```text
AstroNexLunarSRV2
```

Current architecture:
- 48 feature channels,
- 6 residual blocks,
- 4× reconstruction,
- bicubic upsampling as the base,
- CNN-predicted residual details added to the bicubic result,
- output clamped to the valid image range.

Checkpoint:

```text
models/lunar_sr_v2_best.pth
```

## Current Training Pipeline

Generate the V2 dataset:

```powershell
python preprocessing\create_dataset_v2.py
```

Train:

```powershell
python training\train_v2.py
```

Evaluate:

```powershell
python evaluation\evaluate_v2.py
```

The V2 experiment uses OHRC-derived high-resolution patches with synthetic degradation. It is not a directly registered real TMC-2 → OHRC training dataset.

## Current Benchmark

Held-out test-set benchmark for the synthetic-degradation experiment:

| Metric | Bicubic | AstroNex V2 |
|---|---:|---:|
| PSNR | 35.3105 dB | 36.5302 dB |
| SSIM | 0.8911 | 0.9075 |
| RMSE | 0.017328 | 0.015083 |

Do not present these numbers as real cross-sensor TMC-2 → OHRC performance.

## Scientific Rules

When modifying the project:

1. Preserve working functionality unless a change is technically justified.
2. Do not invent or exaggerate scientific results.
3. Clearly distinguish prototype/demo features from validated research outputs.
4. Keep train/validation/test separation.
5. Check for data leakage.
6. Report quantitative metrics after model changes.
7. Compare new models against bicubic and the existing AstroNex V2 baseline.
8. Prefer real, correctly co-located Chandrayaan-2 observations for final cross-sensor training.
9. Verify product XML metadata and geographic footprint before assuming that two products overlap.
10. Do not make mineral/composition claims without appropriate spectral data and validation.
11. Do not use IIRS data as the spatial high-resolution target for super-resolution; its main role is spectral information.
12. Preserve PDS4/XML metadata and scientific provenance where applicable.

## Desired Future Model

Preferred future SR training path:

```text
Real TMC-2 (~5 m)
        +
Real OHRC (~0.25 m)
        ↓
Common geographic footprint
        ↓
Geometric registration
        ↓
Paired patches
        ↓
Cross-sensor SR model
        ↓
Enhanced lunar image
```

Spectral analysis:

```text
IIRS hyperspectral cube
        ↓
Spectral preprocessing
        ↓
Spectral features
        ↓
Spectral model
```

Future fusion:

```text
Spatial features (TMC-2/OHRC)
        +
Spectral features (IIRS)
        ↓
Spatial-spectral fusion
        ↓
Potential composition/mineral map
        ↓
Confidence / uncertainty
```

## Data Policy

Large scientific datasets are intentionally excluded from GitHub.

Do not commit:
- multi-gigabyte `.img` products,
- raw/real data directories,
- large `.npy`/`.npz` hyperspectral cubes,
- temporary generated datasets.

Expected local data locations are documented in:

```text
data/README.md
```

## How an AI Should Work on This Repository

When a user asks to upgrade AstroNex, first read:
1. `README.md`
2. `AI_INSTRUCTIONS.md`
3. `PROJECT_CONTEXT.md`
4. relevant source files under `training/`, `preprocessing/`, `evaluation/`, `backend/`, and `frontend_new/`

Before changing code, explain:
- current architecture,
- current data flow,
- current model,
- current limitations,
- proposed modifications,
- files that will change,
- expected validation strategy.

During implementation:
- modify the smallest reasonable set of files,
- preserve APIs and frontend behavior unless requested otherwise,
- add comments where scientific logic is non-obvious,
- keep paths configurable where practical,
- avoid hard-coded machine-specific paths.

After implementation provide:
- changed-file summary,
- exact Windows PowerShell commands,
- training command,
- evaluation command,
- expected output files,
- metrics to compare against the baseline,
- known limitations.

## Do Not Assume Missing Data

If TMC-2, OHRC, or IIRS files are unavailable, do not fabricate them.

If real IIRS data are unavailable:
- keep spectral/mineral functionality explicitly labeled as DEMO or RESEARCH PREVIEW,
- do not claim validated mineral classification.

If a TMC-2 candidate has not been verified against OHRC metadata:
- do not assume it is a valid paired training product.

## Preferred Development Goal

The strongest upgrade path is not simply making the CNN larger.

Prioritize:
1. correct multi-source data pairing,
2. registration and geolocation accuracy,
3. realistic sensor/degradation modeling,
4. leakage-free evaluation,
5. cross-sensor model design,
6. uncertainty/confidence reporting,
7. only then model complexity and optimization.
