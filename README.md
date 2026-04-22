# Medical Imaging 3D Pipeline

DICOM CT scans → full organ segmentation → tumour detection → 3D meshes → VR

---

## File Structure

```
medical_pipeline/
├── step1_load_dicom.py        # DICOM → preprocessed NIfTI
├── step2_totalsegmentator.py  # Full-body organ segmentation (104 organs)
├── step3_nnunet_tumour.py     # Tumour detection per organ
├── step4_mesh_generation.py   # NIfTI masks → OBJ + GLB meshes
├── run_pipeline.py            # Master runner (all steps)
├── requirements.txt
└── output/
    ├── preprocessed.nii.gz
    ├── organ_colors.json
    ├── tumour_findings.json
    ├── segmentations/          ← one .nii.gz per organ (104 files)
    ├── tumours/                ← tumour masks + predictions
    └── meshes/
        ├── manifest.json       ← used by the 3D viewer
        ├── liver.obj / .glb
        ├── lungs.obj / .glb
        ├── lung_tumour.obj / .glb   ← only if tumour detected
        └── ...
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set nnU-Net environment variables

```bash
export nnUNet_raw=/path/to/nnunet_raw
export nnUNet_preprocessed=/path/to/nnunet_preprocessed
export nnUNet_results=/path/to/nnunet_models
```

### 3. Download nnU-Net model weights

You currently have lung nodule weights. Download others as needed:

```bash
# Lung nodule (Task006) — already have this
nnUNetv2_download_pretrained_model_by_url Task006_Lung

# Liver tumour
nnUNetv2_download_pretrained_model_by_url Task003_Liver

# Kidney tumour
nnUNetv2_download_pretrained_model_by_url Task064_KiTS19

# Pancreas tumour
nnUNetv2_download_pretrained_model_by_url Task007_Pancreas

# Colon cancer
nnUNetv2_download_pretrained_model_by_url Task010_Colon
```

After downloading, set `"enabled": True` for those organs in `step3_nnunet_tumour.py → TASK_REGISTRY`.

### 4. Run the pipeline

```bash
# Chest CT scan (lung window)
python run_pipeline.py --input ./data/dicom_series --window lung

# Abdominal CT scan
python run_pipeline.py --input ./data/dicom_series --window abdomen

# From ZIP file
python run_pipeline.py --input ./data/scan.zip

# Fast mode (lower res, ~5x faster) — good for testing
python run_pipeline.py --input ./data/dicom_series --fast

# No GPU
python run_pipeline.py --input ./data/dicom_series --cpu

# Skip steps already done
python run_pipeline.py --input ./data/dicom_series --skip-preprocess --skip-segment
```

---

## Step-by-Step Guide

### Step 1 — DICOM Preprocessing (`step1_load_dicom.py`)

- Accepts: DICOM folder, ZIP of DICOM folder, NIfTI file
- Sorts slices by DICOM position
- Applies HU windowing (lung: -1000 to 400)
- Resamples to 1mm isotropic voxels
- Fixes orientation to RAS+
- Output: `./output/preprocessed.nii.gz`

### Step 2 — TotalSegmentator (`step2_totalsegmentator.py`)

- Segments ~104 anatomical structures automatically
- No manual input needed
- GPU: ~1 min per scan | CPU: ~20-30 min
- Each organ gets a unique colour defined in `ORGAN_COLORS`
- Output: `./output/segmentations/<organ>.nii.gz` (one per organ)
- Also writes `organ_colors.json` with volumes + colours

### Step 3 — nnU-Net Tumour Detection (`step3_nnunet_tumour.py`)

- Uses TotalSegmentator masks to crop organ ROIs
- Runs nnU-Net inference on each crop
- Pastes tumour mask back into full image space
- Only enabled organs (with downloaded weights) are processed
- Output: `./output/tumours/<organ>_tumour.nii.gz`
- Also writes `tumour_findings.json` with volumes + centroids

### Step 4 — Mesh Generation (`step4_mesh_generation.py`)

- Marching cubes on each NIfTI mask (VTK primary, scikit-image fallback)
- Laplacian smoothing + mesh decimation (keeps ~30% triangles)
- Saves OBJ (with MTL colours) + GLB (for Three.js / VR)
- Output: `./output/meshes/`
- Also writes `manifest.json` — the 3D viewer loads this

---

## HU Window Reference

| Window   | Min HU | Max HU | Use case                    |
|----------|--------|--------|-----------------------------|
| lung     | -1000  | 400    | Chest CT, lung parenchyma   |
| abdomen  | -150   | 250    | Abdominal organs, liver etc |
| bone     | -500   | 1800   | Skeletal structures         |
| brain    | 0      | 80     | Head CT, brain tissue       |

---

## nnU-Net Task Registry

| Organ     | Task ID   | Labels detected               | Enabled by default |
|-----------|-----------|-------------------------------|--------------------|
| lung      | Task006   | Lung nodule / mass            | ✓ (you have this)  |
| liver     | Task003   | Liver tumour / HCC            | ✗                  |
| kidney    | Task064   | RCC / cyst                    | ✗                  |
| pancreas  | Task007   | Pancreatic mass / PDAC        | ✗                  |
| colon     | Task010   | Colorectal cancer             | ✗                  |
| brain     | Task001   | Glioma (needs MRI, not CT)    | ✗                  |

---

## Next Steps — 3D Viewer & VR

After running the pipeline:

1. Load `output/meshes/manifest.json` in your Three.js viewer
2. Each `organ.glb` file is a separate toggleable mesh
3. Tumour meshes are flagged `"is_tumour": true` in manifest
4. For VR: import the same GLB files into Unity/Unreal or use WebXR

Ask Claude to generate:
- `step5_web_server.py` — Flask server to serve meshes + manifest
- `viewer/` — Three.js viewer with organ toggle checklist
- `vr/` — WebXR viewer or Unity C# importer script
