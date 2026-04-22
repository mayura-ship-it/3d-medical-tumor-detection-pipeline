# 🧠 Medical Imaging 3D Pipeline

### CT Scan → Organ Segmentation → Tumour Detection → 3D Visualization

An end-to-end pipeline for processing 3D medical CT scans using deep learning.
This project performs **automatic organ segmentation, tumour detection, and 3D mesh generation** for visualization and analysis.

---

## 🚀 Pipeline Overview

```text
DICOM CT Scan
      ↓
Preprocessing (HU windowing + resampling)
      ↓
TotalSegmentator (multi-organ segmentation)
      ↓
ROI Extraction
      ↓
nnU-Net Tumour Detection
      ↓
3D Mesh Generation (OBJ / GLB)
```

---

## 📁 Project Structure

```text
medical_pipeline/
├── data/                         # Input CT scan (DICOM format)
│   └── dicom_series/
│
├── nnunet_models/                # Folder for pretrained models (not included)
│   └── nnUNet/
│       └── 3d_fullres/
│           └── Task006_Lung/
│               └── nnUNetTrainer__nnUNetPlansv2.1/
│                   └── (model files go here)
│
├── step1_load_dicom.py           # DICOM → preprocessed NIfTI
├── step2_totalsegmentator.py     # Organ segmentation
├── step3_nnunet_tumour.py        # Tumour detection
├── step4_mesh_generation.py      # 3D mesh generation
├── run_pipeline.py               # Run entire pipeline
│
├── requirements.txt
└── output/                       # Generated automatically
```

---

## ⚠️ Important Notes

* Pretrained nnU-Net model weights are **not included** due to size limits
* Output files are generated after running the pipeline
* Ensure correct folder structure before running

---

## 📥 Model Setup (Required)

You must download pretrained nnU-Net models before running tumour detection.

### 🔹 Steps:

1. Download pretrained models from:
   👉 https://zenodo.org/records/4485926

2. Extract the downloaded files

3. Place them inside the following directory:

```text
nnunet_models/
   nnUNet/
      3d_fullres/
         Task006_Lung/
            nnUNetTrainer__nnUNetPlansv2.1/
               fold_0/
               fold_1/
               fold_2/
               fold_3/
               fold_4/
```

> ⚠️ Ensure all fold folders are present

---

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux / WSL:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Environment Setup

Set nnU-Net results directory:

```bash
# Linux / WSL
export RESULTS_FOLDER=./nnunet_models

# Windows (PowerShell)
setx RESULTS_FOLDER nnunet_models
```

Restart terminal after setting environment variable.

---

## ▶️ Running the Pipeline

### 🔹 Basic Run

```bash
python run_pipeline.py --input ./data/dicom_series
```

---

### 🔹 Optional Arguments

```bash
# Fast mode (lower resolution)
python run_pipeline.py --input ./data/dicom_series --fast

# CPU mode
python run_pipeline.py --input ./data/dicom_series --cpu

# Skip preprocessing or segmentation
python run_pipeline.py --input ./data/dicom_series --skip-preprocess --skip-segment
```

---

## 🧪 Pipeline Breakdown

### 🔹 Step 1 — Preprocessing

* Converts DICOM → NIfTI
* Applies HU windowing
* Resamples to uniform spacing

Output:

```text
output/preprocessed.nii.gz
```

---

### 🔹 Step 2 — Organ Segmentation

* Uses TotalSegmentator
* Automatically segments multiple organs

Output:

```text
output/segmentations/
```

---

### 🔹 Step 3 — Tumour Detection

* Crops organ regions
* Runs nnU-Net inference
* Generates tumour masks

Output:

```text
output/tumours/
output/tumour_findings.json
```

---

### 🔹 Step 4 — Mesh Generation

* Converts segmentation masks into 3D meshes
* Outputs OBJ and GLB formats

Output:

```text
output/meshes/
```

---

## 📊 HU Window Reference

| Window  | Min HU | Max HU | Use Case         |
| ------- | ------ | ------ | ---------------- |
| lung    | -1000  | 400    | Chest CT         |
| abdomen | -150   | 250    | Abdominal organs |
| bone    | -500   | 1800   | Skeletal         |
| brain   | 0      | 80     | Brain CT         |

---

## 🎯 Features

* Automated end-to-end pipeline
* Multi-organ segmentation
* Deep learning-based tumour detection
* 3D mesh generation for visualization
* Modular and extensible design

---

## 🚀 Future Improvements

* Web-based 3D viewer (Three.js)
* VR integration (Unity / WebXR)
* REST API (FastAPI)
* Multi-organ tumour support

---

## 🧑‍💻 Usage Note

This project is intended for **research and educational purposes** in medical imaging and AI.

---

## 📜 License

For academic use only.
