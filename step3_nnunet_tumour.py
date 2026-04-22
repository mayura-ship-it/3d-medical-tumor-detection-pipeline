"""
STEP 3 — nnU-Net Tumour Detection
===================================

YOUR SETUP (confirmed from screenshot + pip output):
  - nnunet v1 installed  (1.7.1)
  - nnunetv2 ALSO installed  ← this was causing detect_nnunet_version() to return "v2"
  - Weights folder structure:

    nnunet_models/
      nnUNet/                          ← this middle folder was being missed
        3d_fullres/
          Task001_BrainTumour/
            nnUNetTrainer_nnUNetPlan.../
              fold_0/ fold_1/ ... fold_4/
              plans.pkl
              postprocessing.json
          Task006_Lung/
            nnUNetTrainer_nnUNetPlan.../
              fold_0/ ... fold_4/
              plans.pkl
              postprocessing.json

  This is the STANDARD nnU-Net v1 layout.
  The RESULTS_FOLDER env var should point to: nnunet_models/
  nnU-Net v1 then internally looks for: <RESULTS_FOLDER>/nnUNet/3d_fullres/TaskXXX/

FIXES in this version:
  [FIX 1] Path detection now scans for the nnUNet/3d_fullres/TaskXXX/ subfolder
           pattern — the previous code looked in the root which was wrong.
  [FIX 2] detect_nnunet_version() now prefers v1 when both are installed,
           because your weights are v1 format.
  [FIX 3] detect_weight_version() now correctly handles the v1 subfolder layout.
  [FIX 4] RESULTS_FOLDER is now set to the actual models root (not a subfolder).
  [FIX 5] run_nnunet_v1() now also tries the Python module fallback if the
           nnUNet_predict console script is not on PATH.
  [FIX 6] Trainer name matched to what's in YOUR folder:
           "nnUNetTrainer_nnUNetPlan..." not "nnUNetTrainerV2__nnUNetPlansv2.1"
           The code now auto-reads the trainer name from the folder itself.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONFIG — SET THIS TO YOUR ACTUAL PATH
# ─────────────────────────────────────────────────────────────
INPUT_NIFTI       = "./output/preprocessed.nii.gz"
SEGMENTATION_DIR  = "./output/segmentations"
TUMOUR_OUTPUT_DIR = "./output/tumours"
FINDINGS_JSON     = "./output/tumour_findings.json"

# This should be the folder that CONTAINS the "nnUNet" subfolder.
# From your screenshot: E:\medical_new_claude\nnunet_models
# The code will look for: <NNUNET_MODELS_DIR>/nnUNet/3d_fullres/Task006_Lung/
NNUNET_MODELS_DIR = os.environ.get(
    "RESULTS_FOLDER",
    os.environ.get("nnUNet_results", "./nnunet_models")
)
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────────────────────
TASK_REGISTRY = {
    "lung": {
        "task_name"    : "Task006_Lung",
        "dataset_id_v2": 6,
        "config"       : "3d_fullres",
        "fold"         : "all",
        # TotalSegmentator (ml=False) writes these individual lobe files.
        # Run: python step2_totalsegmentator.py --debug   to confirm exact filenames.
        "organ_masks"  : [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ],
        "labels"       : {0: "background", 1: "lung", 2: "nodule"},
        "tumour_label" : 2,
        "colour_rgb"   : [255, 50, 50],
        "enabled"      : True,
        "description"  : "Lung nodule / mass detection",
    },
    "brain": {
        "task_name"    : "Task001_BrainTumour",
        "dataset_id_v2": 1,
        "config"       : "3d_fullres",
        "fold"         : "all",
        "organ_masks"  : ["brain"],
        "labels"       : {0: "background", 1: "edema", 2: "enhancing", 3: "necrosis"},
        "tumour_label" : 2,
        "colour_rgb"   : [255, 0, 200],
        "enabled"      : False,   # requires multi-modal MRI, not standard CT
        "description"  : "Brain tumour glioma (needs MRI not CT)",
    },
    "liver": {
        "task_name"    : "Task003_Liver",
        "dataset_id_v2": 3,
        "config"       : "3d_fullres",
        "fold"         : "all",
        "organ_masks"  : ["liver"],
        "labels"       : {0: "background", 1: "liver", 2: "liver_tumour"},
        "tumour_label" : 2,
        "colour_rgb"   : [255, 100, 0],
        "enabled"      : False,
        "description"  : "Liver tumour / HCC / metastasis",
    },
    "kidney": {
        "task_name"    : "Task064_KiTS19",
        "dataset_id_v2": 64,
        "config"       : "3d_fullres",
        "fold"         : "all",
        "organ_masks"  : ["kidney_left", "kidney_right"],
        "labels"       : {0: "background", 1: "kidney", 2: "tumour", 3: "cyst"},
        "tumour_label" : 2,
        "colour_rgb"   : [255, 80, 80],
        "enabled"      : False,
        "description"  : "Kidney tumour / RCC",
    },
    "pancreas": {
        "task_name"    : "Task007_Pancreas",
        "dataset_id_v2": 7,
        "config"       : "3d_fullres",
        "fold"         : "all",
        "organ_masks"  : ["pancreas"],
        "labels"       : {0: "background", 1: "pancreas", 2: "mass"},
        "tumour_label" : 2,
        "colour_rgb"   : [255, 200, 0],
        "enabled"      : False,
        "description"  : "Pancreatic mass / PDAC",
    },
    "colon": {
        "task_name"    : "Task010_Colon",
        "dataset_id_v2": 10,
        "config"       : "3d_fullres",
        "fold"         : "all",
        "organ_masks"  : ["colon"],
        "labels"       : {0: "background", 1: "colon_cancer"},
        "tumour_label" : 1,
        "colour_rgb"   : [200, 100, 50],
        "enabled"      : False,
        "description"  : "Colon / colorectal cancer",
    },
}


# ─────────────────────────────────────────────────────────────
# VERSION + PATH DETECTION
# ─────────────────────────────────────────────────────────────

def detect_nnunet_version() -> str:
    """
    Return 'v1', 'v2', or 'none'.

    IMPORTANT: If BOTH are installed (which is your case), prefer v1
    because your weights are v1 format. You can uninstall nnunetv2 to
    avoid confusion: pip uninstall nnunetv2
    """
    has_v1 = False
    has_v2 = False

    try:
        import nnunet
        has_v1 = True
    except ImportError:
        pass

    try:
        import nnunetv2
        has_v2 = True
    except ImportError:
        pass

    if has_v1 and has_v2:
        print("      WARNING: Both nnunet (v1) and nnunetv2 are installed.")
        print("               Your weights are v1 format — preferring v1.")
        print("               To avoid future confusion: pip uninstall nnunetv2")
        return "v1"
    elif has_v1:
        return "v1"
    elif has_v2:
        return "v2"
    return "none"


def find_task_folder(models_dir: str, task_name: str) -> Path | None:
    """
    Search for the task folder under models_dir.

    Handles both possible layouts:
      Layout A (v1 standard):
        models_dir/nnUNet/3d_fullres/Task006_Lung/
      Layout B (v1 non-standard / direct):
        models_dir/Task006_Lung/
      Layout C (v2):
        models_dir/Dataset006_Lung/ or models_dir/Dataset006/

    Returns the Path to the TaskXXX folder, or None if not found.
    """
    base = Path(models_dir)

    # Layout A — standard v1: models_dir/nnUNet/3d_fullres/TaskXXX/
    layout_a = base / "nnUNet" / "3d_fullres" / task_name
    if layout_a.exists():
        return layout_a

    # Layout B — direct: models_dir/TaskXXX/
    layout_b = base / task_name
    if layout_b.exists():
        return layout_b

    # Search recursively (handles any nesting depth)
    for candidate in base.rglob(task_name):
        if candidate.is_dir():
            return candidate

    return None


def find_trainer_name(task_folder: Path) -> str | None:
    """
    Read the trainer name from the folder that lives inside the task folder.
    e.g. Task006_Lung/nnUNetTrainer__nnUNetPlans__3d_fullres/ → returns that folder name.
    We need this because the exact trainer string varies between weight downloads.
    """
    if not task_folder or not task_folder.exists():
        return None

    # List immediate subdirectories — the trainer folder is one level down
    subdirs = [d for d in task_folder.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # Return the first subdirectory name (there's usually only one trainer per task)
    trainer = subdirs[0].name
    return trainer


def probe_weights(models_dir: str, task_name: str) -> dict:
    """
    Return a dict with all information needed to run inference.
    Scans the actual folder structure rather than assuming a fixed layout.
    Also detects which folds are available so we don't pass -f all
    when only fold_0..fold_4 exist.
    """
    task_folder = find_task_folder(models_dir, task_name)

    if task_folder is None:
        return {"found": False, "task_folder": None, "trainer": None, "version": "missing"}

    trainer = find_trainer_name(task_folder)

    v1_checkpoints = list(task_folder.rglob("model_final_checkpoint.model"))
    v1_plans       = list(task_folder.rglob("plans.pkl"))
    v2_checkpoints = list(task_folder.rglob("checkpoint_final.pth"))

    if v1_checkpoints or v1_plans:
        version = "v1"
    elif v2_checkpoints:
        version = "v2"
    else:
        version = "v1_partial"

    # Detect available folds
    # nnU-Net v1: fold_all means a single model trained on all data
    # fold_0..fold_4 means cross-validation — use "0 1 2 3 4" or just "0"
    available_folds = []
    if trainer:
        trainer_folder = task_folder / trainer
        if trainer_folder.exists():
            for child in trainer_folder.iterdir():
                if child.is_dir() and child.name.startswith("fold_"):
                    available_folds.append(child.name.replace("fold_", ""))

    # Pick the best fold strategy:
    #  - "all"      if fold_all exists (single model trained on all data)
    #  - "0 1 2 3 4" if all 5 cross-val folds exist (ensemble — best accuracy)
    #  - "0"         fallback to just fold 0
    if "all" in available_folds:
        best_fold = "all"
    elif len(available_folds) >= 5:
        best_fold = " ".join(sorted(available_folds))   # "0 1 2 3 4"
    elif available_folds:
        best_fold = available_folds[0]
    else:
        best_fold = "0"

    return {
        "found"           : True,
        "task_folder"     : task_folder,
        "trainer"         : trainer,
        "version"         : version,
        "v1_ckpts"        : len(v1_checkpoints),
        "v1_plans"        : len(v1_plans),
        "v2_ckpts"        : len(v2_checkpoints),
        "available_folds" : available_folds,
        "best_fold"       : best_fold,
    }


# ─────────────────────────────────────────────────────────────
# ORGAN MASK HELPERS
# ─────────────────────────────────────────────────────────────

def find_available_organ_masks(seg_dir: str, wanted: list) -> list:
    """Return only masks that actually exist on disk. Print diagnosis if missing."""
    seg_path = Path(seg_dir)
    found    = []
    missing  = []

    for name in wanted:
        if (seg_path / f"{name}.nii.gz").exists():
            found.append(name)
        else:
            missing.append(name)

    if missing:
        print(f"      Missing masks : {missing}")
        existing   = sorted(f.stem.replace(".nii","") for f in seg_path.glob("*.nii.gz"))
        lung_files = [f for f in existing if "lung" in f.lower()]
        if lung_files:
            print(f"      Lung files found : {lung_files}")
        elif existing:
            print(f"      Seg dir has (first 10): {existing[:10]}")
        else:
            print(f"      Seg dir appears empty — did step 2 complete?")

    if found:
        print(f"      Using masks   : {found}")
    return found


def extract_organ_roi(image_path: str,
                       seg_dir: str,
                       organ_masks: list,
                       padding_mm: float = 20.0) -> tuple:
    """
    Crop a bounding-box ROI around the union of the given organ masks.
    Returns (cropped_sitk_image, [x_min, y_min, z_min]) or (None, None).
    """
    seg_path  = Path(seg_dir)
    image     = sitk.ReadImage(image_path)
    union     = None
    ref_image = None

    for mask_name in organ_masks:
        mf = seg_path / f"{mask_name}.nii.gz"
        if not mf.exists():
            continue
        m = sitk.ReadImage(str(mf))
        a = sitk.GetArrayFromImage(m).astype(np.uint8)
        if union is None:
            union = a;  ref_image = m
        else:
            union = np.logical_or(union, a).astype(np.uint8)

    if union is None:
        return None, None

    coords = np.argwhere(union > 0)
    if len(coords) == 0:
        return None, None

    z_min, y_min, x_min = [int(v) for v in coords.min(axis=0)]
    z_max, y_max, x_max = [int(v) for v in coords.max(axis=0)]

    sp   = ref_image.GetSpacing()
    sz   = ref_image.GetSize()
    px, py, pz = int(padding_mm/sp[0]), int(padding_mm/sp[1]), int(padding_mm/sp[2])

    x_min = max(0, x_min-px);  x_max = min(sz[0]-1, x_max+px)
    y_min = max(0, y_min-py);  y_max = min(sz[1]-1, y_max+py)
    z_min = max(0, z_min-pz);  z_max = min(sz[2]-1, z_max+pz)

    f = sitk.RegionOfInterestImageFilter()
    f.SetIndex([x_min, y_min, z_min])
    f.SetSize([x_max-x_min, y_max-y_min, z_max-z_min])
    return f.Execute(image), [x_min, y_min, z_min]


# ─────────────────────────────────────────────────────────────
# INFERENCE RUNNERS
# ─────────────────────────────────────────────────────────────

def parse_trainer_and_plans(trainer_folder_name: str) -> tuple:
    """
    Split the trainer folder name into (trainer_class, plans_identifier).

    Your folder: nnUNetTrainer__nnUNetPlansv2.1
    nnU-Net v1 -tr flag = trainer class name only: "nnUNetTrainer"
    nnU-Net v1 -p  flag = plans identifier:        "nnUNetPlansv2.1"

    The double-underscore __ separates them.
    Single-underscore _ is part of the name itself.
    """
    if "__" in trainer_folder_name:
        parts   = trainer_folder_name.split("__")
        trainer = parts[0]          # e.g. "nnUNetTrainer"
        plans   = parts[1]          # e.g. "nnUNetPlansv2.1"
    else:
        # No double underscore — just use what we have
        trainer = trainer_folder_name
        plans   = "nnUNetPlansv2.1"  # safe default for v1 weights
    return trainer, plans


def run_nnunet_v1(crop_path:  str,
                   pred_dir:   str,
                   task_name:  str,
                   trainer:    str,       # full folder name, e.g. "nnUNetTrainer__nnUNetPlansv2.1"
                   models_dir: str,
                   fold:       str = "all") -> str | None:
    """
    Run nnU-Net v1 inference.

    KEY FIXES vs previous version:
      - Splits trainer folder name into -tr (trainer class) and -p (plans)
        nnU-Net v1 internally reconstructs the folder as:
          <RESULTS_FOLDER>/nnUNet/3d_fullres/<task>/<trainer>__<plans>/
        So -tr and -p must be passed separately, not the full folder name.
      - Sets dummy nnUNet_raw_data_base and nnUNet_preprocessed env vars.
        nnU-Net v1 warns if these are missing but inference still works —
        the warnings are just noise. Setting them to temp dirs suppresses them.
    """
    os.makedirs(pred_dir, exist_ok=True)

    # Prepare temp input folder — file must end in _0000.nii.gz
    input_folder = tempfile.mkdtemp()
    shutil.copy(crop_path, os.path.join(input_folder, "case_0000.nii.gz"))

    # Split "nnUNetTrainer__nnUNetPlansv2.1" → trainer="nnUNetTrainer", plans="nnUNetPlansv2.1"
    trainer_class, plans_id = parse_trainer_and_plans(trainer or "nnUNetTrainer__nnUNetPlansv2.1")

    print(f"      Trainer class : {trainer_class}")
    print(f"      Plans id      : {plans_id}")

    # Build environment — all three RESULTS vars needed by nnU-Net v1
    models_abs = str(Path(models_dir).resolve())
    tmp_dummy  = tempfile.mkdtemp()   # dummy path for raw/preprocessed (not used for inference)
    env = os.environ.copy()
    env["RESULTS_FOLDER"]          = models_abs
    env["nnUNet_results"]          = models_abs   # some v1 builds read this name
    env["nnUNet_raw_data_base"]    = tmp_dummy     # silences the warning
    env["nnUNet_preprocessed"]     = tmp_dummy     # silences the warning

    # Task number for -t flag  e.g. "Task006_Lung" → "6"
    task_num = task_name.replace("Task", "").split("_")[0].lstrip("0") or "0"

    # Base inference args — NOTE: -tr and -p are now included
    def make_cmd(prefix, t_flag):
        return prefix + [
            "-t",  t_flag,
            "-i",  input_folder,
            "-o",  pred_dir,
            "-m",  "3d_fullres",
            "-tr", trainer_class,
            "-p",  plans_id,
            "-f", *fold.split(),
            "--disable_tta",
        ]

    # Try 4 combinations: (console script | python -m) × (task name | task number)
    prefixes = [
        ["nnUNet_predict"],
        [sys.executable, "-m", "nnunet.inference.predict_simple"],
    ]

    for prefix in prefixes:
        for t_flag in [task_name, task_num]:
            cmd = make_cmd(prefix, t_flag)
            print(f"\n      CMD: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env)
            if result.returncode == 0:
                shutil.rmtree(input_folder, ignore_errors=True)
                shutil.rmtree(tmp_dummy,    ignore_errors=True)
                outputs = list(Path(pred_dir).glob("*.nii.gz"))
                if outputs:
                    print(f"      ✓ Prediction saved: {outputs[0]}")
                    return str(outputs[0])
                else:
                    print(f"      ✗ Inference succeeded but no output .nii.gz found in {pred_dir}")
                    return None

    shutil.rmtree(input_folder, ignore_errors=True)
    shutil.rmtree(tmp_dummy,    ignore_errors=True)
    print(f"\n      ✗ All nnU-Net v1 inference attempts failed")
    print(f"        Check that fold_all or fold_0..fold_4 exist under:")
    print(f"        {models_abs}\\nnUNet\\3d_fullres\\{task_name}\\{trainer}\\")
    return None


def run_nnunet_v2(crop_path: str,
                   pred_dir:  str,
                   dataset_id: int,
                   config:    str,
                   trainer:   str,
                   models_dir: str,
                   fold:      str = "all") -> str | None:
    """Run nnU-Net v2 inference."""
    os.makedirs(pred_dir, exist_ok=True)
    input_folder = tempfile.mkdtemp()
    shutil.copy(crop_path, os.path.join(input_folder, "case_0000.nii.gz"))

    env = os.environ.copy()
    env["nnUNet_results"] = str(Path(models_dir).resolve())

    cmd = [
        "nnUNetv2_predict",
        "-i",  input_folder,
        "-o",  pred_dir,
        "-d",  str(dataset_id),
        "-c",  config,
        "-tr", trainer,
        "-f",  fold,
        "--disable_tta",
    ]
    print(f"      CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    shutil.rmtree(input_folder, ignore_errors=True)

    if result.returncode != 0:
        return None
    outputs = list(Path(pred_dir).glob("*.nii.gz"))
    return str(outputs[0]) if outputs else None


# ─────────────────────────────────────────────────────────────
# TUMOUR MASK EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_tumour_mask(prediction_path: str,
                         tumour_label:   int,
                         origin_index:   list,
                         output_path:    str,
                         ref_image_path: str) -> dict:
    """Extract tumour label and paste back into full-image space."""
    pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(prediction_path))
    tumour   = (pred_arr == tumour_label).astype(np.uint8)

    ref_img  = sitk.ReadImage(ref_image_path)
    full_arr = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)

    ox, oy, oz = origin_index
    dz, dy, dx = tumour.shape
    z2 = min(oz+dz, full_arr.shape[0])
    y2 = min(oy+dy, full_arr.shape[1])
    x2 = min(ox+dx, full_arr.shape[2])
    full_arr[oz:z2, oy:y2, ox:x2] = tumour[:z2-oz, :y2-oy, :x2-ox]

    out = sitk.GetImageFromArray(full_arr)
    out.CopyInformation(ref_img)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sitk.WriteImage(out, output_path)

    sp          = ref_img.GetSpacing()
    voxel_count = int(np.sum(full_arr > 0))
    volume_ml   = voxel_count * sp[0] * sp[1] * sp[2] / 1000.0
    centroid_mm = [0.0, 0.0, 0.0]
    if voxel_count > 0:
        c = np.argwhere(full_arr > 0).mean(axis=0)
        centroid_mm = list(ref_img.TransformIndexToPhysicalPoint([int(c[2]), int(c[1]), int(c[0])]))

    return {
        "voxel_count": voxel_count,
        "volume_ml"  : round(volume_ml, 3),
        "centroid_mm": [round(v, 1) for v in centroid_mm],
        "present"    : voxel_count > 50,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all_tumour_detection(
    input_path:    str = INPUT_NIFTI,
    seg_dir:       str = SEGMENTATION_DIR,
    output_dir:    str = TUMOUR_OUTPUT_DIR,
    findings_json: str = FINDINGS_JSON,
    models_dir:    str = NNUNET_MODELS_DIR,
) -> dict:

    findings     = {}
    enabled      = {k: v for k, v in TASK_REGISTRY.items() if v["enabled"]}
    pkg_version  = detect_nnunet_version()
    models_dir   = str(Path(models_dir).resolve())   # always use absolute path

    print(f"\n[nnU-Net] Package version : {pkg_version}")
    print(f"[nnU-Net] Models dir      : {models_dir}")
    print(f"[nnU-Net] Tasks enabled   : {len(enabled)}\n")

    if not enabled:
        print("  No tasks enabled. Set 'enabled': True in TASK_REGISTRY.")
        return {}

    if pkg_version == "none":
        print("  ✗ No nnU-Net package found. Install: pip install nnunet")
        return {}

    for organ_name, task in enabled.items():
        task_name = task["task_name"]
        print(f"{'─'*55}")
        print(f"  {organ_name.upper()} — {task['description']}")
        print(f"{'─'*55}")

        # ── 1. Locate weights ─────────────────────────────────
        info = probe_weights(models_dir, task_name)
        print(f"  Weight scan:")
        print(f"    found         : {info['found']}")
        if info["found"]:
            print(f"    task_folder   : {info['task_folder']}")
            print(f"    trainer       : {info['trainer']}")
            print(f"    version       : {info['version']}")
            print(f"    v1 plans.pkl  : {info.get('v1_plans',0)}")
            print(f"    v1 checkpoints: {info.get('v1_ckpts',0)}")
            print(f"    v2 checkpoints: {info.get('v2_ckpts',0)}")

        if not info["found"]:
            print(f"\n  ✗ Weights not found.")
            print(f"    Searched for '{task_name}' under: {models_dir}")
            print(f"    Your folder structure should be:")
            print(f"      {models_dir}/nnUNet/3d_fullres/{task_name}/")
            print(f"    Download command (v1):")
            print(f"      set RESULTS_FOLDER={models_dir}")
            print(f"      nnUNet_download_pretrained_model {task_name}")
            findings[organ_name] = {"skipped": True, "reason": "Weights not found"}
            continue

        # ── 2. Version compatibility check ────────────────────
        weight_ver = info["version"].replace("_partial", "")  # "v1" or "v2"
        if weight_ver != pkg_version:
            print(f"\n  ✗ VERSION MISMATCH:")
            print(f"    Weights format : {weight_ver}")
            print(f"    Package        : {pkg_version}")
            if weight_ver == "v1" and pkg_version == "v2":
                print(f"    Fix A: pip uninstall nnunetv2  (keep your v1 weights)")
                print(f"    Fix B: pip install nnunet      (install v1 package)")
            else:
                print(f"    Fix: pip install nnunetv2")
            findings[organ_name] = {"skipped": True, "reason": f"Version mismatch: weights={weight_ver} pkg={pkg_version}"}
            continue

        # ── 3. Find organ masks ───────────────────────────────
        print(f"\n  Organ masks:")
        available = find_available_organ_masks(seg_dir, task["organ_masks"])
        if not available:
            print(f"  ✗ No organ masks found. Run step2 first.")
            findings[organ_name] = {"skipped": True, "reason": "No organ masks"}
            continue

        # ── 4. Extract ROI crop ───────────────────────────────
        print(f"\n  Extracting ROI crop…")
        crop_path = os.path.join(output_dir, f"{organ_name}_crop.nii.gz")
        cropped, origin_index = extract_organ_roi(input_path, seg_dir, available)
        if cropped is None:
            print(f"  ✗ Organ region is empty in masks")
            findings[organ_name] = {"skipped": True, "reason": "Empty organ region"}
            continue

        os.makedirs(output_dir, exist_ok=True)
        sitk.WriteImage(cropped, crop_path)
        print(f"  Crop size   : {cropped.GetSize()}")
        print(f"  Origin index: {origin_index}")

        # ── 5. Run inference ──────────────────────────────────
        best_fold = info.get("best_fold", "0")
        print(f"\n  Running nnU-Net {pkg_version} inference…")
        print(f"  Folds available: {info.get('available_folds', [])}  →  using: {best_fold}")
        pred_dir  = os.path.join(output_dir, f"{organ_name}_prediction")
        pred_path = None

        if pkg_version == "v1":
            pred_path = run_nnunet_v1(
                crop_path  = crop_path,
                pred_dir   = pred_dir,
                task_name  = task_name,
                trainer    = info["trainer"] or "nnUNetTrainer__nnUNetPlansv2.1",
                models_dir = models_dir,
                fold       = best_fold,
            )
        else:
            pred_path = run_nnunet_v2(
                crop_path  = crop_path,
                pred_dir   = pred_dir,
                dataset_id = task["dataset_id_v2"],
                config     = task["config"],
                trainer    = info["trainer"] or "nnUNetTrainer",
                models_dir = models_dir,
                fold       = best_fold,
            )

        if pred_path is None:
            findings[organ_name] = {"skipped": True, "reason": "Inference failed"}
            continue

        # ── 6. Extract tumour mask ────────────────────────────
        print(f"\n  Extracting tumour label {task['tumour_label']}…")
        tumour_mask = os.path.join(output_dir, f"{organ_name}_tumour.nii.gz")
        stats = extract_tumour_mask(
            pred_path, task["tumour_label"],
            origin_index, tumour_mask, input_path
        )

        findings[organ_name] = {
            "skipped"       : False,
            "tumour_present": stats["present"],
            "volume_ml"     : stats["volume_ml"],
            "voxel_count"   : stats["voxel_count"],
            "centroid_mm"   : stats["centroid_mm"],
            "mask_path"     : tumour_mask,
            "colour_rgb"    : task["colour_rgb"],
            "colour_hex"    : "#{:02x}{:02x}{:02x}".format(*task["colour_rgb"]),
            "description"   : task["description"],
            "task_id"       : task_name,
            "nnunet_version": pkg_version,
        }

        status = "TUMOUR FOUND" if stats["present"] else "Not detected"
        vol    = f"{stats['volume_ml']:.2f} mL" if stats["present"] else ""
        print(f"\n  ━━ Result: {status} {vol}")

    # ── Save findings ─────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(findings_json)), exist_ok=True)
    with open(findings_json, "w") as f:
        json.dump(findings, f, indent=2)

    detected = [k for k, v in findings.items()
                if not v.get("skipped") and v.get("tumour_present")]

    print(f"\n{'='*55}")
    print(f"  DETECTION COMPLETE  |  Tumours found: {len(detected)}")
    for org in detected:
        fi = findings[org]
        print(f"  • {org:<12}  {fi['volume_ml']:.2f} mL  @ {fi['centroid_mm']} mm")
    print(f"  Findings → {findings_json}")
    print(f"{'='*55}\n")
    return findings


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # required on Windows

    import argparse
    parser = argparse.ArgumentParser(description="nnU-Net tumour detection (v1 + v2)")
    parser.add_argument("--input",    default=INPUT_NIFTI)
    parser.add_argument("--seg_dir",  default=SEGMENTATION_DIR)
    parser.add_argument("--output",   default=TUMOUR_OUTPUT_DIR)
    parser.add_argument("--findings", default=FINDINGS_JSON)
    parser.add_argument("--models",   default=NNUNET_MODELS_DIR,
                        help="Root models dir (contains nnUNet/ subfolder)")
    parser.add_argument("--check",    action="store_true",
                        help="Diagnose versions and paths without running inference")
    args = parser.parse_args()

    if args.check:
        # ── Diagnostic mode ───────────────────────────────────
        pkg = detect_nnunet_version()
        mdir = str(Path(args.models).resolve())
        print(f"\n{'='*55}")
        print(f"  DIAGNOSTIC REPORT")
        print(f"{'='*55}")
        print(f"  Package version : {pkg}")
        print(f"  Models dir      : {mdir}")
        print(f"  Models dir exists: {Path(mdir).exists()}\n")
        for name, task in TASK_REGISTRY.items():
            info = probe_weights(mdir, task["task_name"])
            enabled_str = "ENABLED" if task["enabled"] else "disabled"
            if info["found"]:
                compat = "✓" if info["version"].startswith(pkg) else "✗ MISMATCH"
                print(f"  {name:<12} [{enabled_str:<8}]  "
                      f"found=YES  version={info['version']}  "
                      f"trainer={info['trainer']}  {compat}")
            else:
                print(f"  {name:<12} [{enabled_str:<8}]  found=NO")
        print()
    else:
        run_all_tumour_detection(
            args.input, args.seg_dir, args.output, args.findings, args.models
        )