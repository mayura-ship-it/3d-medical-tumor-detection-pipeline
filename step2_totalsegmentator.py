"""
STEP 2 — TotalSegmentator: Full-Body Organ Segmentation
=========================================================
FIXES in this version:
  [FIX 1] voxel detection threshold changed from > 100 to > 0
           Reason: TotalSegmentator output masks are binary 0/1.
           The > 100 threshold was silently hiding ALL organs.
  [FIX 2] Windows multiprocessing crash fixed with freeze_support()
           and proper if __name__ == '__main__' guard.
  [FIX 3] ml flag removed — ml=True was TotalSegmentator v1 syntax.
           Current versions use individual per-organ .nii.gz files
           by default (which is exactly what we want).
  [FIX 4] Added debug_masks() helper so you can see raw voxel stats
           if "0 organs present" appears again.

Input  : preprocessed NIfTI from step 1  (./output/preprocessed.nii.gz)
Output : one NIfTI mask per organ         (./output/segmentations/<organ>.nii.gz)
         combined colour map JSON         (./output/organ_colors.json)
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_NIFTI  = "./output/preprocessed.nii.gz"
OUTPUT_DIR   = "./output/segmentations"
COLORS_JSON  = "./output/organ_colors.json"
FAST_MODE    = False
USE_GPU      = True
# ─────────────────────────────────────────────────────────────


ORGAN_COLORS = {
    "lung_upper_lobe_left"        : [100, 160, 220],
    "lung_lower_lobe_left"        : [ 70, 130, 200],
    "lung_upper_lobe_right"       : [130, 190, 240],
    "lung_middle_lobe_right"      : [100, 170, 230],
    "lung_lower_lobe_right"       : [ 60, 120, 195],
    "lung"                        : [100, 160, 220],
    "trachea"                     : [180, 200, 210],
    "heart"                       : [220,  60,  70],
    "heart_myocardium"            : [200,  50,  60],
    "heart_atrium_left"           : [230,  80,  80],
    "heart_ventricle_left"        : [210,  40,  50],
    "heart_atrium_right"          : [240, 100,  90],
    "heart_ventricle_right"       : [220,  70,  70],
    "aorta"                       : [220, 120,  50],
    "pulmonary_vein"              : [180, 100, 160],
    "inferior_vena_cava"          : [ 90, 130, 200],
    "portal_vein_and_splenic_vein": [110, 150, 190],
    "esophagus"                   : [180, 150, 120],
    "liver"                       : [210, 100,  60],
    "gallbladder"                 : [170, 200,  80],
    "stomach"                     : [230, 170,  90],
    "pancreas"                    : [255, 200,  80],
    "spleen"                      : [180,  80, 140],
    "small_bowel"                 : [230, 200, 150],
    "duodenum"                    : [210, 180, 120],
    "colon"                       : [200, 160, 100],
    "urinary_bladder"             : [ 90, 160, 200],
    "prostate"                    : [200, 130,  90],
    "kidney_left"                 : [220,  90,  80],
    "kidney_right"                : [230, 100,  90],
    "adrenal_gland_left"          : [160, 200, 120],
    "adrenal_gland_right"         : [150, 190, 110],
    "vertebrae_L5"                : [220, 210, 180],
    "vertebrae_L4"                : [215, 205, 175],
    "vertebrae_L3"                : [210, 200, 170],
    "vertebrae_L2"                : [205, 195, 165],
    "vertebrae_L1"                : [200, 190, 160],
    "vertebrae_T12"               : [195, 185, 155],
    "vertebrae_T11"               : [190, 180, 150],
    "vertebrae_T10"               : [185, 175, 145],
    "vertebrae_T9"                : [180, 170, 140],
    "vertebrae_T8"                : [175, 165, 135],
    "vertebrae_T7"                : [170, 160, 130],
    "vertebrae_T6"                : [165, 155, 125],
    "vertebrae_T5"                : [160, 150, 120],
    "vertebrae_T4"                : [155, 145, 115],
    "vertebrae_T3"                : [150, 140, 110],
    "vertebrae_T2"                : [145, 135, 105],
    "vertebrae_T1"                : [140, 130, 100],
    "vertebrae_C7"                : [135, 125,  95],
    "vertebrae_C6"                : [130, 120,  90],
    "vertebrae_C5"                : [125, 115,  85],
    "vertebrae_C4"                : [120, 110,  80],
    "vertebrae_C3"                : [115, 105,  75],
    "vertebrae_C2"                : [110, 100,  70],
    "vertebrae_C1"                : [105,  95,  65],
    "hip_left"                    : [200, 185, 160],
    "hip_right"                   : [195, 180, 155],
    "femur_left"                  : [190, 175, 150],
    "femur_right"                 : [185, 170, 145],
    "sacrum"                      : [210, 195, 170],
    "sternum"                     : [225, 215, 190],
    "gluteus_maximus_left"        : [180, 120, 100],
    "gluteus_maximus_right"       : [175, 115,  95],
    "gluteus_medius_left"         : [170, 110,  90],
    "gluteus_medius_right"        : [165, 105,  85],
    "iliopsoas_left"              : [160, 100,  80],
    "iliopsoas_right"             : [155,  95,  75],
    "brain"                       : [230, 200, 170],
    "_default"                    : [180, 180, 180],
}


def get_color(organ_name: str) -> list:
    if organ_name in ORGAN_COLORS:
        return ORGAN_COLORS[organ_name]
    for key, color in ORGAN_COLORS.items():
        if key in organ_name or organ_name in key:
            return color
    return ORGAN_COLORS["_default"]


def run_totalsegmentator(input_path: str,
                          output_dir: str,
                          fast: bool = FAST_MODE,
                          gpu:  bool = USE_GPU) -> list:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        raise ImportError("Run: pip install TotalSegmentator")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[1/3] Running TotalSegmentator")
    print(f"      Input : {input_path}")
    print(f"      Output: {output_dir}")
    print(f"      Fast  : {fast}  |  GPU: {gpu}")
    print(f"      NOTE  : ml flag removed (v2 default = per-organ files)\n")

    # FIX: do NOT pass ml=True — that was a TotalSegmentator v1-only flag.
    # Current TotalSegmentator writes one .nii.gz per organ by default.
    totalsegmentator(
        input  = input_path,
        output = output_dir,
        fast   = fast,
        device = "gpu" if gpu else "cpu",
    )

    masks = sorted(Path(output_dir).glob("*.nii.gz"))
    print(f"\n      ✓ {len(masks)} mask files written to {output_dir}")
    return [str(m) for m in masks]


def debug_masks(output_dir: str) -> None:
    """
    Print raw voxel stats for every mask file.
    Run this when you see '0 organs present' to find out why.
    
    Usage:
        python step2_totalsegmentator.py --debug
    """
    masks = sorted(Path(output_dir).glob("*.nii.gz"))
    if not masks:
        print(f"[DEBUG] No .nii.gz files found in {output_dir}")
        return

    print(f"\n[DEBUG] {len(masks)} mask files in {output_dir}")
    print(f"  {'File':<50} {'dtype':<8} {'min':>8} {'max':>8} {'nonzero':>10}")
    print(f"  {'─'*50} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    for m in masks[:30]:
        try:
            img = sitk.ReadImage(str(m))
            arr = sitk.GetArrayFromImage(img)
            nz  = int(np.sum(arr > 0))
            print(f"  {m.name:<50} {str(arr.dtype):<8} "
                  f"{float(arr.min()):>8.3f} {float(arr.max()):>8.3f} {nz:>10,}")
        except Exception as e:
            print(f"  {m.name:<50} ERROR: {e}")

    if len(masks) > 30:
        print(f"  … and {len(masks)-30} more files")


def build_organ_summary(output_dir: str, colors_path: str) -> dict:
    """
    Read each mask, compute voxel count + volume, attach colour.

    FIX: present = voxel_count > 0
    Previously used > 100 which was hiding all organs because:
    - TotalSegmentator masks are binary (0 or 1 integers)
    - Small organs like adrenal glands can have just a few hundred voxels
    - The > 100 filter is only appropriate if masks were label-integer volumes
      (which ml=True used to produce) — NOT per-organ binary masks
    """
    print(f"[2/3] Building organ summary…")
    masks = sorted(Path(output_dir).glob("*.nii.gz"))

    if not masks:
        print(f"      ✗ No mask files in {output_dir}")
        return {}

    summary = {}
    for mask_path in masks:
        # Strip .nii from stem if double extension (e.g. liver.nii.gz → stem = liver.nii)
        organ = mask_path.stem
        if organ.endswith(".nii"):
            organ = organ[:-4]

        try:
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)
        except Exception as e:
            print(f"      ✗ Could not read {mask_path.name}: {e}")
            continue

        voxel_count = int(np.sum(arr > 0))
        spacing     = img.GetSpacing()
        voxel_vol   = spacing[0] * spacing[1] * spacing[2]
        volume_ml   = voxel_count * voxel_vol / 1000.0
        color       = get_color(organ)
        hex_color   = "#{:02x}{:02x}{:02x}".format(*color)

        summary[organ] = {
            "mask_path"  : str(mask_path),
            "voxel_count": voxel_count,
            "volume_ml"  : round(volume_ml, 2),
            "color_rgb"  : color,
            "color_hex"  : hex_color,
            "present"    : voxel_count > 0,   # ← FIXED (was > 100)
        }

    os.makedirs(os.path.dirname(os.path.abspath(colors_path)), exist_ok=True)
    with open(colors_path, "w") as f:
        json.dump(summary, f, indent=2)

    present = sum(1 for v in summary.values() if v["present"])
    print(f"      ✓ {len(summary)} masks read  |  {present} organs present")
    print(f"        Saved → {colors_path}")
    return summary


def print_organ_report(summary: dict) -> None:
    present = {k: v for k, v in summary.items() if v["present"]}
    if not present:
        print("\n  ✗ No organs in report — run with --debug to investigate")
        return
    print(f"\n[3/3] Organ Report  ({len(present)} organs)")
    print(f"  {'Organ':<45} {'Volume (mL)':>12}  Colour")
    print(f"  {'─'*45} {'─'*12}  {'─'*7}")
    for organ, info in sorted(present.items()):
        print(f"  {organ:<45} {info['volume_ml']:>12.1f}  {info['color_hex']}")


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def run(input_path:  str  = INPUT_NIFTI,
        output_dir:  str  = OUTPUT_DIR,
        colors_json: str  = COLORS_JSON,
        fast:        bool = FAST_MODE,
        gpu:         bool = USE_GPU) -> dict:

    run_totalsegmentator(input_path, output_dir, fast, gpu)
    debug_masks(output_dir)   # always print raw stats so you can see what happened
    summary = build_organ_summary(output_dir, colors_json)
    print_organ_report(summary)
    print(f"\n✓ Segmentation complete → {colors_json}")
    return summary


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# FIX: __main__ guard is REQUIRED on Windows.
# TotalSegmentator internally uses multiprocessing.
# On Windows, Python uses "spawn" instead of "fork" — child processes
# re-import the module. Without this guard the child tries to run the
# full pipeline again → recursive crash / "context already set" error.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # required for Windows frozen executables

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_NIFTI)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--fast",   action="store_true")
    parser.add_argument("--cpu",    action="store_true")
    parser.add_argument("--debug",  action="store_true",
                        help="Only show raw mask stats — no re-segmentation")
    args = parser.parse_args()

    if args.debug:
        debug_masks(args.output)
    else:
        run(args.input, args.output, COLORS_JSON,
            fast=args.fast, gpu=not args.cpu)