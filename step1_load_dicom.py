"""
STEP 1 — DICOM Loading & Preprocessing
=======================================
Input  : folder of .dcm files (or a .zip)
Output : preprocessed NIfTI file saved to ./output/preprocessed.nii.gz

Install:
    pip install pydicom SimpleITK numpy nibabel tqdm
"""

import os
import zipfile
import shutil
import tempfile
import numpy as np
import pydicom
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths before running
# ─────────────────────────────────────────────────────────────
INPUT_PATH   = r"E:\medical_new_claude\data\dicom_series\LIDC-IDRI-0001"   # folder of .dcm files OR a .zip file
OUTPUT_DIR   = "./output"
OUTPUT_FILE  = "preprocessed.nii.gz"

# Hounsfield Unit windows (min, max) for different scan types
HU_WINDOWS = {
    "lung":    (-1000,  400),   # lung parenchyma + soft tissue
    "abdomen": ( -150,  250),   # abdominal organs
    "bone":    (  -500, 1800),  # bone structures
    "brain":   (   0,   80),    # brain tissue
}
ACTIVE_WINDOW = "lung"          # change to match your scan type

# Target voxel spacing in mm (isotropic resampling)
TARGET_SPACING = [1.0, 1.0, 1.0]
# ─────────────────────────────────────────────────────────────


def extract_zip(zip_path: str, dest_dir: str) -> str:
    """Extract a zip archive and return the folder containing .dcm files."""
    print(f"[1/5] Extracting ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # Find the deepest folder that actually contains .dcm files
    for root, dirs, files in os.walk(dest_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            return root
    raise FileNotFoundError("No .dcm files found inside the ZIP archive.")


def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """Load an ordered DICOM series from a directory."""
    print(f"[2/5] Loading DICOM series from: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    if not dicom_names:
        raise ValueError(f"No readable DICOM series found in {dicom_dir}")

    print(f"      Found {len(dicom_names)} slices")
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()   # keep DICOM metadata
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    print(f"      Original size    : {image.GetSize()}")
    print(f"      Original spacing : {[round(s,3) for s in image.GetSpacing()]} mm")
    print(f"      Origin           : {[round(o,3) for o in image.GetOrigin()]}")
    return image


def apply_hu_window(image: sitk.Image, window: str = ACTIVE_WINDOW) -> sitk.Image:
    """Clip to the HU window and normalise to [0, 1]."""
    lo, hi = HU_WINDOWS[window]
    print(f"[3/5] Applying HU window '{window}': [{lo}, {hi}]")

    clamp = sitk.ClampImageFilter()
    clamp.SetLowerBound(float(lo))
    clamp.SetUpperBound(float(hi))
    image = clamp.Execute(sitk.Cast(image, sitk.sitkFloat32))

    # Normalise to [0, 1]
    rescale = sitk.RescaleIntensityImageFilter()
    rescale.SetOutputMinimum(0.0)
    rescale.SetOutputMaximum(1.0)
    return rescale.Execute(image)


def resample_isotropic(image: sitk.Image,
                       target_spacing: list = TARGET_SPACING) -> sitk.Image:
    """Resample the volume to isotropic voxel spacing."""
    original_spacing = list(image.GetSpacing())
    original_size    = list(image.GetSize())

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    print(f"[4/5] Resampling to {target_spacing} mm isotropic")
    print(f"      {original_size} → {new_size} voxels")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)   # smooth resampling
    return resampler.Execute(image)


def fix_orientation(image: sitk.Image) -> sitk.Image:
    """Reorient volume to standard RAS+ (Right-Anterior-Superior) orientation."""
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAS")
    return orient_filter.Execute(image)


def save_nifti(image: sitk.Image, output_path: str) -> None:
    """Save the processed volume as a compressed NIfTI file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[5/5] Saving NIfTI → {output_path}")
    sitk.WriteImage(image, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Done. File size: {size_mb:.1f} MB")


def print_dicom_metadata(dicom_dir: str) -> None:
    """Print key DICOM tags from the first slice for reference."""
    files = sorted(Path(dicom_dir).glob("*.dcm"))
    if not files:
        return
    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    tags = {
        "Patient ID"      : getattr(ds, "PatientID",       "N/A"),
        "Patient Name"    : str(getattr(ds, "PatientName",  "N/A")),
        "Modality"        : getattr(ds, "Modality",        "N/A"),
        "Study Date"      : getattr(ds, "StudyDate",       "N/A"),
        "Body Part"       : getattr(ds, "BodyPartExamined","N/A"),
        "Slice Thickness" : getattr(ds, "SliceThickness",  "N/A"),
        "Rows x Cols"     : f"{getattr(ds,'Rows','?')} x {getattr(ds,'Columns','?')}",
        "Manufacturer"    : getattr(ds, "Manufacturer",    "N/A"),
    }
    print("\n  ── DICOM Metadata ──────────────────────")
    for k, v in tags.items():
        print(f"  {k:<20}: {v}")
    print("  ────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run(input_path: str = INPUT_PATH,
        output_dir: str  = OUTPUT_DIR,
        output_file: str = OUTPUT_FILE,
        window: str      = ACTIVE_WINDOW) -> str:
    """
    Full preprocessing pipeline.
    Returns the path to the saved NIfTI file.
    """
    tmp_dir = None

    try:
        # Handle ZIP input
        if input_path.lower().endswith(".zip"):
            tmp_dir    = tempfile.mkdtemp()
            dicom_dir  = extract_zip(input_path, tmp_dir)
        else:
            dicom_dir  = input_path

        # Print metadata
        print_dicom_metadata(dicom_dir)

        # Pipeline
        image = load_dicom_series(dicom_dir)
        # image = apply_hu_window(image, window)
        image = resample_isotropic(image)
        image = fix_orientation(image)

        out_path = os.path.join(output_dir, output_file)
        save_nifti(image, out_path)
        return out_path

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DICOM → NIfTI preprocessing")
    parser.add_argument("--input",  default=INPUT_PATH,   help="DICOM folder or .zip")
    parser.add_argument("--output", default=OUTPUT_DIR,   help="Output directory")
    parser.add_argument("--window", default=ACTIVE_WINDOW,
                        choices=list(HU_WINDOWS.keys()),  help="HU window preset")
    args = parser.parse_args()

    result = run(args.input, args.output, OUTPUT_FILE, args.window)
    print(f"\n✓ Preprocessing complete: {result}")
