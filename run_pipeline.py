"""
STEP 5 — Full Pipeline Runner
================================
Orchestrates all steps end-to-end:
  1. DICOM loading & preprocessing
  2. TotalSegmentator organ segmentation
  3. nnU-Net tumour detection
  4. 3D mesh generation

Usage:
    # Basic (DICOM folder):
    python run_pipeline.py --input ./data/dicom_series

    # From ZIP:
    python run_pipeline.py --input ./data/scan.zip

    # Skip steps you've already run:
    python run_pipeline.py --input ./data/dicom_series --skip-preprocess
    python run_pipeline.py --input ./data/dicom_series --skip-preprocess --skip-segment

    # CPU only (no GPU):
    python run_pipeline.py --input ./data/dicom_series --cpu

    # Fast mode (lower resolution, much quicker):
    python run_pipeline.py --input ./data/dicom_series --fast
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# CONFIG — edit output root if needed
# ─────────────────────────────────────────────────────────────
OUTPUT_ROOT       = "./output"
PREPROCESSED_FILE = "preprocessed.nii.gz"
SEG_DIR           = "segmentations"
TUMOUR_DIR        = "tumours"
MESH_DIR          = "meshes"
ORGAN_COLORS_FILE = "organ_colors.json"
FINDINGS_FILE     = "tumour_findings.json"
MANIFEST_FILE     = "meshes/manifest.json"
PIPELINE_LOG      = "pipeline_log.json"
# ─────────────────────────────────────────────────────────────


def path(relative: str) -> str:
    return os.path.join(OUTPUT_ROOT, relative)


def log_step(log: dict, step: str, status: str,
             duration_s: float, output: str = "") -> None:
    log[step] = {
        "status"     : status,
        "duration_s" : round(duration_s, 1),
        "output"     : output,
        "timestamp"  : datetime.now().isoformat(),
    }
    with open(path(PIPELINE_LOG), "w") as f:
        json.dump(log, f, indent=2)


def print_header(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def print_step(n: int, title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*60}")


def run_pipeline(
    input_path:       str,
    output_root:      str  = OUTPUT_ROOT,
    window:           str  = "lung",
    fast:             bool = False,
    use_gpu:          bool = True,
    skip_preprocess:  bool = False,
    skip_segment:     bool = False,
    skip_tumour:      bool = False,
    skip_mesh:        bool = False,
) -> dict:
    global OUTPUT_ROOT
    OUTPUT_ROOT = output_root
    os.makedirs(output_root, exist_ok=True)

    log = {}
    total_start = time.time()

    print_header(f"Medical Imaging 3D Pipeline")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_root}")
    print(f"  Window : {window}  |  Fast: {fast}  |  GPU: {use_gpu}")
    print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ──────────────────────────────────────────────────────────
    # STEP 1 — DICOM Loading & Preprocessing
    # ──────────────────────────────────────────────────────────
    preprocessed_path = path(PREPROCESSED_FILE)
    if skip_preprocess and os.path.exists(preprocessed_path):
        print_step(1, "DICOM Preprocessing — SKIPPED (file exists)")
        log_step(log, "step1_preprocess", "skipped", 0, preprocessed_path)
    else:
        print_step(1, "DICOM Loading & Preprocessing")
        t = time.time()
        try:
            import step1_load_dicom as s1
            preprocessed_path = s1.run(
                input_path  = input_path,
                output_dir  = output_root,
                output_file = PREPROCESSED_FILE,
                window      = window,
            )
            log_step(log, "step1_preprocess", "success",
                     time.time()-t, preprocessed_path)
        except Exception as e:
            print(f"\n  ✗ Step 1 failed: {e}")
            log_step(log, "step1_preprocess", f"failed: {e}", time.time()-t)
            raise

    # ──────────────────────────────────────────────────────────
    # STEP 2 — TotalSegmentator
    # ──────────────────────────────────────────────────────────
    seg_dir        = path(SEG_DIR)
    organ_colors   = path(ORGAN_COLORS_FILE)
    if skip_segment and os.path.exists(organ_colors):
        print_step(2, "TotalSegmentator — SKIPPED (masks exist)")
        log_step(log, "step2_segment", "skipped", 0, seg_dir)
    else:
        print_step(2, "TotalSegmentator: Full-Body Organ Segmentation")
        t = time.time()
        try:
            import step2_totalsegmentator as s2
            summary = s2.run(
                input_path  = preprocessed_path,
                output_dir  = seg_dir,
                colors_json = organ_colors,
                fast        = fast,
                gpu         = use_gpu,
            )
            log_step(log, "step2_segment", "success", time.time()-t, seg_dir)
        except Exception as e:
            print(f"\n  ✗ Step 2 failed: {e}")
            log_step(log, "step2_segment", f"failed: {e}", time.time()-t)
            raise

    # ──────────────────────────────────────────────────────────
    # STEP 3 — nnU-Net Tumour Detection
    # ──────────────────────────────────────────────────────────
    tumour_dir    = path(TUMOUR_DIR)
    findings_json = path(FINDINGS_FILE)
    if skip_tumour and os.path.exists(findings_json):
        print_step(3, "nnU-Net Tumour Detection — SKIPPED")
        log_step(log, "step3_tumour", "skipped", 0, findings_json)
    else:
        print_step(3, "nnU-Net: Tumour Detection")
        t = time.time()
        try:
            import step3_nnunet_tumour as s3
            findings = s3.run_all_tumour_detection(
                input_path    = preprocessed_path,
                seg_dir       = seg_dir,
                output_dir    = tumour_dir,
                findings_json = findings_json,
            )
            log_step(log, "step3_tumour", "success", time.time()-t, findings_json)
        except Exception as e:
            print(f"\n  ✗ Step 3 failed: {e}")
            log_step(log, "step3_tumour", f"failed: {e}", time.time()-t)
            # Non-fatal — continue without tumour data
            with open(findings_json, "w") as f:
                json.dump({}, f)

    # ──────────────────────────────────────────────────────────
    # STEP 4 — 3D Mesh Generation
    # ──────────────────────────────────────────────────────────
    mesh_dir      = path(MESH_DIR)
    manifest_json = path(MANIFEST_FILE)
    if skip_mesh and os.path.exists(manifest_json):
        print_step(4, "Mesh Generation — SKIPPED")
        log_step(log, "step4_mesh", "skipped", 0, mesh_dir)
    else:
        print_step(4, "3D Mesh Generation")
        t = time.time()
        try:
            import step4_mesh_generation as s4
            manifest = s4.build_meshes(
                seg_dir       = seg_dir,
                tumour_dir    = tumour_dir,
                colors_json   = organ_colors,
                findings_json = findings_json,
                mesh_output   = mesh_dir,
                manifest_path = manifest_json,
            )
            log_step(log, "step4_mesh", "success", time.time()-t, mesh_dir)
        except Exception as e:
            print(f"\n  ✗ Step 4 failed: {e}")
            log_step(log, "step4_mesh", f"failed: {e}", time.time()-t)
            raise

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print_header("Pipeline Complete")
    print(f"  Total time : {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"  Output root: {output_root}")
    print(f"\n  Files generated:")
    print(f"    Preprocessed NIfTI : {preprocessed_path}")
    print(f"    Organ masks        : {seg_dir}/")
    print(f"    Tumour masks       : {tumour_dir}/")
    print(f"    3D meshes (OBJ)    : {mesh_dir}/*.obj")
    print(f"    3D meshes (GLB)    : {mesh_dir}/*.glb")
    print(f"    Organ colours      : {organ_colors}")
    print(f"    Tumour findings    : {findings_json}")
    print(f"    Mesh manifest      : {manifest_json}")
    print(f"    Pipeline log       : {path(PIPELINE_LOG)}")

    # Load and show manifest summary if exists
    if os.path.exists(manifest_json):
        with open(manifest_json) as f:
            mf = json.load(f)
        n_organs  = len(mf.get("organs", {}))
        n_tumours = len(mf.get("tumours", {}))
        print(f"\n  3D Viewer ready:")
        print(f"    {n_organs} organ meshes  +  {n_tumours} tumour mesh(es)")
        if mf.get("tumours"):
            print(f"\n  TUMOURS DETECTED:")
            for t_name, t_info in mf["tumours"].items():
                print(f"    • {t_name}  –  {t_info.get('volume_ml',0):.2f} mL")

    print(f"\n  Next: open the web viewer")
    print(f"    cd viewer && npm install && npm run dev")
    print(f"    (or run: python step5_web_server.py)\n")

    return log


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full medical imaging 3D pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --input ./data/dicom_series
  python run_pipeline.py --input ./data/scan.zip --fast --cpu
  python run_pipeline.py --input ./data/dicom_series --skip-preprocess
  python run_pipeline.py --input ./data/dicom_series --window abdomen
        """
    )
    parser.add_argument("--input",            required=True,
                        help="DICOM folder or .zip file")
    parser.add_argument("--output",           default=OUTPUT_ROOT,
                        help="Output directory root")
    parser.add_argument("--window",           default="lung",
                        choices=["lung","abdomen","bone","brain"],
                        help="HU windowing preset")
    parser.add_argument("--fast",             action="store_true",
                        help="TotalSegmentator fast mode")
    parser.add_argument("--cpu",              action="store_true",
                        help="Force CPU (no GPU)")
    parser.add_argument("--skip-preprocess",  action="store_true",
                        help="Skip step 1 if preprocessed.nii.gz exists")
    parser.add_argument("--skip-segment",     action="store_true",
                        help="Skip step 2 if organ masks exist")
    parser.add_argument("--skip-tumour",      action="store_true",
                        help="Skip step 3 tumour detection")
    parser.add_argument("--skip-mesh",        action="store_true",
                        help="Skip step 4 mesh generation")

    args = parser.parse_args()

    run_pipeline(
        input_path      = args.input,
        output_root     = args.output,
        window          = args.window,
        fast            = args.fast,
        use_gpu         = not args.cpu,
        skip_preprocess = args.skip_preprocess,
        skip_segment    = args.skip_segment,
        skip_tumour     = args.skip_tumour,
        skip_mesh       = args.skip_mesh,
    )
