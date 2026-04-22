"""
STEP 4 — 3D Mesh Generation
=============================
Converts NIfTI segmentation masks → coloured 3D meshes (OBJ + GLTF).
Each organ + each tumour becomes a separate mesh.

Input  : organ masks        (./output/segmentations/)
         tumour masks       (./output/tumours/)
         organ colour JSON  (./output/organ_colors.json)
         tumour findings    (./output/tumour_findings.json)
Output : 3D mesh files      (./output/meshes/<organ>.obj / .glb)
         manifest JSON      (./output/meshes/manifest.json)

Install:
    pip install vtk scikit-image trimesh numpy SimpleITK
    pip install pygltflib   # for GLTF export
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SEGMENTATION_DIR = "./output/segmentations"
TUMOUR_DIR       = "./output/tumours"
ORGAN_COLORS     = "./output/organ_colors.json"
TUMOUR_FINDINGS  = "./output/tumour_findings.json"
MESH_OUTPUT_DIR  = "./output/meshes"
MANIFEST_JSON    = "./output/meshes/manifest.json"

# Marching cubes settings
MC_LEVEL         = 0.5       # iso-surface threshold
MC_STEP_SIZE     = 1         # 1 = full res, 2 = half res (faster)

# Mesh simplification: target fraction of original triangles
# 0.3 = keep 30% of triangles (much smaller file, still good quality)
SIMPLIFY_RATIO   = 0.3

# Skip organs with volume below this (mL) — removes tiny noise masks
MIN_VOLUME_ML    = 1.0

# Organs to always skip (too thin/noisy for good mesh)
SKIP_ORGANS      = {"portal_vein_and_splenic_vein", "pulmonary_vein"}
# ─────────────────────────────────────────────────────────────


def nifti_to_mesh_vtk(mask_path: str,
                       smooth_iterations: int = 20,
                       decimate_ratio: float  = SIMPLIFY_RATIO) -> tuple | None:
    """
    VTK-based marching cubes + smoothing + decimation.
    Returns (vertices, faces) as numpy arrays, or None if empty.
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        raise ImportError("Install VTK: pip install vtk")

    # Load NIfTI as VTK image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()

    # Marching cubes
    mc = vtk.vtkMarchingCubes()
    mc.SetInputConnection(reader.GetOutputPort())
    mc.SetValue(0, MC_LEVEL)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() == 0:
        return None

    # Laplacian smoothing
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())
    smoother.SetNumberOfIterations(smooth_iterations)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()

    # Decimate (reduce triangle count)
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smoother.GetOutputPort())
    decimate.SetTargetReduction(1.0 - decimate_ratio)
    decimate.PreserveTopologyOn()
    decimate.Update()

    # Fix normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(decimate.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    poly = normals.GetOutput()

    # Extract vertices and faces
    n_pts  = poly.GetNumberOfPoints()
    n_tris = poly.GetNumberOfCells()
    if n_pts == 0 or n_tris == 0:
        return None

    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).reshape(-1, 3)
    faces = numpy_support.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

    return verts, faces


def nifti_to_mesh_scikit(mask_path: str,
                          step_size: int = MC_STEP_SIZE) -> tuple | None:
    """
    Fallback: scikit-image marching cubes (no VTK dependency).
    Returns (vertices, faces) as numpy arrays.
    """
    try:
        from skimage.measure import marching_cubes
        from skimage.filters import gaussian
    except ImportError:
        raise ImportError("Install scikit-image: pip install scikit-image")

    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    # Light Gaussian smoothing before MC
    arr = gaussian(arr, sigma=0.5)

    if arr.max() < MC_LEVEL:
        return None

    spacing = img.GetSpacing()   # (x, y, z)

    verts, faces, normals, _ = marching_cubes(
        arr,
        level     = MC_LEVEL,
        spacing   = (spacing[2], spacing[1], spacing[0]),  # zyx order
        step_size = step_size,
        allow_degenerate=False,
    )

    # Convert from voxel space to world space using image origin
    origin = np.array(img.GetOrigin())   # (x, y, z)
    verts  = verts[:, ::-1] + origin     # flip zyx→xyz, add origin

    return verts, faces


def save_obj(verts: np.ndarray,
             faces: np.ndarray,
             output_path: str,
             color_rgb: list) -> None:
    """Save mesh as Wavefront OBJ with a companion MTL material file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mtl_path = output_path.replace(".obj", ".mtl")
    organ    = Path(output_path).stem

    # Write MTL
    r, g, b = [c / 255.0 for c in color_rgb]
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {organ}_mat\n")
        f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")   # diffuse
        f.write(f"Ka {r*0.2:.4f} {g*0.2:.4f} {b*0.2:.4f}\n")  # ambient
        f.write(f"Ks 0.1 0.1 0.1\n")               # specular
        f.write(f"Ns 10\n")                          # shininess
        f.write(f"d 0.85\n")                         # alpha (semi-transparent)

    # Write OBJ
    with open(output_path, "w") as f:
        f.write(f"mtllib {Path(mtl_path).name}\n")
        f.write(f"o {organ}\n")
        f.write(f"usemtl {organ}_mat\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def save_glb(verts: np.ndarray,
             faces: np.ndarray,
             output_path: str,
             color_rgb: list) -> None:
    """Save mesh as binary GLTF (.glb) for Three.js / VR use."""
    try:
        import trimesh
    except ImportError:
        print("      (trimesh not installed — skipping GLB export; pip install trimesh)")
        return

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()

    r, g, b = [c / 255.0 for c in color_rgb]
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(np.array([r, g, b, 0.85], dtype=np.float32),
                              (len(verts), 1))
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.export(output_path)


def compute_mesh_stats(verts: np.ndarray) -> dict:
    """Return bounding box and centroid of a mesh."""
    mins  = verts.min(axis=0).tolist()
    maxs  = verts.max(axis=0).tolist()
    center = ((verts.min(axis=0) + verts.max(axis=0)) / 2).tolist()
    dims  = (verts.max(axis=0) - verts.min(axis=0)).tolist()
    return {
        "bbox_min"     : [round(v, 1) for v in mins],
        "bbox_max"     : [round(v, 1) for v in maxs],
        "center_mm"    : [round(v, 1) for v in center],
        "dimensions_mm": [round(v, 1) for v in dims],
        "vertex_count" : len(verts),
    }


def build_meshes(seg_dir:        str = SEGMENTATION_DIR,
                 tumour_dir:     str = TUMOUR_DIR,
                 colors_json:    str = ORGAN_COLORS,
                 findings_json:  str = TUMOUR_FINDINGS,
                 mesh_output:    str = MESH_OUTPUT_DIR,
                 manifest_path:  str = MANIFEST_JSON) -> dict:
    """
    Main pipeline: iterate all organ + tumour masks, generate meshes.
    Returns manifest dict.
    """
    os.makedirs(mesh_output, exist_ok=True)

    # Load colour and findings metadata
    with open(colors_json)   as f: organ_meta   = json.load(f)
    findings = {}
    if os.path.exists(findings_json):
        with open(findings_json) as f: findings = json.load(f)

    manifest = {"organs": {}, "tumours": {}}

    # ── 1. Organ meshes ────────────────────────────────────────
    print(f"\n[Step 4A] Generating organ meshes…")
    organ_masks = sorted(Path(seg_dir).glob("*.nii.gz"))

    for mask_path in organ_masks:
        organ = mask_path.stem.replace(".nii", "")

        if organ in SKIP_ORGANS:
            print(f"  skip    {organ} (in skip list)")
            continue

        meta = organ_meta.get(organ, {})
        if not meta.get("present", True):
            print(f"  absent  {organ}")
            continue
        if meta.get("volume_ml", 999) < MIN_VOLUME_ML:
            print(f"  tiny    {organ}  ({meta.get('volume_ml', 0):.1f} mL)")
            continue

        color = meta.get("color_rgb", [180, 180, 180])
        print(f"  mesh    {organ:<45} {meta.get('volume_ml', 0):>8.1f} mL", end="")

        # Try VTK first, fall back to scikit-image
        result = None
        try:
            result = nifti_to_mesh_vtk(str(mask_path))
        except Exception:
            pass
        if result is None:
            try:
                result = nifti_to_mesh_scikit(str(mask_path))
            except Exception as e:
                print(f"  ✗ {e}")
                continue

        if result is None:
            print(f"  (empty)")
            continue

        verts, faces = result
        obj_path = os.path.join(mesh_output, f"{organ}.obj")
        glb_path = os.path.join(mesh_output, f"{organ}.glb")
        save_obj(verts, faces, obj_path, color)
        save_glb(verts, faces, glb_path, color)

        stats = compute_mesh_stats(verts)
        manifest["organs"][organ] = {
            "obj"         : obj_path,
            "glb"         : glb_path,
            "color_rgb"   : color,
            "color_hex"   : meta.get("color_hex", "#b4b4b4"),
            "volume_ml"   : meta.get("volume_ml", 0),
            "is_tumour"   : False,
            **stats,
        }
        print(f"  ✓  {stats['vertex_count']:,} verts")

    # ── 2. Tumour meshes ───────────────────────────────────────
    print(f"\n[Step 4B] Generating tumour meshes…")
    for organ_name, finding in findings.items():
        if finding.get("skipped") or not finding.get("tumour_present"):
            continue

        tumour_mask = finding.get("mask_path", "")
        if not os.path.exists(tumour_mask):
            print(f"  missing mask: {tumour_mask}")
            continue

        color = finding.get("colour_rgb", [255, 50, 50])
        label = f"{organ_name}_tumour"
        print(f"  mesh    {label:<45} {finding.get('volume_ml', 0):>8.2f} mL", end="")

        result = None
        try:
            result = nifti_to_mesh_vtk(tumour_mask)
        except Exception:
            pass
        if result is None:
            try:
                result = nifti_to_mesh_scikit(tumour_mask)
            except Exception as e:
                print(f"  ✗ {e}")
                continue

        if result is None:
            print(f"  (empty)")
            continue

        verts, faces = result
        obj_path = os.path.join(mesh_output, f"{label}.obj")
        glb_path = os.path.join(mesh_output, f"{label}.glb")
        save_obj(verts, faces, obj_path, color)
        save_glb(verts, faces, glb_path, color)

        stats = compute_mesh_stats(verts)
        manifest["tumours"][label] = {
            "obj"          : obj_path,
            "glb"          : glb_path,
            "color_rgb"    : color,
            "color_hex"    : finding.get("colour_hex", "#ff3232"),
            "volume_ml"    : finding.get("volume_ml", 0),
            "centroid_mm"  : finding.get("centroid_mm", []),
            "is_tumour"    : True,
            "organ"        : organ_name,
            "task_id"      : finding.get("task_id", ""),
            "description"  : finding.get("description", ""),
            **stats,
        }
        print(f"  ✓  {stats['vertex_count']:,} verts")

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_organs  = len(manifest["organs"])
    n_tumours = len(manifest["tumours"])
    print(f"\n✓ Generated {n_organs} organ meshes + {n_tumours} tumour meshes")
    print(f"  Manifest → {manifest_path}")
    return manifest


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NIfTI → 3D mesh generation")
    parser.add_argument("--seg_dir",    default=SEGMENTATION_DIR)
    parser.add_argument("--tumour_dir", default=TUMOUR_DIR)
    parser.add_argument("--colors",     default=ORGAN_COLORS)
    parser.add_argument("--findings",   default=TUMOUR_FINDINGS)
    parser.add_argument("--output",     default=MESH_OUTPUT_DIR)
    args = parser.parse_args()

    build_meshes(args.seg_dir, args.tumour_dir, args.colors,
                 args.findings, args.output)
