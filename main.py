import os
import csv
import argparse
from pathlib import Path

_mpl_cache_dir = Path(__file__).resolve().parent / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

from core.model_loader import get_device, load_3ddfa_models, load_segformer_models
from core.config import ModelConfig
from core.preprocess import load_and_crop_image
from core.segmentation import run_segmentation
from core.inference import run_3ddfa, project_to_original_image
from core.visualization import launch_results_viewer
from core.mapping import map_segmentation_to_mesh
from core.symmetry import apply_symmetry, apply_nose_lip_symmetry
from core.constants import label2id, symmetric_translate
from core.postprocess import merge_eyeglasses_labels, clean_labels_with_connectivity
from core.measurements import (
    nasal_index,
    nose_dimensions,
    palpebral_fissure_length,
    intercanthal_biocular,
    eye_dimensions,
    eyebrow_measures,
    eyebrow_dimensions,
    mouth_measures,
    mouth_dimensions,
    facial_angles,
)
from core.parts import nose_part, eye_part, eyebrow_part, mouth_part

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# CSV column order: primary dimensionless measures and their landmark inputs
# first, then the secondary segmentation-derived signals (diagnostic only).
CSV_FIELDS = [
    "image",
    "nasal_index", "nasal_category", "nasal_width", "nasal_height",
    "palpebral_fissure_left", "palpebral_fissure_right",
    "intercanthal_width", "biocular_width", "canthal_index",
    "mouth_width", "mouth_nose_ratio",
    "eyebrow_length_left", "eyebrow_tilt_left",
    "eyebrow_length_right", "eyebrow_tilt_right",
    "nasal_tip_angle",
    # Secondary (segmentation, diagnostic only)
    "nose_width", "nose_length", "nose_area", "nose_volume",
    "eye_area_left", "eye_volume_left", "eye_area_right", "eye_volume_right",
    "brow_area_left", "brow_volume_left", "brow_area_right", "brow_volume_right",
    "mouth_area", "mouth_volume",
]


def load_models():
    print("[*] Starting the system...")

    device = get_device()
    config = ModelConfig(device=device)

    recon_model, face_detector = load_3ddfa_models(config)
    seg_processor, seg_model = load_segformer_models(device)

    print("[+] All models successfully loaded into memory.")

    return {
        "device": device,
        "config": config,
        "recon_model": recon_model,
        "face_detector": face_detector,
        "seg_processor": seg_processor,
        "seg_model": seg_model,
    }


def run_pipeline(models, image_path):
    """
    Run the full pipeline on one image and return everything the caller needs:
    a flat measurements row (for CSV), the per-part results and facial-level
    measures (for the viewer), and the raw arrays the panel draws.
    """
    device = models["device"]
    config = models["config"]
    recon_model = models["recon_model"]
    face_detector = models["face_detector"]
    seg_processor = models["seg_processor"]
    seg_model = models["seg_model"]

    # Image
    image = load_and_crop_image(image_path, config)

    # Segmentation
    labels = run_segmentation(image, seg_processor, seg_model, device)

    # 3D reconstruction
    results, trans_params = run_3ddfa(image, face_detector, recon_model, device)

    # Back projection
    points = project_to_original_image(results, recon_model, trans_params)

    ldm68_idx = recon_model.ldm68.cpu().numpy()

    # Map segmentation to mesh vertices
    v_labels = map_segmentation_to_mesh(labels, points)

    # Canonical (unposed) 3D shape: the correct space for mirror-based
    # symmetry and neighbor-based cleanup, independent of head rotation
    canonical_shape = results['face_shape'][0]

    # Fold the eyeglasses mask into the eyes so glasses don't blank the eye
    # region (runs on the canonical shape, before symmetry/cleanup)
    v_labels = merge_eyeglasses_labels(canonical_shape, v_labels, label2id, ldm68_idx)

    # Apply symmetry correction
    v_labels_sym = apply_symmetry(
        points=canonical_shape,
        labels=v_labels,
        results=results,
        symmetric_translate=symmetric_translate
    )

    # Enforce nose/lip symmetry against head-turn direction
    v_labels_sym = apply_nose_lip_symmetry(
        points=canonical_shape,
        labels=v_labels_sym,
        angle=results['angle'],
        label2id=label2id,
        symmetric_translate=symmetric_translate
    )

    # Clean small noisy regions
    v_labels_clean = clean_labels_with_connectivity(
        points=canonical_shape,
        labels=v_labels_sym,
        label2id=label2id
    )

    # Anthropometric measurements
    v3d = results['v3d'][0]

    nasal_measurements = nasal_index(v3d, ldm68_idx)
    print(f"[+] Nasal Index: {nasal_measurements['nasal_index']:.2f} ({nasal_measurements['category']})")

    # Segmentation-derived nose dimensions: kept as a secondary/diagnostic
    # signal only (unit-less and unreliable -- the mask over-segments).
    nose_shape = nose_dimensions(canonical_shape, v_labels_clean, label2id)
    print(f"[.] Nose (segmentation, diagnostic) width: {nose_shape['nose_width']:.2f}, "
          f"length: {nose_shape['nose_length']:.2f}, area: {nose_shape['nose_area']:.2f}, "
          f"volume: {nose_shape['nose_volume']:.2f}")

    fissure = {side: palpebral_fissure_length(v3d, ldm68_idx, side) for side in ("left", "right")}
    eye_dims = {side: eye_dimensions(canonical_shape, v_labels_clean, label2id, side) for side in ("left", "right")}
    print(f"[+] Palpebral fissure length: left {fissure['left']:.2f}, right {fissure['right']:.2f}")

    # Eyebrows: exact length + dimensionless, pose-invariant tilt (primary),
    # plus segmentation-hull dimensions (secondary/diagnostic).
    brow_meas = {side: eyebrow_measures(canonical_shape, ldm68_idx, side) for side in ("left", "right")}
    brow_dims = {side: eyebrow_dimensions(canonical_shape, v_labels_clean, label2id, side) for side in ("left", "right")}
    print(f"[+] Eyebrow tilt: left {brow_meas['left']['eyebrow_tilt']:.1f}°, "
          f"right {brow_meas['right']['eyebrow_tilt']:.1f}°")

    # Mouth: exact landmark width + dimensionless mouth/nose ratio (primary),
    # plus the segmentation-hull dimensions (secondary/diagnostic).
    mouth_meas = mouth_measures(v3d, ldm68_idx)
    mouth_shape = mouth_dimensions(canonical_shape, v_labels_clean, label2id)
    print(f"[+] Mouth width: {mouth_meas['mouth_width']:.2f}, "
          f"mouth/nose ratio: {mouth_meas['mouth_nose_ratio']:.2f}")

    # Bilateral eye measures live at the facial level, not inside a single eye
    eye_bilateral = intercanthal_biocular(v3d, ldm68_idx)
    print(f"[+] Intercanthal: {eye_bilateral['intercanthal_width']:.2f}, "
          f"biocular: {eye_bilateral['biocular_width']:.2f}, "
          f"canthal index: {eye_bilateral['canthal_index']:.2f}")

    # Dimensionless profile angle, also facial-level (exact midline landmarks)
    angles = facial_angles(v3d, ldm68_idx)
    print(f"[+] Nasal tip angle: {angles['nasal_tip_angle']:.1f}°")

    # Facial-level measures (dimensionless ratios and angles): they have no
    # per-part mask, so they are reported as a summary rather than a 3D view.
    facial_measures = {
        "Canthal index": f"{eye_bilateral['canthal_index']:.2f}",
        "Nasal tip angle": f"{angles['nasal_tip_angle']:.1f}°",
    }

    # Measured parts, for the viewer's per-part isolated 3D views
    ldm68_canonical = canonical_shape[ldm68_idx]
    parts = [
        nose_part(
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            nasal_measurements=nasal_measurements,
            nose_shape=nose_shape,
            ldm68_canonical=ldm68_canonical,
        ),
        eye_part(
            side="left",
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            fissure_length=fissure["left"],
            eye_dims=eye_dims["left"],
            ldm68_canonical=ldm68_canonical,
        ),
        eye_part(
            side="right",
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            fissure_length=fissure["right"],
            eye_dims=eye_dims["right"],
            ldm68_canonical=ldm68_canonical,
        ),
        eyebrow_part(
            side="left",
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            brow_meas=brow_meas["left"],
            brow_dims=brow_dims["left"],
            ldm68_canonical=ldm68_canonical,
        ),
        eyebrow_part(
            side="right",
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            brow_meas=brow_meas["right"],
            brow_dims=brow_dims["right"],
            ldm68_canonical=ldm68_canonical,
        ),
        mouth_part(
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            mouth_meas=mouth_meas,
            mouth_shape=mouth_shape,
            ldm68_canonical=ldm68_canonical,
        ),
    ]

    # Flat row for CSV export
    row = {
        "image": image_path,
        "nasal_index": nasal_measurements["nasal_index"],
        "nasal_category": nasal_measurements["category"],
        "nasal_width": nasal_measurements["nasal_width"],
        "nasal_height": nasal_measurements["nasal_height"],
        "palpebral_fissure_left": fissure["left"],
        "palpebral_fissure_right": fissure["right"],
        "intercanthal_width": eye_bilateral["intercanthal_width"],
        "biocular_width": eye_bilateral["biocular_width"],
        "canthal_index": eye_bilateral["canthal_index"],
        "mouth_width": mouth_meas["mouth_width"],
        "mouth_nose_ratio": mouth_meas["mouth_nose_ratio"],
        "eyebrow_length_left": brow_meas["left"]["eyebrow_length"],
        "eyebrow_tilt_left": brow_meas["left"]["eyebrow_tilt"],
        "eyebrow_length_right": brow_meas["right"]["eyebrow_length"],
        "eyebrow_tilt_right": brow_meas["right"]["eyebrow_tilt"],
        "nasal_tip_angle": angles["nasal_tip_angle"],
        "nose_width": nose_shape["nose_width"],
        "nose_length": nose_shape["nose_length"],
        "nose_area": nose_shape["nose_area"],
        "nose_volume": nose_shape["nose_volume"],
        "eye_area_left": eye_dims["left"]["eye_area"],
        "eye_volume_left": eye_dims["left"]["eye_volume"],
        "eye_area_right": eye_dims["right"]["eye_area"],
        "eye_volume_right": eye_dims["right"]["eye_volume"],
        "brow_area_left": brow_dims["left"]["brow_area"],
        "brow_volume_left": brow_dims["left"]["brow_volume"],
        "brow_area_right": brow_dims["right"]["brow_area"],
        "brow_volume_right": brow_dims["right"]["brow_volume"],
        "mouth_area": mouth_shape["mouth_area"],
        "mouth_volume": mouth_shape["mouth_volume"],
    }

    # Label names
    label_names = getattr(seg_model.config, "id2label", None)
    if label_names is not None:
        label_names = {int(key): value for key, value in label_names.items()}

    return {
        "row": row,
        "image": image,
        "labels": labels,
        "results": results,
        "points": points,
        "vertex_labels": v_labels_clean,
        "label_names": label_names,
        "parts": parts,
        "facial_measures": facial_measures,
    }


def gather_images(paths):
    """Expand the given paths into a flat list of image files (directories are
    scanned for supported image extensions)."""
    files = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            files.append(str(p))
    return files


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Measurements written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AnthroFace3D: craniofacial anthropometric measurements from a 2D image."
    )
    parser.add_argument(
        "images", nargs="*", default=["examples/foto_5.jpg"],
        help="image files or directories to process (default: examples/foto_5.jpg)"
    )
    parser.add_argument(
        "--csv", metavar="PATH",
        help="write all measurements to this CSV file"
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="skip the interactive results viewer (implied when processing more than one image)"
    )
    args = parser.parse_args()

    image_paths = gather_images(args.images)
    if not image_paths:
        parser.error("no images to process")

    models = load_models()

    rows = []
    last = None
    for image_path in image_paths:
        print(f"\n[*] Processing {image_path} ...")
        outputs = run_pipeline(models, image_path)
        rows.append(outputs["row"])
        last = outputs

    if args.csv:
        write_csv(rows, args.csv)

    # The interactive viewer only makes sense for a single image.
    show_viewer = (len(image_paths) == 1) and not args.no_viewer
    if show_viewer:
        panel_path = launch_results_viewer(
            image=last["image"],
            labels=last["labels"],
            results=last["results"],
            points=last["points"],
            vertex_labels=last["vertex_labels"],
            label_names=last["label_names"],
            parts=last["parts"],
            facial_measures=last["facial_measures"],
        )
        print(f"[+] Results panel saved to: {panel_path}")

    print("[+] Pipeline completed successfully")


if __name__ == "__main__":
    main()
