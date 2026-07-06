import os
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
from core.postprocess import clean_labels_with_connectivity
from core.measurements import (
    nasal_index,
    nose_dimensions,
    palpebral_fissure_length,
    intercanthal_biocular,
    eye_dimensions,
    facial_angles,
)
from core.parts import nose_part, eye_part


def main():
    print("[*] Starting the system...")
    
    device = get_device()

    # Create config 
    config = ModelConfig(device=device)

    # Load models
    recon_model, face_detector = load_3ddfa_models(config)
    seg_processor, seg_model = load_segformer_models(device)
    
    print("[+] All models successfully loaded into memory.")

    # Image
    image = load_and_crop_image("examples/foto_5.jpg", config)

    # Segmentation
    labels = run_segmentation(image, seg_processor, seg_model, device)

    # 3D reconstruction
    results, trans_params = run_3ddfa(image, face_detector, recon_model, device)

    # Back projection
    points = project_to_original_image(results, recon_model, trans_params)

    # Map segmentation to mesh vertices
    v_labels = map_segmentation_to_mesh(labels, points)

    # Canonical (unposed) 3D shape: the correct space for mirror-based
    # symmetry and neighbor-based cleanup, independent of head rotation
    canonical_shape = results['face_shape'][0]

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
    nasal_measurements = nasal_index(results['v3d'][0], recon_model.ldm68.cpu().numpy())
    print(f"[+] Nasal Index: {nasal_measurements['nasal_index']:.2f} ({nasal_measurements['category']})")

    # Segmentation-derived nose dimensions: kept as a secondary/diagnostic
    # signal only (unit-less and unreliable -- the mask over-segments).
    nose_shape = nose_dimensions(canonical_shape, v_labels_clean, label2id)
    print(f"[.] Nose (segmentation, diagnostic) width: {nose_shape['nose_width']:.2f}, "
          f"length: {nose_shape['nose_length']:.2f}, area: {nose_shape['nose_area']:.2f}, "
          f"volume: {nose_shape['nose_volume']:.2f}")

    v3d = results['v3d'][0]
    ldm68_idx = recon_model.ldm68.cpu().numpy()

    fissure = {side: palpebral_fissure_length(v3d, ldm68_idx, side) for side in ("left", "right")}
    eye_dims = {side: eye_dimensions(canonical_shape, v_labels_clean, label2id, side) for side in ("left", "right")}
    print(f"[+] Palpebral fissure length: left {fissure['left']:.2f}, right {fissure['right']:.2f}")

    # Bilateral eye measures live at the facial level, not inside a single eye
    eye_bilateral = intercanthal_biocular(v3d, ldm68_idx)
    print(f"[+] Intercanthal: {eye_bilateral['intercanthal_width']:.2f}, "
          f"biocular: {eye_bilateral['biocular_width']:.2f}, "
          f"canthal index: {eye_bilateral['canthal_index']:.2f}")

    # Dimensionless profile angles, also facial-level (midline landmarks)
    angles = facial_angles(v3d, ldm68_idx)
    print(f"[+] Nasal tip: {angles['nasal_tip_angle']:.1f}°, "
          f"nasomental: {angles['nasomental_angle']:.1f}°, "
          f"facial convexity: {angles['facial_convexity_angle']:.1f}°")

    # Facial-level measures (dimensionless ratios and angles): they have no
    # per-part mask, so they are reported as a summary rather than a 3D view.
    facial_measures = {
        "Canthal index": f"{eye_bilateral['canthal_index']:.2f}",
        "Nasal tip angle": f"{angles['nasal_tip_angle']:.1f}°",
        "Nasomental angle": f"{angles['nasomental_angle']:.1f}°",
        "Facial convexity angle": f"{angles['facial_convexity_angle']:.1f}°",
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
    ]

    # Label names
    label_names = getattr(seg_model.config, "id2label", None)
    if label_names is not None:
        label_names = {int(key): value for key, value in label_names.items()}

    # Visualization
    panel_path = launch_results_viewer(
        image=image,
        labels=labels,
        results=results,
        points=points,
        vertex_labels=v_labels_clean,
        label_names=label_names,
        parts=parts,
        facial_measures=facial_measures,
    )

    print(f"[+] Results panel saved to: {panel_path}")
    print("[+] Pipeline completed successfully")

if __name__ == "__main__":
    main()
