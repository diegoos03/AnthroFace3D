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
from core.measurements import nasal_index, nose_dimensions
from core.parts import nose_part


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

    nose_shape = nose_dimensions(canonical_shape, v_labels_clean, label2id)
    print(f"[+] Nose width: {nose_shape['nose_width']:.2f}, length: {nose_shape['nose_length']:.2f}, "
          f"area: {nose_shape['nose_area']:.2f}, volume: {nose_shape['nose_volume']:.2f}")

    # Measured parts, for the viewer's per-part isolated 3D views
    ldm68_canonical = canonical_shape[recon_model.ldm68.cpu().numpy()]
    parts = [
        nose_part(
            canonical_shape=canonical_shape,
            vertex_labels=v_labels_clean,
            label2id=label2id,
            nasal_measurements=nasal_measurements,
            nose_shape=nose_shape,
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
    )

    print(f"[+] Results panel saved to: {panel_path}")
    print("[+] Pipeline completed successfully")

if __name__ == "__main__":
    main()
