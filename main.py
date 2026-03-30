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
from core.symmetry import apply_symmetry
from core.constants import label2id, symmetric_translate
from core.postprocess import clean_labels_with_connectivity


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
    image = load_and_crop_image("examples/foto_3.jpg", config)

    # Segmentation
    labels = run_segmentation(image, seg_processor, seg_model, device)

    # 3D reconstruction
    results, trans_params = run_3ddfa(image, face_detector, recon_model, device)

    # Back projection
    points = project_to_original_image(results, recon_model, trans_params)

    # Map segmentation to mesh vertices
    v_labels = map_segmentation_to_mesh(labels, points)

    # Apply symmetry correction
    v_labels_sym = apply_symmetry(
        points=points,
        labels=v_labels,
        results=results,
        symmetric_translate=symmetric_translate
    )

    # Clean small noisy regions
    v_labels_clean = clean_labels_with_connectivity(
        points=points,
        labels=v_labels_sym,
        label2id=label2id
    )

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
    )

    print(f"[+] Results panel saved to: {panel_path}")
    print("[+] Pipeline completed successfully")

if __name__ == "__main__":
    main()
