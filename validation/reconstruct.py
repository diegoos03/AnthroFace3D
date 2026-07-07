import os
from pathlib import Path

_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

from core.model_loader import get_device, load_3ddfa_models, load_segformer_models
from core.config import ModelConfig
from core.preprocess import load_and_crop_image
from core.segmentation import run_segmentation
from core.inference import run_3ddfa, project_to_original_image
from core.mapping import map_segmentation_to_mesh
from core.symmetry import apply_symmetry, apply_nose_lip_symmetry
from core.constants import label2id, symmetric_translate
from core.postprocess import merge_eyeglasses_labels, clean_labels_with_connectivity


def reconstruct_from_image(image_path):
    """
    Run the same image -> reconstruction + segmentation pipeline as main.py,
    returning the arrays the validation experiments need. Calls the same core
    functions in the same order, so it stays faithful to the production path.
    """
    device = get_device()
    config = ModelConfig(device=device)
    recon_model, face_detector = load_3ddfa_models(config)
    seg_processor, seg_model = load_segformer_models(device)

    image = load_and_crop_image(image_path, config)
    labels = run_segmentation(image, seg_processor, seg_model, device)
    results, trans_params = run_3ddfa(image, face_detector, recon_model, device)
    points = project_to_original_image(results, recon_model, trans_params)

    v_labels = map_segmentation_to_mesh(labels, points)
    canonical_shape = results["face_shape"][0]
    ldm68_idx = recon_model.ldm68.cpu().numpy()
    v_labels = merge_eyeglasses_labels(canonical_shape, v_labels, label2id, ldm68_idx)
    v_labels = apply_symmetry(points=canonical_shape, labels=v_labels, results=results, symmetric_translate=symmetric_translate)
    v_labels = apply_nose_lip_symmetry(points=canonical_shape, labels=v_labels, angle=results["angle"], label2id=label2id, symmetric_translate=symmetric_translate)
    v_labels = clean_labels_with_connectivity(points=canonical_shape, labels=v_labels, label2id=label2id)

    return {
        "recon_model": recon_model,
        "results": results,
        "canonical_shape": canonical_shape,
        "seg_vertex_labels": v_labels,
        "ldm68_idx": ldm68_idx,
    }
