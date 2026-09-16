import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from external.face_box import face_box
from external.model.recon import face_model

from core.config import ModelConfig


# GPU disabled by design (Python-version incompatibility), CPU only.
def get_device():
    device = "cpu"
    print(f"[*] Device initialized: {device}")
    return device


def load_3ddfa_models(config: ModelConfig):
    """Load the face detector (RetinaFace) and the 3D reconstruction model."""
    print("[*] Loading geometric models (3DDFA-V3)...")

    recon_model = face_model(config)
    facebox_detector = face_box(config).detector

    return recon_model, facebox_detector


_segformer_cache = None


def load_segformer_models(device):
    """Load the SegFormer segmentation model, cached to avoid reloading weights."""
    global _segformer_cache

    if _segformer_cache is not None:
        print("[*] SegFormer already loaded (cache hit).")
        return _segformer_cache

    print("[*] Loading semantic model (SegFormer)...")

    model_id = "jonathandinu/face-parsing"

    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)

    _segformer_cache = (processor, model)
    return _segformer_cache