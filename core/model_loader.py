# ==========================================
# File: core/model_loader.py
# ==========================================
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Import 3DDFA_v3
from external.face_box import face_box
from external.model.recon import face_model

# Import config
from core.config import ModelConfig


# -----------------------------
# Device
# -----------------------------

# Note: GPU-related issues because of Python version, we default to CPU.
#   def get_device():
#        """Detect and return available hardware (GPU or CPU)."""
#        device = "cuda" if torch.cuda.is_available() else "cpu"
#        print(f"[*] Device initialized: {device}")
#        return device

def get_device():
    device = "cpu"
    print(f"[*] Device initialized: {device}")
    return device


# -----------------------------
# 3DDFA Loader
# -----------------------------
def load_3ddfa_models(config: ModelConfig):
    """
    Load geometric models:
    - Face detector (RetinaFace)
    - 3D face reconstruction model (ResNet50 backbone)
    """
    print("[*] Loading geometric models (3DDFA-V3)...")

    recon_model = face_model(config)
    facebox_detector = face_box(config).detector

    return recon_model, facebox_detector


# -----------------------------
# SegFormer Loader (with cache)
# -----------------------------
_segformer_cache = None


def load_segformer_models(device):
    """
    Load semantic segmentation model (SegFormer).
    Uses caching to avoid reloading weights multiple times.
    """
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