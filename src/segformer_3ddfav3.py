import torch
import argparse
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 3DDFA_v3 imports
from external.face_box import face_box
from model.recon import face_model


def get_device():
    # Detects and returns available hardware (GPU or CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device initialized: {device}")
    return device


def load_3ddfa_models(device):
    # Initializes the face detector (RetinaFace) and the 3DMM parameter regressor (ResNet50).
    print("[*] Loading geometric models (3DDFA-V3)...")
    
    # Create a clean configuration object (replaces DotDict)
    args = type('Args', (), {})()
    args.savepath = 'examples/results'
    args.device = device
    args.iscrop = True
    args.detector = 'retinaface'
    args.ldm68 = True
    args.ldm106 = True
    args.ldm106_2d = True
    args.ldm134 = True
    args.seg_visible = True
    args.seg = True
    args.useTex = True
    args.extractTex = True
    args.backbone = 'resnet50'

    # Initialize models
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    
    return recon_model, facebox_detector


def load_segformer_models(device):
    # Initializes the processor and the Transformer model for 2D semantic segmentation.
    print("[*] Loading semantic model (SegFormer)...")
    model_id = "jonathandinu/face-parsing"
    
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    
    return processor, model


# --- MAIN EXECUTION  ---
# This ensures the models are only loaded if you run this script directly
if __name__ == "__main__":
    device = get_device()
    
    # Load models
    recon_model, face_detector = load_3ddfa_models(device)
    seg_processor, seg_model = load_segformer_models(device)
    
    print("[+] All models successfully loaded into memory.")
    
    # Your next code block will go here!