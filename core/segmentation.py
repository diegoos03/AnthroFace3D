import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def mask_overlay(image, mask):
    structure = np.ones((3, 3))
    dilated = ndimage.grey_dilation(mask, footprint=structure)
    eroded = ndimage.grey_erosion(mask, footprint=structure)
    borders = dilated - eroded

    img = np.array(image.resize(mask.shape[::-1]))
    overlay = img + (borders != 0)[:, :, None] * 255
    overlay = np.clip(overlay, 0, 255)

    return overlay.astype(np.uint8)


def run_segmentation(image, processor, model, device):
    print("[*] Running segmentation...")

    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    labels = upsampled_logits.argmax(dim=1)[0]
    labels_np = labels.cpu().numpy()

    print("[+] Segmentation completed")

    return labels_np

def visualize_segmentation(image, labels):
    overlay = mask_overlay(image, labels)
    return overlay