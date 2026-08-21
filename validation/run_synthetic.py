import os
from pathlib import Path

_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import numpy as np
import torch

from core.model_loader import get_device, load_3ddfa_models
from core.config import ModelConfig
from core.preprocess import load_and_crop_image
from validation.synthetic import (
    base_alpha_from_image,
    sample_identity,
    true_measures,
    recovered_measures,
)

# Fraction of the anchor identity's RMS magnitude used as the perturbation
# std. Chosen so the generated nasal-index distribution stays in a human range
# (checked in the sanity block below).
SIGMA_FRAC = 0.6
N_SAMPLES = 40
SEED = 0

MEASURES = [
    "nasal_index", "mouth_nose_ratio", "canthal_index", "nasal_tip_angle", "eyebrow_tilt",
    "facial_index", "naso_intercanthal_ratio", "fissure_biocular_ratio", "mouth_biocular_ratio",
    "fissure_asymmetry", "eyebrow_length_asymmetry",
]


def run(image_path="examples/foto_5.jpg", n_samples=N_SAMPLES, sigma_frac=SIGMA_FRAC, seed=SEED):
    device = get_device()
    config = ModelConfig(device=device)
    recon_model, face_detector = load_3ddfa_models(config)
    ldm68_idx = recon_model.ldm68.cpu().numpy()

    image = load_and_crop_image(image_path, config)
    alpha_base = base_alpha_from_image(image, recon_model, face_detector, device)

    rms = float(torch.sqrt((alpha_base[0, :80] ** 2).mean()))
    sigma = sigma_frac * rms
    print(f"\n=== Experiment 2: synthetic round-trip validation ===")
    print(f"anchor: {image_path} | identity RMS={rms:.3f} | sigma={sigma:.3f} | N={n_samples}\n")

    generator = torch.Generator().manual_seed(seed)
    rows_true = {m: [] for m in MEASURES}
    rows_hat = {m: [] for m in MEASURES}

    for i in range(n_samples):
        alpha = sample_identity(alpha_base, sigma, generator)
        t = true_measures(recon_model, alpha, ldm68_idx)
        h = recovered_measures(recon_model, alpha, ldm68_idx, device)
        for m in MEASURES:
            rows_true[m].append(t[m])
            rows_hat[m].append(h[m])
        print(f"  sample {i + 1:3d}/{n_samples}  NI true={t['nasal_index']:.1f} hat={h['nasal_index']:.1f}")

    print(f"\n{'measure':18s} {'true mean±std':>16s} {'MAE':>8s} {'bias':>8s} {'rel MAE':>9s}")
    for m in MEASURES:
        t = np.array(rows_true[m])
        h = np.array(rows_hat[m])
        mae = np.mean(np.abs(h - t))
        bias = np.mean(h - t)
        rel = mae / np.mean(np.abs(t)) * 100
        print(f"{m:18s} {t.mean():8.2f}±{t.std():5.2f} {mae:8.2f} {bias:+8.2f} {rel:8.1f}%")

    print("\n(true values span a realistic range if the sanity distribution above looks human;")
    print(" MAE/bias measure how well the pipeline recovers a known measure under ideal,")
    print(" in-model conditions -- an optimistic floor, not the real-world error.)")


if __name__ == "__main__":
    run()
