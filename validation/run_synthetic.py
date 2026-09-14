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

# Perturbation std as a fraction of the anchor's identity RMS.
SIGMA_FRAC = 0.6
N_SAMPLES = 40          # samples per anchor
SEED = 0

# Real anchors (git-ignored); results are pooled across them.
ANCHORS = [
    "examples/person-5896670_1920.jpg",   # caucasian
    "examples/mebeauty_black_1.jpg",       # black
    "examples/mebeauty_asian_13.jpg",      # asian
]

# The nine dimensionless descriptors (scale-free ratios and angles).
MEASURES = [
    "nasal_index",
    "canthal_index",
    "fissure_biocular_ratio",
    "mouth_nose_ratio",
    "mouth_biocular_ratio",
    "nasal_tip_angle",
    "eyebrow_tilt",
    "facial_index",
    "naso_intercanthal_ratio",
]


def run(anchors=ANCHORS, n_samples=N_SAMPLES, sigma_frac=SIGMA_FRAC, seed=SEED):
    device = get_device()
    config = ModelConfig(device=device)
    recon_model, face_detector = load_3ddfa_models(config)
    ldm68_idx = recon_model.ldm68.cpu().numpy()

    total = len(anchors) * n_samples
    print(f"\n=== Experiment 2: synthetic round-trip validation ===")
    print(f"{len(anchors)} anchors x {n_samples} samples = {total} round-trips\n")

    # Pooled samples across all anchors.
    rows_true = {m: [] for m in MEASURES}
    rows_hat = {m: [] for m in MEASURES}
    done = 0

    for a_i, image_path in enumerate(anchors, start=1):
        tag = Path(image_path).stem
        image = load_and_crop_image(image_path, config)
        alpha_base = base_alpha_from_image(image, recon_model, face_detector, device)
        rms = float(torch.sqrt((alpha_base[0, :80] ** 2).mean()))
        sigma = sigma_frac * rms
        # Anchor's own true descriptors, so the console shows the spread of
        # facial types the pooled result is built from.
        base_true = true_measures(recon_model, alpha_base, ldm68_idx)
        print(f"[anchor {a_i}/{len(anchors)}] {tag}  RMS={rms:.3f} sigma={sigma:.3f}"
              f"  NI={base_true['nasal_index']:.1f} facial={base_true['facial_index']:.1f}"
              f" tilt={base_true['eyebrow_tilt']:.1f}")

        generator = torch.Generator().manual_seed(seed + a_i)
        for i in range(n_samples):
            alpha = sample_identity(alpha_base, sigma, generator)
            t = true_measures(recon_model, alpha, ldm68_idx)
            h = recovered_measures(recon_model, alpha, ldm68_idx, device)
            for m in MEASURES:
                rows_true[m].append(t[m])
                rows_hat[m].append(h[m])
            done += 1
            print(f"  [anchor {a_i}/{len(anchors)} {tag}] sample {i + 1:3d}/{n_samples}"
                  f"  (global {done:3d}/{total})  NI true={t['nasal_index']:5.1f} hat={h['nasal_index']:5.1f}",
                  flush=True)
        print()

    print(f"=== pooled results over {done} samples ({len(anchors)} anchors) ===")
    print(f"{'measure':24s} {'true mean±std':>16s} {'MAE':>8s} {'bias':>8s} {'rel MAE':>9s}")
    for m in MEASURES:
        t = np.array(rows_true[m])
        h = np.array(rows_hat[m])
        mae = np.mean(np.abs(h - t))
        bias = np.mean(h - t)
        rel = mae / np.mean(np.abs(t)) * 100
        print(f"{m:24s} {t.mean():8.2f}±{t.std():5.2f} {mae:8.2f} {bias:+8.2f} {rel:8.1f}%")

    print("\n(MAE/bias measure how well the pipeline recovers a known measure under ideal,")
    print(" in-model conditions -- an optimistic floor, not the real-world error: the")
    print(" synthetic faces come from the same 3DMM, so out-of-model error is invisible.)")


if __name__ == "__main__":
    run()
