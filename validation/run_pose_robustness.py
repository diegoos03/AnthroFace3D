import os
from pathlib import Path

_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import numpy as np

from core.model_loader import get_device, load_3ddfa_models
from core.config import ModelConfig
from core.preprocess import load_and_crop_image
from validation.synthetic import (
    base_alpha_from_image,
    set_pose,
    true_measures,
    recovered_measures,
)

# Yaw angles (degrees) to render the same identity at. The measures are
# computed on the reconstructed 3D shape and claim to be pose-invariant; this
# checks whether they actually hold as self-occlusion grows with yaw.
YAW_DEGREES = [-45, -30, -15, 0, 15, 30, 45]

MEASURES = ["nasal_index", "mouth_nose_ratio", "canthal_index", "nasal_tip_angle"]


def run(image_path="examples/foto_5.jpg"):
    device = get_device()
    config = ModelConfig(device=device)
    recon_model, face_detector = load_3ddfa_models(config)
    ldm68_idx = recon_model.ldm68.cpu().numpy()

    image = load_and_crop_image(image_path, config)
    alpha_base = base_alpha_from_image(image, recon_model, face_detector, device)

    # Ground truth: measures on the (pose-invariant) generated shape.
    t = true_measures(recon_model, set_pose(alpha_base, 0, 0, 0), ldm68_idx)

    print("\n=== Experiment 2b: pose robustness of the dimensionless measures ===")
    print(f"anchor: {image_path} | same identity rendered at varying yaw\n")

    hats = {m: [] for m in MEASURES}
    header = "  yaw   " + "".join(f"{m:>18s}" for m in MEASURES)
    print(header)
    for deg in YAW_DEGREES:
        alpha = set_pose(alpha_base, yaw=np.radians(deg))
        h = recovered_measures(recon_model, alpha, ldm68_idx, device)
        for m in MEASURES:
            hats[m].append(h[m])
        print(f"  {deg:+4d}   " + "".join(f"{h[m]:>18.2f}" for m in MEASURES))

    print("  " + "-" * (5 + 18 * len(MEASURES)))
    print("  true " + "".join(f"{t[m]:>18.2f}" for m in MEASURES))

    print(f"\n{'measure':18s} {'true':>8s} {'pose std':>9s} {'max|hat-true|':>14s}")
    for m in MEASURES:
        h = np.array(hats[m])
        drift = h.std()
        maxdev = np.max(np.abs(h - t[m]))
        print(f"{m:18s} {t[m]:8.2f} {drift:9.2f} {maxdev:14.2f}")

    print("\n(low 'pose std' and 'max|hat-true|' => the measure is genuinely pose-robust;")
    print(" growth at large |yaw| would flag reconstruction degradation from self-occlusion.)")


if __name__ == "__main__":
    run()
