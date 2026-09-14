import os
from pathlib import Path

_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import re
import csv
import numpy as np

from core.model_loader import get_device, load_3ddfa_models
from core.config import ModelConfig
from core.preprocess import load_and_crop_image
from core.inference import run_3ddfa
from validation.synthetic import dimensionless_measures

FEI_ROOT = os.path.expanduser("~/Escritorio/FEI")
OUT_CSV = "e3_pose_per_image.csv"  # written to the current working directory

REF_POSE = 11                      # neutral frontal, used as reference
SWEEP_POSES = list(range(1, 11))   # 01..10: head-rotation sequence
YAW_BINS = [(0, 15), (15, 30), (30, 45), (45, 999)]

MEASURES = [
    "nasal_index", "canthal_index", "fissure_biocular_ratio", "mouth_nose_ratio",
    "mouth_biocular_ratio", "nasal_tip_angle", "eyebrow_tilt", "facial_index",
    "naso_intercanthal_ratio",
]


def discover_subjects(root):
    subs = set()
    for f in os.listdir(root):
        m = re.match(r"(\d+)-(\d+)\.jpg$", f)
        if m:
            subs.add(int(m.group(1)))
    return sorted(subs)


def reconstruct(path, config, face_detector, recon_model, device, ldm):
    image = load_and_crop_image(path, config)
    results, _ = run_3ddfa(image, face_detector, recon_model, device)
    yaw = float(np.array(results["angle"]).ravel()[1]) * 180.0 / np.pi
    d = dimensionless_measures(results["v3d"][0], results["face_shape"][0], ldm)
    return yaw, d


def _bin(ay):
    for b in YAW_BINS:
        if b[0] <= ay < b[1]:
            return b
    return None


def _print_side_table(title, devs):
    print(f"\n=== {title} -- desviacion relativa media (%) por |yaw| ===")
    header = "descriptor".ljust(24) + "".join(
        f"{lo}-{hi if hi < 999 else '+'}".rjust(9) for lo, hi in YAW_BINS)
    print(header)
    for m in MEASURES:
        line = m.ljust(24)
        for b in YAW_BINS:
            v = devs[m][b]
            line += (f"{100*np.mean(v):6.1f}%  " if v else f"{'--':>7}  ")
        print(line)
    print("n vistas".ljust(24) + "".join(str(len(devs[MEASURES[0]][b])).rjust(9) for b in YAW_BINS))


def run(root=FEI_ROOT, subjects=None):
    device = get_device()
    config = ModelConfig(device=device)
    recon_model, face_detector = load_3ddfa_models(config)
    ldm = recon_model.ldm68.cpu().numpy()

    if subjects is None:
        subjects = discover_subjects(root)
    total = len(subjects) * (1 + len(SWEEP_POSES))
    print(f"\n=== Experimento 3: robustez a la pose (FEI) ===")
    print(f"{len(subjects)} sujetos | referencia = pose {REF_POSE} | barrido = poses {SWEEP_POSES}\n")

    rows = []
    devs = {side: {m: {b: [] for b in YAW_BINS} for m in MEASURES} for side in ("izq", "der")}
    fails = 0
    skipped_no_ref = 0
    done = 0

    for subj in subjects:
        # reference: pose 11
        done += 1
        ref_path = os.path.join(root, f"{subj}-{REF_POSE:02d}.jpg")
        ref = None
        if os.path.exists(ref_path):
            try:
                yaw_ref, dref = reconstruct(ref_path, config, face_detector, recon_model, device, ldm)
                ref = dref
                rows.append(dict(subject=subj, pose=REF_POSE, role="ref", side="-", yaw=yaw_ref, status="ok", **dref))
                print(f"  [{done:4d}/{total}] s{subj:03d} p{REF_POSE:02d} ref   yaw={yaw_ref:+6.1f}", flush=True)
            except Exception as e:
                rows.append(dict(subject=subj, pose=REF_POSE, role="ref", side="-", yaw=np.nan, status=f"fail:{type(e).__name__}"))
                print(f"  [{done:4d}/{total}] s{subj:03d} p{REF_POSE:02d} ref   FAIL -> sujeto descartado", flush=True)
        if ref is None:
            skipped_no_ref += 1
            done += len(SWEEP_POSES)  # skip this subject's sweep
            continue

        # rotation sweep
        for p in SWEEP_POSES:
            done += 1
            path = os.path.join(root, f"{subj}-{p:02d}.jpg")
            if not os.path.exists(path):
                continue
            try:
                yaw, d = reconstruct(path, config, face_detector, recon_model, device, ldm)
                side = "izq" if yaw < 0 else "der"
                rows.append(dict(subject=subj, pose=p, role="sweep", side=side, yaw=yaw, status="ok", **d))
                b = _bin(abs(yaw))
                if b is not None:
                    for m in MEASURES:
                        if ref[m] != 0:
                            devs[side][m][b].append(abs(d[m] - ref[m]) / abs(ref[m]))
                print(f"  [{done:4d}/{total}] s{subj:03d} p{p:02d} {side}  yaw={yaw:+6.1f}  NI={d['nasal_index']:5.1f}", flush=True)
            except Exception as e:
                fails += 1
                rows.append(dict(subject=subj, pose=p, role="sweep", side="-", yaw=np.nan, status=f"fail:{type(e).__name__}"))
                print(f"  [{done:4d}/{total}] s{subj:03d} p{p:02d} ?    FAIL", flush=True)

    # per-image CSV, in the current working directory
    _d = os.path.dirname(OUT_CSV)
    if _d:
        os.makedirs(_d, exist_ok=True)
    cols = ["subject", "pose", "role", "side", "yaw", "status"] + MEASURES
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # summary per turn direction
    _print_side_table("IZQUIERDA (yaw<0)", devs["izq"])
    _print_side_table("DERECHA (yaw>0)", devs["der"])
    print(f"\nsujetos descartados por fallar la referencia (pose {REF_POSE}): {skipped_no_ref}")
    print(f"fallos de reconstruccion en el barrido: {fails}")
    print(f"CSV por imagen: {OUT_CSV}")


if __name__ == "__main__":
    run()
