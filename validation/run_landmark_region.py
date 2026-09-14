import argparse
import csv
import io
import random
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from validation.landmark_region import LANDMARK_REGIONS, landmark_region_report
from validation.reconstruct import load_reconstruction_models, reconstruct_from_image

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "e1_fairface"
RESULTS_DIR = REPO / "results" / "e1_landmark_region"
DEFAULT_SOURCE = Path("/home/diegoos_03/Escritorio/fairface_cribado/images")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def _candidate_order(source, n, seed):
    """Ordered images to try, and whether to freeze successes. Reuses the frozen
    sample if it already holds n, else shuffles the source with `seed`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frozen = _list_images(DATA_DIR)
    if len(frozen) >= n:
        return frozen[:n], False
    if not source.is_dir():
        sys.exit(f"[!] frozen sample has {len(frozen)}/{n} images and source "
                 f"folder does not exist: {source}")
    pool = _list_images(source)
    random.Random(seed).shuffle(pool)
    return pool, True


def main():
    ap = argparse.ArgumentParser(description="Landmark vs SegFormer region coherence over a FairFace sample.")
    ap.add_argument("--n", type=int, default=200, help="number of successful images to reach")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (reproducible)")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="folder to sample from on first run")
    args = ap.parse_args()

    candidates, freeze = _candidate_order(args.source, args.n, args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = load_reconstruction_models()

    per_image_path = RESULTS_DIR / "per_image.csv"
    per_image = open(per_image_path, "w", newline="")
    writer = csv.writer(per_image)
    writer.writerow(["image", "landmark", "region", "inside", "distance", "rel_distance", "status"])

    names = [r[0] for r in LANDMARK_REGIONS]
    inside = {name: [] for name in names}   # rel distances of ok rows (0 if inside)
    absent = {name: 0 for name in names}    # times the region was missing
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    for img in candidates:
        if n_ok >= args.n:
            break
        try:
            with redirect_stdout(io.StringIO()):
                data = reconstruct_from_image(str(img), models=models)
                rows, _ = landmark_region_report(
                    data["canonical_shape"], data["seg_vertex_labels"], data["ldm68_idx"])
        except Exception as e:  # a bad image must not stop the run; draw another
            n_fail += 1
            print(f"[fail {n_fail}] {img.name}: {type(e).__name__}", flush=True)
            continue

        if freeze:
            shutil.copy2(img, DATA_DIR / img.name)
        n_ok += 1
        for name, region, is_in, dist, rel, status in rows:
            writer.writerow([img.name, name, region, is_in, f"{dist:.6f}", f"{rel:.6f}", status])
            if status == "region_absent":
                absent[name] += 1
            else:
                inside[name].append(rel)
        per_image.flush()

        elapsed = time.time() - t0
        print(f"[{n_ok:4d}/{args.n}] {img.name:24s} | {elapsed/n_ok:4.1f}s/img "
              f"| fails so far {n_fail}", flush=True)

    per_image.close()

    if n_ok < args.n:
        print(f"\n[!] only {n_ok}/{args.n} images succeeded (source exhausted).")

    # ---- aggregate summary ----
    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["landmark", "region", "n", "hit_rate",
                     "mean_rel", "mean_rel_miss", "max_rel", "region_absent"])
        print(f"\n=== Experiment 1: landmark vs SegFormer region ({n_ok} images) ===")
        print(f"{'landmark':20s} {'region':7s} {'hit':>6s} {'meanΔ':>7s} {'missΔ':>7s} {'maxΔ':>7s} {'absent':>7s}")
        for name, _, region, _ in LANDMARK_REGIONS:
            rels = np.array(inside[name], dtype=float)
            n = len(rels)
            hit = float((rels == 0).mean()) if n else float("nan")
            mean_rel = float(rels.mean()) if n else float("nan")
            miss = rels[rels > 0]
            mean_miss = float(miss.mean()) if miss.size else 0.0
            max_rel = float(rels.max()) if n else float("nan")
            sw.writerow([name, region, n, f"{hit:.4f}", f"{mean_rel:.4f}",
                         f"{mean_miss:.4f}", f"{max_rel:.4f}", absent[name]])
            print(f"{name:20s} {region:7s} {hit:6.1%} {mean_rel:7.1%} "
                  f"{mean_miss:7.1%} {max_rel:7.1%} {absent[name]:7d}")

    all_rels = np.array([r for name in names for r in inside[name]], dtype=float)
    print(f"\noverall: {(all_rels == 0).mean():.1%} of landmarks inside their region; "
          f"mean offset {all_rels.mean():.1%}, max {all_rels.max():.1%} of alar width")
    print(f"images processed {n_ok}, failed/replaced {n_fail}")
    print(f"[i] per-image rows -> {per_image_path}")
    print(f"[i] summary        -> {summary_path}")


if __name__ == "__main__":
    main()
