"""Download the pretrained weights the pipeline needs into assets/.

These files are not kept in the repository. There are seven of them and they
come to around 450 MB, so they are downloaded from the releases of the people
who published them. This script only does what the README would otherwise ask
you to do by hand.

    python scripts/download_assets.py

Anything already there with the right size is left alone, so you can run this
again after an interrupted download.
"""

import bz2
import sys
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

_HF = "https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/main/assets"

# name, source, expected size in bytes, whether the download is bz2-compressed.
# The sizes are checked after every download: a truncated transfer or an error
# page served in place of the file both show up as a mismatch.
ASSETS = [
    ("face_model.npy", f"{_HF}/face_model.npy", 103_918_535, False),
    ("net_recon.pth", f"{_HF}/net_recon.pth", 96_464_581, False),
    ("large_base_net.pth", f"{_HF}/large_base_net.pth", 27_795_941, False),
    ("retinaface_resnet50_2020-07-20_old_torch.pth",
     f"{_HF}/retinaface_resnet50_2020-07-20_old_torch.pth", 109_477_455, False),
    ("similarity_Lm3D_all.mat", f"{_HF}/similarity_Lm3D_all.mat", 994, False),
    # Only used when the reconstruction backbone is switched to MobileNet-V3.
    ("net_recon_mbnet.pth", f"{_HF}/net_recon_mbnet.pth", 13_091_101, False),
    ("shape_predictor_68_face_landmarks.dat",
     "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
     99_693_937, True),
]


def _report(done, total):
    if total <= 0:
        sys.stdout.write(f"\r    {done / 1e6:.1f} MB")
    else:
        sys.stdout.write(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB"
                         f"  ({100 * done / total:3.0f}%)")
    sys.stdout.flush()


def fetch(url, destination, compressed):
    """Stream `url` into `destination`, decompressing bz2 on the way."""
    request = urllib.request.Request(url, headers={"User-Agent": "AnthroFace3D-setup"})
    partial = destination.with_suffix(destination.suffix + ".part")
    decompressor = bz2.BZ2Decompressor() if compressed else None

    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            read += len(chunk)
            out.write(decompressor.decompress(chunk) if decompressor else chunk)
            _report(read, total)
    print()
    partial.replace(destination)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Assets directory: {ASSETS_DIR}")
    failed = []

    for name, url, size, compressed in ASSETS:
        destination = ASSETS_DIR / name
        if destination.exists() and destination.stat().st_size == size:
            print(f"[=] {name} already in place")
            continue

        print(f"[>] {name}")
        try:
            fetch(url, destination, compressed)
        except Exception as error:                      # noqa: BLE001
            print(f"[!] {name}: download failed ({error})")
            failed.append(name)
            continue

        actual = destination.stat().st_size
        if actual != size:
            print(f"[!] {name}: expected {size} bytes, got {actual}")
            failed.append(name)
        else:
            print(f"[+] {name}")

    if failed:
        print("\nThese files are missing or incomplete:")
        for name in failed:
            print(f"  - {name}")
        print(f"Re-run the script, or download them by hand into {ASSETS_DIR}.")
        return 1

    print("\n[+] All assets in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
