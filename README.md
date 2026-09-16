# AnthroFace3D

Dimensionless craniofacial anthropometric descriptors from a single 2D image.

AnthroFace3D reconstructs a 3D face from one photo and reads a set of scale-free
anthropometric descriptors (ratios, indices and angles) off the reconstructed
mesh. A single image carries no absolute scale, so every reported measure is
dimensionless and, by construction, pose-invariant.

## Pipeline

Face detection → 3D reconstruction → 2D face parsing → label fusion → symmetry &
cleanup → measurement.

1. **Detection** — RetinaFace locates the face and crops it (FFHQ-style).
2. **3D reconstruction** — [3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3)
   (Wang et al., CVPR 2024) regresses a 3D morphable model and a dense mesh with
   68 landmarks.
3. **Segmentation** — [SegFormer](https://huggingface.co/jonathandinu/face-parsing)
   (`jonathandinu/face-parsing`, Xie et al., NeurIPS 2021) parses the image into
   19 semantic face regions.
4. **Fusion** — the 2D segmentation is projected onto the mesh vertices by
   nearest neighbour.
5. **Correction** — occluded labels are recovered through facial symmetry, and
   small mislabelled clusters are cleaned on a k-NN graph of the mesh.
6. **Measurement** — dimensionless descriptors are computed from the labelled
   mesh and its landmarks.

## Descriptors

Nine dimensionless descriptors, including nasal index, canthal index, facial
index, mouth/nose ratio, nasal tip angle and eyebrow tilt. They are derived from
3D landmark distances and angles, so they are pose-invariant; areas and volumes
taken from the segmentation masks are reported only as secondary, diagnostic
signals (unit-less and less reliable).

## Installation

Requires Python 3.13, CPU-only.

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The interactive viewer needs Tk (`python3.13-tk` on Ubuntu). Model weights
(3DMM, RetinaFace, ResNet-50, dlib landmarks) live under `assets/` and are not
versioned.

## Usage

```bash
python main.py                                  # default example image + interactive viewer
python main.py path/to/image.jpg --no-viewer    # single image, no viewer
python main.py path/to/folder/ --csv out.csv    # batch a folder, export every measure to CSV
```

## Structure

- `main.py` — end-to-end pipeline with a small CLI.
- `core/` — detection, reconstruction, segmentation, 2D→3D fusion, symmetry,
  measurements and visualization.
- `external/` — vendored 3DDFA-V3.
- `validation/` — reproducible validation experiments (landmark–segmentation
  coherence, synthetic round-trip, pose robustness, demographic analysis).

## Tech stack

3DDFA-V3 · SegFormer (face-parsing) · RetinaFace · PyTorch · NumPy / SciPy ·
Matplotlib / Plotly / Tkinter.

## About

Final-degree project (TFG) in Computer Engineering, Universidad de Granada:
*3D craniofacial anthropometric measurement from 2D images with computer vision.*
