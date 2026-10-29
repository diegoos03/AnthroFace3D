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

Requires Python 3.13 and runs on CPU. Two of the four steps below are easy to
miss. The mesh renderer has to be built on your own machine, and the model
weights have to be downloaded separately, around 450 MB of them. Neither one is
kept in the repository.

### 1. System packages

`dlib` and the mesh renderer are C++ extensions that get compiled while you
install them, so you need a compiler and CMake first. The interactive viewer
needs Tk.

```bash
# Ubuntu / Debian
sudo apt install python3.13-venv python3.13-tk build-essential cmake
```

On macOS, `xcode-select --install` and `brew install cmake python-tk@3.13`. On
Windows, install CMake and the Visual Studio Build Tools with the C++ workload;
Tk comes with the python.org installer.

### 2. Virtual environment

```bash
python3.13 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Compile the CPU mesh renderer

The renderer is a Cython extension that is not kept in the repository, so it has
to be built once. Do not skip this step. The default settings render on CPU and
fail without it.

```bash
cd external/util/cython_renderer
python setup.py build_ext -i
cd ../../..
```

### 4. Download the model weights

```bash
python scripts/download_assets.py
```

The script downloads seven files into `assets/` from the original releases,
checks the size of each one and skips anything already there. If you would
rather fetch them by hand:

| File | Source | Used for |
| --- | --- | --- |
| `face_model.npy` | [3DDFA-V3 assets][a] | 3DMM shape, expression and texture bases |
| `net_recon.pth` | [3DDFA-V3 assets][a] | ResNet-50 coefficient regressor |
| `large_base_net.pth` | [3DDFA-V3 assets][a] | internal landmark model |
| `retinaface_resnet50_2020-07-20_old_torch.pth` | [3DDFA-V3 assets][a] | face detection |
| `similarity_Lm3D_all.mat` | [3DDFA-V3 assets][a] | reference landmarks for the internal crop |
| `net_recon_mbnet.pth` | [3DDFA-V3 assets][a] | only for the MobileNet-V3 backbone |
| `shape_predictor_68_face_landmarks.dat` | [dlib][b] | 68 landmarks for the FFHQ crop |

[a]: https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/tree/main/assets
[b]: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

SegFormer is the one model you do not have to fetch yourself. It comes down from
the Hugging Face Hub the first time you run the pipeline and stays cached
afterwards, so that first run needs an internet connection.

Each of these files is published by its own authors under its own terms.
`face_model.npy` comes from the Basel Face Model, licensed by the University of
Basel, and the dlib predictor is trained on iBUG 300-W, which rules out
commercial use.

## Usage

```bash
python main.py                                  # default example image + interactive viewer
python main.py path/to/image.jpg --no-viewer    # single image, no viewer
python main.py path/to/folder/ --csv out.csv    # batch a folder, export every measure to CSV
```

Run these from the repository root. Part of the vendored 3DDFA-V3 code resolves
`assets/` relative to the working directory, so launching from anywhere else
fails to find the weights.

## Structure

- `main.py` — end-to-end pipeline with a small CLI.
- `core/` — detection, reconstruction, segmentation, 2D→3D fusion, symmetry,
  measurements and visualization.
- `external/` — vendored 3DDFA-V3 (MIT, see `external/LICENSE`).
- `validation/` — reproducible validation experiments (landmark–segmentation
  coherence, synthetic round-trip, pose robustness, demographic analysis).

## Tech stack

3DDFA-V3 · SegFormer (face-parsing) · RetinaFace · PyTorch · NumPy / SciPy ·
Matplotlib / Plotly / Tkinter.

## About

Final-degree project (TFG) in Computer Engineering, Universidad de Granada:
*3D craniofacial anthropometric measurement from 2D images with computer vision.*
