import numpy as np
from scipy.spatial import ConvexHull

# Standard 68-point (iBUG/300W) landmark indices used by 3DDFA-V3's ldm68
NASION = 27
SUBNASALE = 33
ALA_LEFT = 31
ALA_RIGHT = 35


def nasal_index(v3d, ldm68_vertex_idx):
    """
    Nasal Index (NI) = (nasal width / nasal height) * 100, computed on the
    3D mesh so it is pose-independent. Classification thresholds from the
    standard anthropometric convention (Leptorrhine/Mesorrhine/Platyrrhine).

    Parameters:
        v3d               -- np.ndarray, size (N, 3), reconstructed mesh vertices
        ldm68_vertex_idx   -- np.ndarray, size (68,), vertex indices for the 68 landmarks
    """
    landmarks = v3d[ldm68_vertex_idx]

    nasal_width = np.linalg.norm(landmarks[ALA_LEFT] - landmarks[ALA_RIGHT])
    nasal_height = np.linalg.norm(landmarks[NASION] - landmarks[SUBNASALE])
    ni = (nasal_width / nasal_height) * 100

    if ni < 70:
        category = "leptorrhine"
    elif ni < 85:
        category = "mesorrhine"
    else:
        category = "platyrrhine"

    return {
        "nasal_width": float(nasal_width),
        "nasal_height": float(nasal_height),
        "nasal_index": float(ni),
        "category": category,
    }


def nose_dimensions(canonical_shape, vertex_labels, label2id):
    """
    Nose width, length, and convex-hull area/volume, computed from the
    segmentation-labeled canonical (unposed) 3D shape. Width/length are
    axis-aligned, so this requires the unrotated canonical space rather
    than the posed/projected one.

    Parameters:
        canonical_shape -- np.ndarray, size (N, 3), unposed mesh vertices
        vertex_labels    -- np.ndarray, size (N,), per-vertex segmentation labels
        label2id          -- dict, segmentation label name -> id
    """
    nose_points = canonical_shape[vertex_labels == label2id["nose"]]
    hull = ConvexHull(nose_points)

    leftmost = nose_points[np.argsort(nose_points[:, 0])[:3]].mean(axis=0)
    rightmost = nose_points[np.argsort(nose_points[:, 0])[-3:]].mean(axis=0)
    backmost = nose_points[np.argsort(nose_points[:, 2])[:3]].mean(axis=0)
    frontmost = nose_points[np.argsort(nose_points[:, 2])[-3:]].mean(axis=0)

    return {
        "nose_width": float(np.abs(leftmost[0] - rightmost[0])),
        "nose_length": float(np.abs(backmost[2] - frontmost[2])),
        "nose_area": float(hull.area),
        "nose_volume": float(hull.volume),
    }
