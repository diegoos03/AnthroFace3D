import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Standard 68-point (iBUG/300W) landmark indices used by 3DDFA-V3's ldm68
NASION = 27
PRONASALE = 30
SUBNASALE = 33
ALA_LEFT = 31
ALA_RIGHT = 35

# Eye corner landmarks (iBUG/300W), verified empirically on the canonical shape:
# the subject's right eye sits at x<0 (indices 36-41), the left at x>0 (42-47).
EYE_CORNERS = {
    "right": {"outer": 36, "inner": 39},
    "left": {"inner": 42, "outer": 45},
}


def _hull_area_volume(points):
    # Near-planar masks (e.g. eyes) can make Qhull reject the flat initial
    # simplex; QJ joggles the input just enough to build a hull, at the cost
    # of a negligible perturbation (volume stays ~0 for a flat patch anyway).
    try:
        hull = ConvexHull(points)
    except QhullError:
        hull = ConvexHull(points, qhull_options="QJ")
    return float(hull.area), float(hull.volume)


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
    area, volume = _hull_area_volume(nose_points)

    leftmost = nose_points[np.argsort(nose_points[:, 0])[:3]].mean(axis=0)
    rightmost = nose_points[np.argsort(nose_points[:, 0])[-3:]].mean(axis=0)
    backmost = nose_points[np.argsort(nose_points[:, 2])[:3]].mean(axis=0)
    frontmost = nose_points[np.argsort(nose_points[:, 2])[-3:]].mean(axis=0)

    return {
        "nose_width": float(np.abs(leftmost[0] - rightmost[0])),
        "nose_length": float(np.abs(backmost[2] - frontmost[2])),
        "nose_area": area,
        "nose_volume": volume,
    }


def palpebral_fissure_length(v3d, ldm68_vertex_idx, side):
    """
    Palpebral fissure length (eye width) = 3D distance between the inner
    (endocanthion) and outer (exocanthion) corner of one eye. A euclidean
    distance between two landmarks, so pose-independent like the nasal index.

    Parameters:
        v3d               -- np.ndarray, size (N, 3), reconstructed mesh vertices
        ldm68_vertex_idx   -- np.ndarray, size (68,), vertex indices for the 68 landmarks
        side               -- "left" or "right" (subject's own side)
    """
    landmarks = v3d[ldm68_vertex_idx]
    corners = EYE_CORNERS[side]
    length = np.linalg.norm(landmarks[corners["inner"]] - landmarks[corners["outer"]])
    return float(length)


def intercanthal_biocular(v3d, ldm68_vertex_idx):
    """
    Bilateral horizontal eye measures from the 68 landmarks:
      - intercanthal width: endocanthion to endocanthion (inner corners)
      - biocular width: exocanthion to exocanthion (outer corners)
      - canthal index: (intercanthal / biocular) * 100
    All euclidean 3D distances, pose-independent. The canthal index is
    reported as a bare number: unlike the nasal index it has no established
    classification thresholds anchored in this project's references.
    """
    landmarks = v3d[ldm68_vertex_idx]
    endocanthion_r = landmarks[EYE_CORNERS["right"]["inner"]]
    endocanthion_l = landmarks[EYE_CORNERS["left"]["inner"]]
    exocanthion_r = landmarks[EYE_CORNERS["right"]["outer"]]
    exocanthion_l = landmarks[EYE_CORNERS["left"]["outer"]]

    intercanthal = np.linalg.norm(endocanthion_r - endocanthion_l)
    biocular = np.linalg.norm(exocanthion_r - exocanthion_l)
    canthal_index = (intercanthal / biocular) * 100

    return {
        "intercanthal_width": float(intercanthal),
        "biocular_width": float(biocular),
        "canthal_index": float(canthal_index),
    }


def _angle_at(vertex, a, b):
    """
    Angle in degrees at `vertex`, subtended by points a and b. Being an angle
    between two directions, it is invariant to rotation, translation and
    uniform scale -- so it needs no metric unit, unlike a raw distance.
    """
    va = a - vertex
    vb = b - vertex
    cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def facial_angles(v3d, ldm68_vertex_idx):
    """
    Dimensionless, pose-invariant profile angles from the 68 midline
    landmarks -- an ideal descriptor family given the 3DMM carries no absolute
    scale. Only angles whose three landmarks are exact are reported: no proxies.

    Currently that is a single angle:
      - nasal_tip_angle: nasion-pronasale-subnasale, all exact midline points.

    Other classic profile angles are deliberately NOT computed here, because
    the 68/106/134 landmark sets (all the same face-alignment family) lack the
    points they require and would force approximations:
      - nasolabial angle needs a columella-apex point (absent in every set;
        the 106 nostril-base points sit at subnasale depth, so proxying gives
        a near-straight, meaningless angle).
      - nasomental / facial convexity need pogonion and glabella; the sets have
        neither (menton/nasion are not those points, and no set has forehead
        landmarks). Getting those is the cephalometric-refinement route.

    Parameters:
        v3d               -- np.ndarray, size (N, 3), reconstructed mesh vertices
        ldm68_vertex_idx   -- np.ndarray, size (68,), vertex indices for the 68 landmarks
    """
    landmarks = v3d[ldm68_vertex_idx]
    nasion = landmarks[NASION]
    pronasale = landmarks[PRONASALE]
    subnasale = landmarks[SUBNASALE]

    return {
        "nasal_tip_angle": _angle_at(pronasale, nasion, subnasale),
    }


def eye_dimensions(canonical_shape, vertex_labels, label2id, side):
    """
    Convex-hull area/volume of one eye's segmentation mask, on the canonical
    (unposed) shape. Secondary signal, analogous to nose area/volume.

    Parameters:
        side -- "left" or "right" (subject's own side); maps to the
                 'l_eye'/'r_eye' segmentation labels.
    """
    label_key = "l_eye" if side == "left" else "r_eye"
    eye_points = canonical_shape[vertex_labels == label2id[label_key]]
    area, volume = _hull_area_volume(eye_points)

    return {
        "eye_area": area,
        "eye_volume": volume,
    }
