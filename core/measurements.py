import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Standard 68-point (iBUG/300W) landmark indices used by 3DDFA-V3's ldm68
NASION = 27
PRONASALE = 30
SUBNASALE = 33
ALA_LEFT = 31
ALA_RIGHT = 35

# Mouth landmarks (iBUG/300W): cheilia 48 (subject's right, x<0), 54 (left, x>0).
CHEILION_RIGHT = 48
CHEILION_LEFT = 54
LABIALE_SUPERIUS = 51
LABIALE_INFERIUS = 57

# Eye corner landmarks (iBUG/300W), verified empirically on the canonical shape:
# the subject's right eye sits at x<0 (indices 36-41), the left at x>0 (42-47).
EYE_CORNERS = {
    "right": {"outer": 36, "inner": 39},
    "left": {"inner": 42, "outer": 45},
}

# Eyebrow endpoints (iBUG/300W), verified empirically: right brow 17-21 (x<0),
# left brow 22-26 (x>0); medial = end nearest the midline, lateral = temple end.
BROW_ENDS = {
    "right": {"medial": 21, "lateral": 17},
    "left": {"medial": 22, "lateral": 26},
}

# Face-contour landmarks (iBUG/300W) for the facial index: contour points 2/14
# approximate the bizygomatic breadth (no true zygion in the 68 set), gnathion 8.
GNATHION = 8
ZYGION_RIGHT = 2   # subject's right cheek contour (x<0)
ZYGION_LEFT = 14   # subject's left cheek contour (x>0)


def _hull_area_volume(points):
    # A convex hull needs >=4 points; empty or near-planar masks return NaN
    # (QJ joggles near-planar input so Qhull doesn't reject the flat simplex).
    if len(points) < 4:
        return float("nan"), float("nan")
    try:
        hull = ConvexHull(points)
    except QhullError:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
        except QhullError:
            return float("nan"), float("nan")
    return float(hull.area), float(hull.volume)


def nasal_index(v3d, ldm68_vertex_idx):
    """Nasal Index = (nasal width / nasal height) * 100 on the 3D mesh
    (pose-independent), classified as leptorrhine/mesorrhine/platyrrhine."""
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
    """Nose width/length and hull area/volume from the segmentation mask, on the
    canonical (unposed) shape (width/length are axis-aligned)."""
    nose_points = canonical_shape[vertex_labels == label2id["nose"]]
    if len(nose_points) == 0:
        nan = float("nan")
        return {"nose_width": nan, "nose_length": nan, "nose_area": nan, "nose_volume": nan}

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
    """Palpebral fissure length (eye width): 3D endocanthion-to-exocanthion
    distance for one eye, pose-independent."""
    landmarks = v3d[ldm68_vertex_idx]
    corners = EYE_CORNERS[side]
    length = np.linalg.norm(landmarks[corners["inner"]] - landmarks[corners["outer"]])
    return float(length)


def intercanthal_biocular(v3d, ldm68_vertex_idx):
    """Bilateral eye measures from the 68 landmarks: intercanthal (inner-inner)
    and biocular (outer-outer) widths, canthal index = intercanthal/biocular*100
    (a bare number, no anchored thresholds). All pose-independent."""
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


def mouth_measures(v3d, ldm68_vertex_idx):
    """Mouth width (cheilion-to-cheilion) and the dimensionless mouth/nose width
    ratio, from exact 68 landmarks (pose-independent, scale-free)."""
    landmarks = v3d[ldm68_vertex_idx]
    m_width = np.linalg.norm(landmarks[CHEILION_LEFT] - landmarks[CHEILION_RIGHT])
    n_width = np.linalg.norm(landmarks[ALA_LEFT] - landmarks[ALA_RIGHT])

    return {
        "mouth_width": float(m_width),
        "mouth_nose_ratio": float(m_width / n_width),
    }


def mouth_dimensions(canonical_shape, vertex_labels, label2id):
    """Convex-hull area/volume of the lip region (u_lip+l_lip+mouth) on the
    canonical shape. Secondary segmentation signal."""
    lip_ids = [label2id["u_lip"], label2id["l_lip"], label2id["mouth"]]
    mouth_points = canonical_shape[np.isin(vertex_labels, lip_ids)]
    area, volume = _hull_area_volume(mouth_points)

    return {
        "mouth_area": area,
        "mouth_volume": volume,
    }


def _angle_at(vertex, a, b):
    """Angle in degrees at `vertex` between points a and b; invariant to
    rotation, translation and scale."""
    va = a - vertex
    vb = b - vertex
    cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _acute_vector_angle(v1, v2):
    """Acute angle in degrees between two free vectors (undirected), in [0, 90]."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(min(ang, 180.0 - ang))


def eyebrow_measures(canonical_shape, ldm68_vertex_idx, side):
    """Per-eyebrow length (3D medial-to-lateral chord) and tilt (acute angle to
    the biocular axis, in the frontal x,y plane to avoid the temporal z-wrap),
    on the canonical shape so they are pose-invariant."""
    landmarks = canonical_shape[ldm68_vertex_idx]
    ends = BROW_ENDS[side]
    brow_chord = landmarks[ends["lateral"]] - landmarks[ends["medial"]]
    biocular_axis = landmarks[EYE_CORNERS["left"]["outer"]] - landmarks[EYE_CORNERS["right"]["outer"]]

    return {
        "eyebrow_length": float(np.linalg.norm(brow_chord)),
        "eyebrow_tilt": _acute_vector_angle(brow_chord[:2], biocular_axis[:2]),
    }


def eyebrow_dimensions(canonical_shape, vertex_labels, label2id, side):
    """Convex-hull area/volume of one eyebrow's mask on the canonical shape;
    side maps to 'l_brow'/'r_brow'. Secondary signal."""
    label_key = "l_brow" if side == "left" else "r_brow"
    brow_points = canonical_shape[vertex_labels == label2id[label_key]]
    area, volume = _hull_area_volume(brow_points)

    return {
        "brow_area": area,
        "brow_volume": volume,
    }


def facial_angles(v3d, ldm68_vertex_idx):
    """Dimensionless, pose-invariant profile angles from exact 68 midline
    landmarks. Only nasal_tip_angle (nasion-pronasale-subnasale) is exact; other
    classic angles need points absent from the sets and are not approximated."""
    landmarks = v3d[ldm68_vertex_idx]
    nasion = landmarks[NASION]
    pronasale = landmarks[PRONASALE]
    subnasale = landmarks[SUBNASALE]

    return {
        "nasal_tip_angle": _angle_at(pronasale, nasion, subnasale),
    }


def eye_dimensions(canonical_shape, vertex_labels, label2id, side):
    """Convex-hull area/volume of one eye's mask on the canonical shape; side
    maps to 'l_eye'/'r_eye'. Secondary signal."""
    label_key = "l_eye" if side == "left" else "r_eye"
    eye_points = canonical_shape[vertex_labels == label2id[label_key]]
    area, volume = _hull_area_volume(eye_points)

    return {
        "eye_area": area,
        "eye_volume": volume,
    }


def facial_index(v3d, ldm68_vertex_idx):
    """Facial (prosopic) index = (facial height / facial width) * 100 from exact
    68 landmarks (nasion-gnathion over cheek-contour 2/14 as bizygomatic proxy);
    pose-independent, reported as a bare number since the width is a proxy."""
    landmarks = v3d[ldm68_vertex_idx]
    facial_height = np.linalg.norm(landmarks[NASION] - landmarks[GNATHION])
    facial_width = np.linalg.norm(landmarks[ZYGION_LEFT] - landmarks[ZYGION_RIGHT])
    fi = (facial_height / facial_width) * 100

    return {
        "facial_height": float(facial_height),
        "facial_width": float(facial_width),
        "facial_index": float(fi),
    }


def facial_width_ratios(v3d, ldm68_vertex_idx):
    """Dimensionless width ratios from exact 68 landmarks (pose-independent),
    all over the biocular width: naso_intercanthal (alar/intercanthal, the
    "rule of fifths"), fissure_biocular and mouth_biocular."""
    lm = v3d[ldm68_vertex_idx]
    biocular = np.linalg.norm(lm[EYE_CORNERS["left"]["outer"]] - lm[EYE_CORNERS["right"]["outer"]])
    intercanthal = np.linalg.norm(lm[EYE_CORNERS["right"]["inner"]] - lm[EYE_CORNERS["left"]["inner"]])
    fissure_r = np.linalg.norm(lm[EYE_CORNERS["right"]["inner"]] - lm[EYE_CORNERS["right"]["outer"]])
    fissure_l = np.linalg.norm(lm[EYE_CORNERS["left"]["inner"]] - lm[EYE_CORNERS["left"]["outer"]])
    nasal_width = np.linalg.norm(lm[ALA_LEFT] - lm[ALA_RIGHT])
    mouth_width = np.linalg.norm(lm[CHEILION_LEFT] - lm[CHEILION_RIGHT])

    return {
        "naso_intercanthal_ratio": float(nasal_width / intercanthal),
        "fissure_biocular_ratio": float(0.5 * (fissure_r + fissure_l) / biocular),
        "mouth_biocular_ratio": float(mouth_width / biocular),
    }


def facial_asymmetry(v3d, canonical_shape, ldm68_vertex_idx):
    """Left/right asymmetry as |L-R|/(L+R) for fissure length (on v3d) and
    eyebrow length (on the canonical shape). Dimensionless."""
    lm = v3d[ldm68_vertex_idx]
    fissure_r = np.linalg.norm(lm[EYE_CORNERS["right"]["inner"]] - lm[EYE_CORNERS["right"]["outer"]])
    fissure_l = np.linalg.norm(lm[EYE_CORNERS["left"]["inner"]] - lm[EYE_CORNERS["left"]["outer"]])

    cn = canonical_shape[ldm68_vertex_idx]
    brow_r = np.linalg.norm(cn[BROW_ENDS["right"]["lateral"]] - cn[BROW_ENDS["right"]["medial"]])
    brow_l = np.linalg.norm(cn[BROW_ENDS["left"]["lateral"]] - cn[BROW_ENDS["left"]["medial"]])

    def rel(a, b):
        return float(abs(a - b) / (a + b)) if (a + b) > 0 else float("nan")

    return {
        "fissure_asymmetry": rel(fissure_l, fissure_r),
        "eyebrow_length_asymmetry": rel(brow_l, brow_r),
    }
