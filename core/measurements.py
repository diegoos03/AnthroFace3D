import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Standard 68-point (iBUG/300W) landmark indices used by 3DDFA-V3's ldm68
NASION = 27
PRONASALE = 30
SUBNASALE = 33
ALA_LEFT = 31
ALA_RIGHT = 35

# Mouth landmarks (iBUG/300W). Cheilia (mouth corners) verified empirically on
# the canonical shape: 48 = subject's right corner (x<0), 54 = left (x>0);
# they are the most lateral outer-lip points, so they are the true cheilia.
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

# Face-contour landmarks (iBUG/300W) for the global facial index. The 68 set
# has no true zygion; contour points 2 and 14 sit at cheekbone level and
# approximate the bizygomatic breadth. Verified on the mean shape: they land in
# the anthropometric facial-index range, unlike the ear-level extremes 0/16,
# which give a bitemporal width instead. Gnathion is the lowest midline chin
# point (index 8), also used as the chin apex elsewhere.
GNATHION = 8
ZYGION_RIGHT = 2   # subject's right cheek contour (x<0)
ZYGION_LEFT = 14   # subject's left cheek contour (x>0)


def _hull_area_volume(points):
    # A convex hull needs at least 4 points; an empty or tiny mask (e.g. an
    # occluded part, or an eye hidden behind glasses on a different photo) has
    # no hull, so we report NaN rather than crashing. Near-planar masks (e.g.
    # eyes) can still make Qhull reject the flat initial simplex; QJ joggles
    # the input just enough to build a hull, at the cost of a negligible
    # perturbation (volume stays ~0 for a flat patch anyway).
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


def mouth_measures(v3d, ldm68_vertex_idx):
    """
    Mouth width and the dimensionless mouth/nose width ratio, both from exact
    68 landmarks:
      - mouth_width: cheilion-to-cheilion 3D distance (mouth corners).
      - mouth_nose_ratio: mouth width / nasal (alar) width.
    Both are euclidean distances between two landmarks, so pose-independent;
    the ratio is scale-free and comparable across faces, like the nasal index.

    Parameters:
        v3d               -- np.ndarray, size (N, 3), reconstructed mesh vertices
        ldm68_vertex_idx   -- np.ndarray, size (68,), vertex indices for the 68 landmarks
    """
    landmarks = v3d[ldm68_vertex_idx]
    m_width = np.linalg.norm(landmarks[CHEILION_LEFT] - landmarks[CHEILION_RIGHT])
    n_width = np.linalg.norm(landmarks[ALA_LEFT] - landmarks[ALA_RIGHT])

    return {
        "mouth_width": float(m_width),
        "mouth_nose_ratio": float(m_width / n_width),
    }


def mouth_dimensions(canonical_shape, vertex_labels, label2id):
    """
    Convex-hull area/volume of the lip region (upper lip + lower lip + inner
    mouth masks), on the canonical (unposed) shape. Secondary segmentation
    signal, analogous to nose/eye area/volume.
    """
    lip_ids = [label2id["u_lip"], label2id["l_lip"], label2id["mouth"]]
    mouth_points = canonical_shape[np.isin(vertex_labels, lip_ids)]
    area, volume = _hull_area_volume(mouth_points)

    return {
        "mouth_area": area,
        "mouth_volume": volume,
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


def _acute_vector_angle(v1, v2):
    """Acute angle in degrees between two free vectors (undirected), in [0, 90]."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(min(ang, 180.0 - ang))


def eyebrow_measures(canonical_shape, ldm68_vertex_idx, side):
    """
    Per-eyebrow measures from exact 68 landmarks, on the canonical (unposed)
    shape so they are pose-invariant:
      - eyebrow_length: 3D chord distance medial-to-lateral brow endpoint.
      - eyebrow_tilt: acute angle (degrees) between the brow chord and the
        biocular axis (exocanthion-to-exocanthion), measured in the FRONTAL
        (x, y) plane only. The frontal projection is deliberate: the full-3D
        brow chord wraps backward toward the temple (large z), which would
        conflate that temporal wrap with the up/down slant that "tilt" means.
        Working in the canonical frame keeps the frontal angle pose-invariant.

    Parameters:
        canonical_shape   -- np.ndarray, size (N, 3), unposed mesh vertices
        ldm68_vertex_idx   -- np.ndarray, size (68,), vertex indices for the 68 landmarks
        side               -- "left" or "right" (subject's own side)
    """
    landmarks = canonical_shape[ldm68_vertex_idx]
    ends = BROW_ENDS[side]
    brow_chord = landmarks[ends["lateral"]] - landmarks[ends["medial"]]
    biocular_axis = landmarks[EYE_CORNERS["left"]["outer"]] - landmarks[EYE_CORNERS["right"]["outer"]]

    return {
        "eyebrow_length": float(np.linalg.norm(brow_chord)),
        "eyebrow_tilt": _acute_vector_angle(brow_chord[:2], biocular_axis[:2]),
    }


def eyebrow_dimensions(canonical_shape, vertex_labels, label2id, side):
    """
    Convex-hull area/volume of one eyebrow's segmentation mask, on the
    canonical (unposed) shape. Secondary signal, analogous to eye area/volume.

    Parameters:
        side -- "left" or "right" (subject's own side); maps to the
                 'l_brow'/'r_brow' segmentation labels.
    """
    label_key = "l_brow" if side == "left" else "r_brow"
    brow_points = canonical_shape[vertex_labels == label2id[label_key]]
    area, volume = _hull_area_volume(brow_points)

    return {
        "brow_area": area,
        "brow_volume": volume,
    }


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


def facial_index(v3d, ldm68_vertex_idx):
    """
    Facial (prosopic) index = (facial height / facial width) * 100, a global
    face proportion from exact 68 landmarks. Both are euclidean distances, so
    pose-independent like the nasal index.

      - facial height: nasion(27) to gnathion(8), midline points.
      - facial width:  cheek-level contour points (2, 14), a proxy for the
        bizygomatic breadth since the 68 set has no true zygion.

    Reported as a bare number: the width is a proxy, so the classic prosopic
    categories (leptoprosopic/euryprosopic) are not applied here.
    """
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
    """
    Dimensionless width ratios from exact 68 landmarks, all euclidean-distance
    ratios (pose-independent, scale-free). The biocular width (exocanthion to
    exocanthion) is the common reference.

      - naso_intercanthal_ratio: nasal (alar) width / intercanthal width. A
        recognised facial-analysis proportion (the "rule of fifths": alar base
        approximately equal to the intercanthal distance).
      - fissure_biocular_ratio:  mean palpebral fissure length / biocular width.
      - mouth_biocular_ratio:    mouth width / biocular width.
    """
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
    """
    Left/right asymmetry indices from the paired measures, as |L-R|/(L+R).
    Fissure length is a pose-independent distance (measured on v3d); eyebrow
    length is measured on the canonical shape, consistent with the eyebrow
    measures. Both dimensionless.
    """
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
