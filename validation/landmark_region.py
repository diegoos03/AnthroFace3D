import numpy as np
from scipy.spatial import cKDTree

from core.constants import label2id
from core.measurements import (
    NASION, PRONASALE, SUBNASALE, ALA_LEFT, ALA_RIGHT,
    EYE_CORNERS, BROW_ENDS,
    CHEILION_LEFT, CHEILION_RIGHT, LABIALE_SUPERIUS, LABIALE_INFERIUS,
)

# (name, landmark index, display region, segmentation labels of that region).
# Indices imported from core.measurements so the test uses the pipeline constants.
LANDMARK_REGIONS = [
    ("nasion",            NASION,                        "nose",   ("nose",)),
    ("pronasale",         PRONASALE,                     "nose",   ("nose",)),
    ("subnasale",         SUBNASALE,                     "nose",   ("nose",)),
    ("ala_right",         ALA_RIGHT,                     "nose",   ("nose",)),
    ("ala_left",          ALA_LEFT,                      "nose",   ("nose",)),
    ("eye_right_outer",   EYE_CORNERS["right"]["outer"], "r_eye",  ("r_eye",)),
    ("eye_right_inner",   EYE_CORNERS["right"]["inner"], "r_eye",  ("r_eye",)),
    ("eye_left_inner",    EYE_CORNERS["left"]["inner"],  "l_eye",  ("l_eye",)),
    ("eye_left_outer",    EYE_CORNERS["left"]["outer"],  "l_eye",  ("l_eye",)),
    ("brow_right_medial", BROW_ENDS["right"]["medial"],  "r_brow", ("r_brow",)),
    ("brow_right_lateral",BROW_ENDS["right"]["lateral"], "r_brow", ("r_brow",)),
    ("brow_left_medial",  BROW_ENDS["left"]["medial"],   "l_brow", ("l_brow",)),
    ("brow_left_lateral", BROW_ENDS["left"]["lateral"],  "l_brow", ("l_brow",)),
    ("cheilion_right",    CHEILION_RIGHT,                "lip",    ("u_lip", "l_lip", "mouth")),
    ("cheilion_left",     CHEILION_LEFT,                 "lip",    ("u_lip", "l_lip", "mouth")),
    ("labiale_superius",  LABIALE_SUPERIUS,              "lip",    ("u_lip", "l_lip", "mouth")),
    ("labiale_inferius",  LABIALE_INFERIUS,              "lip",    ("u_lip", "l_lip", "mouth")),
]


def landmark_region_report(canonical_shape, seg_vertex_labels, ldm68_idx):
    """For each landmark, the graded distance to its expected region (0 if
    inside, else distance to the nearest region vertex over the alar width).
    Returns (rows, scale), row = (name, region, inside, distance, rel, status)."""
    seg_vertex_labels = np.asarray(seg_vertex_labels).reshape(-1)
    scale = float(np.linalg.norm(canonical_shape[ldm68_idx[ALA_LEFT]]
                                 - canonical_shape[ldm68_idx[ALA_RIGHT]]))

    rows = []
    for name, ldm_i, region, region_labels in LANDMARK_REGIONS:
        vertex = int(ldm68_idx[ldm_i])
        target_ids = [label2id[r] for r in region_labels]
        region_mask = np.isin(seg_vertex_labels, target_ids)

        if not region_mask.any():
            rows.append((name, region, False, float("nan"), float("nan"), "region_absent"))
            continue

        inside = bool(region_mask[vertex])
        if inside:
            distance = 0.0
        else:
            reg_pts = canonical_shape[region_mask]
            distance = float(cKDTree(reg_pts).query(canonical_shape[vertex])[0])
        rows.append((name, region, inside, distance, distance / scale, "ok"))

    return rows, scale
