import numpy as np

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
