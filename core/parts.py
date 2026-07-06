# ==========================================
# File: core/parts.py
# ==========================================
#
# Registry of measured face parts. Each part bundles the 3D points that
# belong to it, its computed measurements, and any reference landmarks —
# everything the viewer needs to give it its own isolated 3D view. Adding a
# new measured part means adding one function here; the viewer picks it up
# without changes.

from dataclasses import dataclass, field

import numpy as np

from core.measurements import EYE_CORNERS
from core.palette import PART_COLORS


@dataclass
class PartResult:
    key: str
    display_name: str
    color: str
    points: np.ndarray                        # (N, 3) vertices belonging to this part
    measurements: dict                        # label -> formatted value string (primary: indices + landmark inputs)
    secondary: dict = field(default_factory=dict)   # label -> value; segmentation-derived, diagnostic only
    landmarks: dict = field(default_factory=dict)   # label -> (3,) point


def nose_part(canonical_shape, vertex_labels, label2id, nasal_measurements, nose_shape, ldm68_canonical=None):
    mask = vertex_labels == label2id["nose"]

    landmarks = {}
    if ldm68_canonical is not None:
        landmarks = {
            "Nasion": ldm68_canonical[27],
            "Subnasale": ldm68_canonical[33],
            "Left ala": ldm68_canonical[31],
            "Right ala": ldm68_canonical[35],
        }

    # Primary: the dimensionless index and its landmark-based inputs (stable,
    # pose-invariant). The raw widths carry no metric unit, so they only make
    # sense as the ingredients of the index, not as standalone measurements.
    measurements = {
        "Nasal Index": f"{nasal_measurements['nasal_index']:.2f} ({nasal_measurements['category']})",
        "Nasal width (landmarks)": f"{nasal_measurements['nasal_width']:.3f}",
        "Nasal height (landmarks)": f"{nasal_measurements['nasal_height']:.3f}",
    }

    # Secondary: everything derived from the segmentation mask. Unreliable
    # (the mask over-segments) and unit-less, kept only as a coherence check.
    secondary = {
        "Width (segmentation)": f"{nose_shape['nose_width']:.3f}",
        "Length (segmentation)": f"{nose_shape['nose_length']:.3f}",
        "Area (convex hull)": f"{nose_shape['nose_area']:.3f}",
        "Volume (convex hull)": f"{nose_shape['nose_volume']:.3f}",
    }

    return PartResult(
        key="nose",
        display_name="Nose",
        color=PART_COLORS["nose"],
        points=canonical_shape[mask],
        measurements=measurements,
        secondary=secondary,
        landmarks=landmarks,
    )


def eye_part(side, canonical_shape, vertex_labels, label2id, fissure_length, eye_dims, ldm68_canonical=None):
    label_key = "l_eye" if side == "left" else "r_eye"
    mask = vertex_labels == label2id[label_key]

    landmarks = {}
    if ldm68_canonical is not None:
        corners = EYE_CORNERS[side]
        landmarks = {
            "Endocanthion (inner)": ldm68_canonical[corners["inner"]],
            "Exocanthion (outer)": ldm68_canonical[corners["outer"]],
        }

    # Primary: the landmark-based fissure length (stable, pose-invariant).
    measurements = {
        "Palpebral fissure length": f"{fissure_length:.3f}",
    }

    # Secondary: segmentation-mask hull. Unreliable and unit-less; the eye is
    # near-planar so its volume is uninformative (~0). Coherence check only.
    secondary = {
        "Area (convex hull)": f"{eye_dims['eye_area']:.3f}",
        "Volume (convex hull)": f"{eye_dims['eye_volume']:.3f}",
    }

    return PartResult(
        key=f"{side}_eye",
        display_name=f"{side.capitalize()} eye",
        color=PART_COLORS[label_key],
        points=canonical_shape[mask],
        measurements=measurements,
        secondary=secondary,
        landmarks=landmarks,
    )
