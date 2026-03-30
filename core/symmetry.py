# ==========================================
# File: core/symmetry.py
# ==========================================

import numpy as np
from scipy.interpolate import NearestNDInterpolator


def apply_symmetry(points, labels, results, symmetric_translate):

    # Visibility mask from 3DDFA results
    seg_vis = results.get('seg_visible', None)

    if seg_vis is not None and seg_vis.shape[0] == points.shape[0]:
        visible = seg_vis.astype(bool)
    else:
        visible = np.ones(points.shape[0], dtype=bool)

    # Interpolators for label 
    value_interp = NearestNDInterpolator(points, labels)
    vis_interp = NearestNDInterpolator(points, visible)

    # Symmetry-based label correction
    labels_sym = []

    for vis, coord, value in zip(visible, points, labels):

        if not vis:
            mirrored = coord * np.array([-1, 1, 1])

            mirrored_visible = bool(vis_interp(mirrored))
            
            if mirrored_visible:
                mirrored_label = int(value_interp(mirrored))
                value = symmetric_translate.get(mirrored_label, mirrored_label)

        labels_sym.append(value)

    return np.array(labels_sym, dtype=np.int32)