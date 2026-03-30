# ==========================================
# File: core/mapping.py
# ==========================================

import numpy as np
from scipy.interpolate import NearestNDInterpolator


def map_segmentation_to_mesh(labels, face_proj_original):

    # Grid coordinates
    x = np.arange(labels.shape[0])
    y = np.arange(labels.shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')

    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = labels.ravel()

    interpolator = NearestNDInterpolator(points, values)

    v2d_label = np.array([
        interpolator(coord[1], coord[0])
        for coord in face_proj_original
    ])

    return v2d_label.astype(np.int32)