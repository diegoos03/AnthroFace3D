import numpy as np
from scipy.interpolate import NearestNDInterpolator

from core.constants import label2id


def map_segmentation_to_mesh(labels, face_proj_original):

    # Grid coordinates
    x = np.arange(labels.shape[0])
    y = np.arange(labels.shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')

    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = labels.ravel()

    interpolator = NearestNDInterpolator(points, values)

    # Projected pixel coords (col=x, row=y), queried in one vectorized call.
    cols = face_proj_original[:, 0]
    rows = face_proj_original[:, 1]
    v2d_label = np.asarray(interpolator(rows, cols))

    # Vertices projecting outside the image get background, not a pasted border
    # label (matters for non-frontal faces with off-frame back-facing vertices).
    out_of_bounds = (
        (cols < 0) | (cols >= labels.shape[1]) |
        (rows < 0) | (rows >= labels.shape[0])
    )
    v2d_label[out_of_bounds] = label2id["background"]

    return v2d_label.astype(np.int32)