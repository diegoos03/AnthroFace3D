import numpy as np

from core.measurements import nose_dimensions, eye_dimensions

# Part order inside face_model.npy's 'annotation' array (the 3DMM's own
# per-vertex part membership, usable as ground truth for the mask-based
# measures).
ANNOTATION_PARTS = [
    "right_eye", "left_eye", "right_eyebrow", "left_eyebrow",
    "nose", "up_lip", "down_lip", "skin",
]


def annotation_ground_truth_labels(canonical_shape, annotation, label2id):
    """
    Per-vertex labels built from the 3DMM's own part annotation, to stand in as
    ground truth for the segmentation-based measures. Eyes are assigned to
    l_eye/r_eye by their x-side (matching the segmentation convention we
    verified: subject's right eye sits at x<0), so the naming in the annotation
    array never has to be trusted.
    """
    n_vertices = canonical_shape.shape[0]
    parts = dict(zip(ANNOTATION_PARTS, annotation))

    labels = np.full(n_vertices, label2id["background"], dtype=np.int32)
    labels[np.asarray(parts["nose"])] = label2id["nose"]

    for eye_key in ("right_eye", "left_eye"):
        idx = np.asarray(parts[eye_key])
        mean_x = canonical_shape[idx][:, 0].mean()
        labels[idx] = label2id["r_eye"] if mean_x < 0 else label2id["l_eye"]

    return labels


def compare_segmentation_measures(canonical_shape, seg_labels, gt_labels, label2id):
    """
    Compute each mask-based measure twice on the same reconstructed mesh -- once
    with the SegFormer vertex labels, once with the annotation ground truth --
    and return (part, measure, seg_value, gt_value) rows.
    """
    rows = []

    seg_nose = nose_dimensions(canonical_shape, seg_labels, label2id)
    gt_nose = nose_dimensions(canonical_shape, gt_labels, label2id)
    for measure in ("nose_width", "nose_area", "nose_volume"):
        rows.append(("nose", measure, seg_nose[measure], gt_nose[measure]))

    for side in ("left", "right"):
        seg_eye = eye_dimensions(canonical_shape, seg_labels, label2id, side)
        gt_eye = eye_dimensions(canonical_shape, gt_labels, label2id, side)
        for measure in ("eye_area", "eye_volume"):
            rows.append((f"{side}_eye", measure, seg_eye[measure], gt_eye[measure]))

    return rows
