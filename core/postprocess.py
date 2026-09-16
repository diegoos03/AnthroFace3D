import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from core.measurements import EYE_CORNERS


def merge_eyeglasses_labels(points, labels, label2id, ldm68_vertex_idx):
    """Fold the eyeglasses mask (eye_g) into the nearer eye when within ~1.2x the
    palpebral-fissure length of an eye corner, the rest to skin, so glasses don't
    blank the eye region. Runs on the canonical (unposed) shape."""
    labels = labels.copy().astype(int)

    mask = labels == label2id["eye_g"]
    if not mask.any():
        return labels

    lm = points[ldm68_vertex_idx]
    left_corners = np.stack([lm[EYE_CORNERS["left"]["inner"]], lm[EYE_CORNERS["left"]["outer"]]])
    right_corners = np.stack([lm[EYE_CORNERS["right"]["inner"]], lm[EYE_CORNERS["right"]["outer"]]])

    # Scale reference: the mean palpebral fissure length across both eyes.
    fissure = 0.5 * (
        np.linalg.norm(left_corners[0] - left_corners[1])
        + np.linalg.norm(right_corners[0] - right_corners[1])
    )
    radius = 1.2 * fissure

    idx = np.where(mask)[0]
    P = points[idx]
    d_left = np.linalg.norm(P[:, None, :] - left_corners[None], axis=2).min(axis=1)
    d_right = np.linalg.norm(P[:, None, :] - right_corners[None], axis=2).min(axis=1)
    nearest = np.minimum(d_left, d_right)

    new = np.where(d_left <= d_right, label2id["l_eye"], label2id["r_eye"])
    new = np.where(nearest <= radius, new, label2id["skin"])
    labels[idx] = new

    return labels


def clean_labels_with_connectivity(points, labels, label2id, k=8):

    print("[*] Cleaning labels with connectivity...")

    labels = labels.copy().astype(int)

    # Build k-NN graph
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k+1)

    row_indices = np.repeat(np.arange(len(points)), k)
    col_indices = indices[:, 1:].flatten()
    data = np.ones(len(row_indices))

    adjacency_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(points), len(points))
    )

    unique_labels = np.unique(labels)

    # Parameters
    size_threshold = 200

    allowed_labels = [
        'background', 'skin', 'hat',
        'neck_l', 'neck', 'cloth'
    ]

    allowed_ids = [label2id[x] for x in allowed_labels if x in label2id]

    # Clean each class
    for label in unique_labels:

        if label in allowed_ids:
            continue

        label_mask = labels == label
        subgraph = adjacency_matrix[label_mask][:, label_mask]

        n_components, component_labels = connected_components(
            csgraph=subgraph,
            directed=False,
            return_labels=True
        )

        unique_components, counts = np.unique(component_labels, return_counts=True)

        # treat hair differently due to its potential fragmentation
        if label != label2id.get('hair', -1):
            size_threshold_label = counts.max()
        else:
            size_threshold_label = size_threshold

        for component_id in range(n_components):
            component_mask = component_labels == component_id
            component_size = np.sum(component_mask)

            point_indices = np.where(label_mask)[0][component_mask]

            if component_size < size_threshold_label:
                neighbor_indices = np.unique(indices[point_indices, 1:].ravel())
                neighbor_labels = labels[neighbor_indices]

                neighbor_labels = neighbor_labels[neighbor_labels != label]

                if len(neighbor_labels) > 0:
                    majority_label = np.bincount(neighbor_labels).argmax()
                    labels[point_indices] = majority_label

    print("[+] Label cleaning completed")

    return labels