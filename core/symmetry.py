import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree


def apply_symmetry(points, labels, results, symmetric_translate):

    print("[*] Applying occlusion symmetry correction...")

    # Per-vertex visibility (matches points/labels ordering)
    visible = results.get('visible_idx', None)

    if visible is None or visible.shape[0] != points.shape[0]:
        visible = np.ones(points.shape[0], dtype=bool)
    else:
        visible = visible.astype(bool)

    # Interpolators for label lookup
    value_interp = NearestNDInterpolator(points, labels)
    vis_interp = NearestNDInterpolator(points, visible)

    # Symmetry-based label correction
    labels_sym = []

    for vis, coord, value in zip(visible, points, labels):

        if not vis:
            mirrored = coord * np.array([-1, 1, 1])

            mirrored_visible = vis_interp(mirrored).item()

            if mirrored_visible:
                mirrored_label = int(value_interp(mirrored).item())
                value = symmetric_translate.get(mirrored_label, mirrored_label)

        labels_sym.append(value)

    print("[+] Occlusion symmetry correction completed")

    return np.array(labels_sym, dtype=np.int32)


def apply_nose_lip_symmetry(points, labels, angle, label2id, symmetric_translate):

    print("[*] Enforcing nose/lip symmetry...")

    labels = labels.copy().astype(int)

    tree = cKDTree(points)
    x_coords = points[:, 0]
    yaw_sign = np.sign(angle.flatten()[1])

    symmetrical_ids = [label2id[e] for e in ('nose', 'u_lip', 'l_lip')]

    # Side turned away from the camera: distrust it, copy the mirrored label instead
    _, mirror_idx = tree.query(points * [-1, 1, 1], k=1)
    turned_away = (x_coords * yaw_sign > 0) & np.isin(labels, symmetrical_ids)
    labels[turned_away] = [symmetric_translate[e] for e in labels[mirror_idx[turned_away]]]

    # Trusted side is nose/lip: force its mirror to match, recovering missed vertices
    _, mirror_idx2 = tree.query(points * [-1, 1, 1], k=2)
    trusted_side = (x_coords * yaw_sign < 0) & np.isin(labels, symmetrical_ids)
    labels[mirror_idx2[trusted_side]] = labels[trusted_side][:, None]

    print("[+] Nose/lip symmetry applied")

    return labels
