import numpy as np

from core.constants import label2id
from core.measurements import nasal_index
from validation.reconstruct import reconstruct_from_image
from validation.segmentation_gt import annotation_ground_truth_labels, compare_segmentation_measures


def _show_landmark_provenance(data):
    """
    Demonstrate empirically where a landmark's position comes from: it is a
    fixed vertex index into the reconstructed mesh, and the 2D landmark is just
    that vertex projected to the image plane -- it is never detected on the photo.
    """
    canonical = data["canonical_shape"]
    ldm68_idx = data["ldm68_idx"]
    v2d = data["results"]["v2d"][0]
    ldm2d = data["results"]["ldm68"][0]

    nasion_vertex = ldm68_idx[27]
    print("=== Where landmarks come from ===")
    print(f"ldm68 are constant vertex indices; nasion -> mesh vertex #{nasion_vertex}")
    print(f"nasion 3D position = that vertex of the reconstructed mesh: {canonical[nasion_vertex].round(3)}")
    print(f"are the 68 2D landmarks equal to the projection of those 3D vertices? {np.allclose(ldm2d, v2d[ldm68_idx])}")
    print()


def _show_segmentation_vs_gt(data):
    canonical = data["canonical_shape"]
    seg_labels = data["seg_vertex_labels"]
    ldm68_idx = data["ldm68_idx"]
    annotation = data["recon_model"].annotation

    gt_labels = annotation_ground_truth_labels(canonical, annotation, label2id)

    print("=== Experiment 1: SegFormer vs 3DMM annotation ground truth ===")
    nasal = nasal_index(data["results"]["v3d"][0], ldm68_idx)
    print(f"landmark reference (segmentation-independent): nasal_width = {nasal['nasal_width']:.3f}")
    print()

    rows = compare_segmentation_measures(canonical, seg_labels, gt_labels, label2id)
    print(f"{'part':10s} {'measure':12s} {'SegFormer':>10s} {'GT annot':>10s} {'rel diff':>9s}")
    for part, measure, seg_v, gt_v in rows:
        rel = (seg_v - gt_v) / gt_v * 100 if gt_v else float("nan")
        print(f"{part:10s} {measure:12s} {seg_v:10.3f} {gt_v:10.3f} {rel:8.1f}%")


def main():
    data = reconstruct_from_image("examples/foto_5.jpg")
    print()
    _show_landmark_provenance(data)
    _show_segmentation_vs_gt(data)


if __name__ == "__main__":
    main()
