import os
from pathlib import Path

import numpy as np

from core.segmentation import visualize_segmentation


# Keep matplotlib cache inside the project to avoid permission warnings.
_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib.pyplot as plt


def _to_uint8_image(image_like):
    image = np.asarray(image_like)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _plot_sampled_points(ax, image, points, max_points=2000):
    ax.imshow(image)
    if points is not None and len(points) > 0:
        stride = max(1, len(points) // max_points)
        sampled = points[::stride]
        ax.scatter(sampled[:, 0], sampled[:, 1], s=1, c="lime", alpha=0.55)
    ax.set_title("3D Back Projection")
    ax.axis("off")


def show_results_panels(image, labels, results, points, save_path="results/pipeline_panels.png", show=True):
    render_face = _to_uint8_image(results["render_face"][0])
    render_shape = _to_uint8_image(results["render_shape"][0])
    seg_overlay = visualize_segmentation(image, labels)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Cropped Input")
    axes[0].axis("off")

    axes[1].imshow(seg_overlay)
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis("off")

    axes[2].imshow(render_face)
    axes[2].set_title("Rendered Face")
    axes[2].axis("off")

    axes[3].imshow(render_shape)
    axes[3].set_title("Rendered Geometry")
    axes[3].axis("off")

    _plot_sampled_points(axes[4], image, points)

    if "ldm68" in results:
        axes[5].imshow(image)
        landmarks = results["ldm68"][0]
        axes[5].scatter(landmarks[:, 0], 223 - landmarks[:, 1], s=10, c="cyan")
        axes[5].set_title("68 Landmarks")
        axes[5].axis("off")
    else:
        axes[5].imshow(image)
        axes[5].set_title("Input Preview")
        axes[5].axis("off")

    fig.suptitle("AnthroFace3D Results", fontsize=16)
    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
