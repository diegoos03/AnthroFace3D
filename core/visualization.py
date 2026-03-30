import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

import numpy as np

from core.segmentation import visualize_segmentation


# Keep matplotlib cache inside the project to avoid permission warnings.
_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


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


def create_results_figure(image, labels, results, points):
    render_face = _to_uint8_image(results["render_face"][0])
    render_shape = _to_uint8_image(results["render_shape"][0])
    seg_overlay = visualize_segmentation(image, labels)

    fig = Figure(figsize=(16, 10), dpi=100)
    axes = fig.subplots(2, 3).ravel()

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
    return fig


def create_browser_mesh_figure(results, vertex_labels=None, label_names=None, show_every=1):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    canonical_face_shape = results["v3d"][0]
    sampled_shape = canonical_face_shape[::show_every]

    if vertex_labels is None:
        sampled_labels = np.zeros(sampled_shape.shape[0], dtype=np.int32)
    else:
        sampled_labels = np.asarray(vertex_labels)[::show_every].astype(np.int32)

    if label_names is None:
        hover_names = [f"Class {int(label)}" for label in sampled_labels]
    else:
        hover_names = [label_names.get(int(label), f"Class {int(label)}") for label in sampled_labels]

    semantic_colors = {
        "background": "#8ecae6",
        "skin": "#a98274",
        "nose": "#3f6db5",
        "eye_g": "#8a8f3e",
        "l_eye": "#4ca549",
        "r_eye": "#bfc75f",
        "l_brow": "#d767d7",
        "r_brow": "#00b4d8",
        "l_ear": "#2316df",
        "r_ear": "#ec541c",
        "mouth": "#54c49b",
        "u_lip": "#eb43aa",
        "l_lip": "#d9bf50",
        "hair": "#1d3557",
        "hat": "#4f5d75",
        "ear_r": "#dddddd",
        "neck_l": "#7f5539",
        "neck": "#7f5539",
        "cloth": "#6c757d",
    }
    fallback_palette = [
        "#3f6db5",
        "#d767d7",
        "#bfc75f",
        "#54c49b",
        "#d9bf50",
        "#8ecae6",
        "#7f5539",
        "#6c757d",
    ]
    unique_labels = np.unique(sampled_labels)
    fig = go.Figure()

    for idx, label in enumerate(unique_labels):
        mask = sampled_labels == label
        label_name = label_names.get(int(label), f"Class {int(label)}") if label_names else f"Class {int(label)}"
        marker_color = semantic_colors.get(label_name, fallback_palette[idx % len(fallback_palette)])
        fig.add_trace(
            go.Scatter3d(
                x=sampled_shape[mask, 0],
                y=sampled_shape[mask, 1],
                z=sampled_shape[mask, 2],
                mode="markers",
                name=label_name,
                text=np.asarray(hover_names)[mask],
                customdata=np.column_stack(
                    [
                        sampled_shape[mask, 0],
                        sampled_shape[mask, 1],
                        sampled_shape[mask, 2],
                    ]
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "x=%{customdata[0]:.4f}<br>"
                    "y=%{customdata[1]:.4f}<br>"
                    "z=%{customdata[2]:.4f}"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=1.4,
                    color=marker_color,
                    opacity=1.0,
                    line=dict(width=0.0),
                ),
                hoverlabel=dict(
                    bgcolor=marker_color,
                    bordercolor=marker_color,
                    font=dict(color="white", size=13),
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Interactive 3D Face Reconstruction",
        scene_aspectmode="data",
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(245,245,245)", gridcolor="rgb(210,210,210)"),
            yaxis=dict(backgroundcolor="rgb(245,245,245)", gridcolor="rgb(210,210,210)"),
            zaxis=dict(backgroundcolor="rgb(245,245,245)", gridcolor="rgb(210,210,210)"),
        ),
        showlegend=False,
    )
    return fig


def open_mesh_in_browser(results, vertex_labels=None, label_names=None, show_every=1):
    fig = create_browser_mesh_figure(
        results=results,
        vertex_labels=vertex_labels,
        label_names=label_names,
        show_every=show_every,
    )
    if fig is None:
        raise RuntimeError("Plotly is not installed in the active environment.")

    html = fig.to_html(include_plotlyjs=True, full_html=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(html)
        temp_path = temp_file.name

    webbrowser.open(f"file://{temp_path}")
    return temp_path


def _save_figure(fig, save_path):
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def _embed_figure(parent, figure, with_toolbar=False):
    container = ttk.Frame(parent)
    container.pack(fill="both", expand=True)

    canvas = FigureCanvasTkAgg(figure, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    if with_toolbar:
        toolbar = NavigationToolbar2Tk(canvas, container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    return canvas


def launch_results_viewer(
    image,
    labels,
    results,
    points,
    vertex_labels=None,
    label_names=None,
    save_path="results/pipeline_panels.png",
):
    results_figure = create_results_figure(image, labels, results, points)
    output_path = _save_figure(results_figure, save_path)

    root = tk.Tk()
    root.title("AnthroFace3D Viewer")
    root.geometry("1440x960")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    overview_tab = ttk.Frame(notebook)
    mesh_tab = ttk.Frame(notebook)
    notebook.add(overview_tab, text="2D Results")
    notebook.add(mesh_tab, text="3D Viewer")

    _embed_figure(overview_tab, results_figure, with_toolbar=True)

    mesh_content = ttk.Frame(mesh_tab, padding=24)
    mesh_content.pack(fill="both", expand=True)

    title = ttk.Label(mesh_content, text="Interactive 3D Viewer", font=("TkDefaultFont", 14, "bold"))
    title.pack(anchor="w", pady=(0, 12))

    description = ttk.Label(
        mesh_content,
        text=(
            "Open the 3D reconstruction in your browser for smooth interaction, "
            "hover labels, zoom, pan, and rotation."
        ),
        wraplength=700,
        justify="left",
    )
    description.pack(anchor="w", pady=(0, 18))

    status_var = tk.StringVar(value="The browser viewer has not been opened yet.")
    status = ttk.Label(mesh_content, textvariable=status_var, wraplength=900, justify="left")
    status.pack(anchor="w", pady=(12, 0))

    def _open_browser():
        try:
            opened_path = open_mesh_in_browser(
                results=results,
                vertex_labels=vertex_labels,
                label_names=label_names,
                show_every=1,
            )
            status_var.set(f"Opened interactive viewer in browser using temporary file: {opened_path}")
        except Exception as exc:
            status_var.set(f"Could not open browser viewer: {exc}")
            messagebox.showerror("3D Viewer", str(exc), parent=root)

    open_button = ttk.Button(mesh_content, text="Open 3D Viewer in Browser", command=_open_browser)
    open_button.pack(anchor="w")

    info = ttk.Label(
        mesh_content,
        text="Tip: keep the browser open while you inspect the mesh. You can reopen it anytime from this button.",
        wraplength=900,
        justify="left",
    )
    info.pack(anchor="w", pady=(18, 0))

    root.mainloop()
    return output_path
