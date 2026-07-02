import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

import numpy as np

from core.segmentation import visualize_segmentation
from core.palette import INK, PART_COLORS, CONTEXT_COLOR, GENERIC_ACCENT
from core.part_viewer import open_part_in_browser


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
        ax.scatter(sampled[:, 0], sampled[:, 1], s=1, c=GENERIC_ACCENT, alpha=0.55)
    ax.axis("off")


def _style_panel_title(ax, text):
    ax.set_title(text, fontsize=12, color=INK["secondary"], fontweight="medium", pad=8)


def create_results_figure(image, labels, results, points):
    render_face = _to_uint8_image(results["render_face"][0])
    render_shape = _to_uint8_image(results["render_shape"][0])
    seg_overlay = visualize_segmentation(image, labels)

    fig = Figure(figsize=(16, 10), dpi=100, facecolor=INK["surface"])
    axes = fig.subplots(2, 3).ravel()

    axes[0].imshow(image)
    _style_panel_title(axes[0], "Cropped Input")
    axes[0].axis("off")

    axes[1].imshow(seg_overlay)
    _style_panel_title(axes[1], "Segmentation Overlay")
    axes[1].axis("off")

    axes[2].imshow(render_face)
    _style_panel_title(axes[2], "Rendered Face")
    axes[2].axis("off")

    axes[3].imshow(render_shape)
    _style_panel_title(axes[3], "Rendered Geometry")
    axes[3].axis("off")

    _plot_sampled_points(axes[4], image, points)
    _style_panel_title(axes[4], "3D Back Projection")

    if "ldm68" in results:
        axes[5].imshow(image)
        landmarks = results["ldm68"][0]
        axes[5].scatter(landmarks[:, 0], 223 - landmarks[:, 1], s=10, c=GENERIC_ACCENT)
        _style_panel_title(axes[5], "68 Landmarks")
        axes[5].axis("off")
    else:
        axes[5].imshow(image)
        _style_panel_title(axes[5], "Input Preview")
        axes[5].axis("off")

    fig.suptitle("AnthroFace3D Results", fontsize=18, color=INK["primary"], fontweight="bold")
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
        hover_names = np.array([f"Class {int(label)}" for label in sampled_labels])
    else:
        hover_names = np.array([label_names.get(int(label), f"Class {int(label)}") for label in sampled_labels])
    sampled_label_names = hover_names

    fig = go.Figure()

    def _add_trace(mask, name, color, marker_size):
        fig.add_trace(
            go.Scatter3d(
                x=sampled_shape[mask, 0],
                y=sampled_shape[mask, 1],
                z=sampled_shape[mask, 2],
                mode="markers",
                name=name,
                text=hover_names[mask],
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
                    size=marker_size,
                    color=color,
                    opacity=1.0,
                    line=dict(width=0.0),
                ),
                hoverlabel=dict(bgcolor=color, bordercolor=color, font=dict(color="white", size=13)),
            )
        )

    # Anatomically relevant parts each get their own fixed categorical color.
    for part_name, color in PART_COLORS.items():
        mask = sampled_label_names == part_name
        if mask.any():
            _add_trace(mask, part_name, color, marker_size=1.4)

    # Everything else (background, hair, hat, clothing...) recedes as context,
    # grouped into a single trace instead of competing for a categorical slot.
    context_mask = ~np.isin(sampled_label_names, list(PART_COLORS.keys()))
    if context_mask.any():
        _add_trace(context_mask, "Other", CONTEXT_COLOR, marker_size=1.0)

    fig.update_layout(
        title=dict(text="Interactive 3D Face Reconstruction", font=dict(color=INK["primary"], size=20)),
        scene_aspectmode="data",
        margin=dict(l=0, r=0, t=56, b=0),
        paper_bgcolor=INK["surface"],
        plot_bgcolor=INK["surface"],
        scene=dict(
            xaxis=dict(backgroundcolor=INK["surface"], gridcolor=INK["gridline"]),
            yaxis=dict(backgroundcolor=INK["surface"], gridcolor=INK["gridline"]),
            zaxis=dict(backgroundcolor=INK["surface"], gridcolor=INK["gridline"]),
        ),
        legend=dict(font=dict(color=INK["secondary"])),
        showlegend=True,
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


def _configure_style(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(background=INK["surface"])
    style.configure("TFrame", background=INK["surface"])
    style.configure("TLabel", background=INK["surface"], foreground=INK["primary"])
    style.configure("Secondary.TLabel", background=INK["surface"], foreground=INK["secondary"])
    style.configure("Muted.TLabel", background=INK["surface"], foreground=INK["muted"])
    style.configure("TNotebook", background=INK["surface"], borderwidth=0)
    style.configure("TNotebook.Tab", padding=(16, 8))
    style.configure("TButton", padding=(12, 6))
    return style


def _build_part_row(parent, part, root):
    row = ttk.Frame(parent, padding=(0, 12))
    row.pack(fill="x", anchor="w")
    row.columnconfigure(1, weight=1)

    swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0, background=INK["surface"])
    swatch.create_oval(2, 2, 14, 14, fill=part.color, outline="")
    swatch.grid(row=0, column=0, rowspan=2, padx=(0, 12), sticky="n")

    name_label = ttk.Label(row, text=part.display_name, font=("TkDefaultFont", 13, "bold"))
    name_label.grid(row=0, column=1, sticky="w")

    summary_text = "    ".join(f"{key}: {value}" for key, value in part.measurements.items())
    summary = ttk.Label(row, text=summary_text, style="Secondary.TLabel", wraplength=760, justify="left")
    summary.grid(row=1, column=1, sticky="w", pady=(2, 0))

    status_var = tk.StringVar(value="")
    status_label = ttk.Label(row, textvariable=status_var, style="Muted.TLabel", wraplength=760, justify="left")

    def _open():
        try:
            opened_path = open_part_in_browser(part)
            status_var.set(f"Opened in browser: {opened_path}")
        except Exception as exc:
            status_var.set(f"Could not open: {exc}")
            messagebox.showerror(part.display_name, str(exc), parent=root)

    button = ttk.Button(row, text=f"Open {part.display_name} in browser", command=_open)
    button.grid(row=0, column=2, rowspan=2, padx=(20, 0), sticky="e")
    status_label.grid(row=2, column=1, columnspan=2, sticky="w", pady=(4, 0))

    return row


def launch_results_viewer(
    image,
    labels,
    results,
    points,
    vertex_labels=None,
    label_names=None,
    parts=None,
    save_path="results/pipeline_panels.png",
):
    parts = parts or []

    results_figure = create_results_figure(image, labels, results, points)
    output_path = _save_figure(results_figure, save_path)

    root = tk.Tk()
    root.title("AnthroFace3D Viewer")
    root.geometry("1440x960")
    _configure_style(root)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    overview_tab = ttk.Frame(notebook)
    mesh_tab = ttk.Frame(notebook)
    parts_tab = ttk.Frame(notebook)
    notebook.add(overview_tab, text="2D Results")
    notebook.add(mesh_tab, text="3D Viewer")
    notebook.add(parts_tab, text="Measured Parts")

    _embed_figure(overview_tab, results_figure, with_toolbar=True)

    mesh_content = ttk.Frame(mesh_tab, padding=24)
    mesh_content.pack(fill="both", expand=True)

    title = ttk.Label(mesh_content, text="Interactive 3D Viewer", font=("TkDefaultFont", 14, "bold"))
    title.pack(anchor="w", pady=(0, 12))

    description = ttk.Label(
        mesh_content,
        text=(
            "Open the full 3D reconstruction in your browser for smooth interaction, "
            "hover labels, zoom, pan, and rotation."
        ),
        style="Secondary.TLabel",
        wraplength=700,
        justify="left",
    )
    description.pack(anchor="w", pady=(0, 18))

    status_var = tk.StringVar(value="The browser viewer has not been opened yet.")
    status = ttk.Label(mesh_content, textvariable=status_var, style="Muted.TLabel", wraplength=900, justify="left")
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

    open_button = ttk.Button(mesh_content, text="Open Full Face in Browser", command=_open_browser)
    open_button.pack(anchor="w")

    info = ttk.Label(
        mesh_content,
        text="Tip: keep the browser open while you inspect the mesh. You can reopen it anytime from this button.",
        style="Muted.TLabel",
        wraplength=900,
        justify="left",
    )
    info.pack(anchor="w", pady=(18, 0))

    parts_content = ttk.Frame(parts_tab, padding=24)
    parts_content.pack(fill="both", expand=True)

    parts_title = ttk.Label(parts_content, text="Measured Parts", font=("TkDefaultFont", 14, "bold"))
    parts_title.pack(anchor="w", pady=(0, 4))

    if parts:
        parts_description = ttk.Label(
            parts_content,
            text="Each measured part opens in its own isolated 3D view: its points, convex hull, and reference landmarks.",
            style="Secondary.TLabel",
            wraplength=900,
            justify="left",
        )
        parts_description.pack(anchor="w", pady=(0, 12))
        for part in parts:
            _build_part_row(parts_content, part, root)
    else:
        empty_label = ttk.Label(
            parts_content,
            text="No measured parts were passed to the viewer.",
            style="Muted.TLabel",
        )
        empty_label.pack(anchor="w", pady=(12, 0))

    root.mainloop()
    return output_path
