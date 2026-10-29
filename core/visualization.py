import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

import numpy as np

from core.segmentation import visualize_segmentation
from core.palette import INK, PART_COLORS, CONTEXT_COLOR, GENERIC_ACCENT, LANDMARK_ACCENT
from core.measurements import LANDMARK_NAMES, landmark_descriptors
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


def _hex_to_rgb(hex_color):
    hex_digits = hex_color.lstrip("#")
    return np.array([int(hex_digits[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _mat_render(render, threshold=10):
    """Replace the renderer's black backdrop with the panel background, so the
    rendered face sits on the page like the photographic panels do."""
    matted = render.copy()
    backdrop = (matted <= threshold).all(axis=-1)
    matted[backdrop] = _hex_to_rgb(INK["surface"])
    return matted


def _plot_sampled_points(ax, image, points, max_points=2000):
    ax.imshow(image)
    if points is not None and len(points) > 0:
        stride = max(1, len(points) // max_points)
        sampled = points[::stride]
        ax.scatter(sampled[:, 0], sampled[:, 1], s=2.4, c=GENERIC_ACCENT, alpha=0.4, linewidths=0)
    ax.axis("off")


def _style_panel_title(ax, text):
    ax.set_title(text, fontsize=11, color=INK["secondary"], pad=6)


def create_results_figure(image, labels, results, points, ldm68_idx=None):
    render_face = _mat_render(_to_uint8_image(results["render_face"][0]))
    render_shape = _mat_render(_to_uint8_image(results["render_shape"][0]))
    seg_overlay = visualize_segmentation(image, labels)

    fig = Figure(figsize=(15, 10.2), dpi=100, facecolor=INK["surface"], layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.01, hspace=0.02)
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

    if ldm68_idx is not None:
        axes[5].imshow(image)
        # results["ldm68"] lives in the model's 224 frame, while `points` holds
        # the same vertices already mapped onto the crop, so the landmarks are
        # the same indices taken from there.
        landmarks = points[ldm68_idx]
        axes[5].scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            s=14,
            c=LANDMARK_ACCENT,
            linewidths=0,
        )
        _style_panel_title(axes[5], "68 Landmarks")
        axes[5].axis("off")
    else:
        axes[5].imshow(image)
        _style_panel_title(axes[5], "Input Preview")
        axes[5].axis("off")

    fig.suptitle("AnthroFace3D results", fontsize=17, color=INK["primary"], fontweight="semibold")
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
        # The toolbar is a plain Tk widget, so it keeps its own grey unless the
        # surface colour is pushed onto it and its children.
        toolbar.configure(background=INK["surface"])
        for child in toolbar.winfo_children():
            try:
                child.configure(background=INK["surface"])
            except tk.TclError:
                pass
        toolbar.pack(side="bottom", fill="x")

    return canvas


# One family across the window and the figures, and four sizes instead of a
# size picked per widget.
TYPE_SCALE = {"display": 16, "value": 14, "title": 12, "body": 10}


def _ui_family(root):
    """DejaVu Sans matches the matplotlib panels; fall back to whatever Tk uses."""
    from tkinter import font as tkfont

    available = set(tkfont.families(root))
    return "DejaVu Sans" if "DejaVu Sans" in available else tkfont.nametofont("TkDefaultFont").cget("family")


def _configure_style(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    family = _ui_family(root)
    body = (family, TYPE_SCALE["body"])

    root.configure(background=INK["surface"])
    style.configure("TFrame", background=INK["surface"])
    style.configure("TLabel", background=INK["surface"], foreground=INK["primary"], font=body)
    style.configure("Secondary.TLabel", background=INK["surface"], foreground=INK["secondary"], font=body)
    style.configure("Muted.TLabel", background=INK["surface"], foreground=INK["muted"], font=body)
    style.configure(
        "Display.TLabel",
        background=INK["surface"],
        foreground=INK["primary"],
        font=(family, TYPE_SCALE["display"]),
    )
    style.configure(
        "Value.TLabel",
        background=INK["surface"],
        foreground=INK["primary"],
        font=(family, TYPE_SCALE["value"]),
    )
    style.configure(
        "Title.TLabel",
        background=INK["surface"],
        foreground=INK["primary"],
        font=(family, TYPE_SCALE["title"], "bold"),
    )

    style.configure("TNotebook", background=INK["surface"], borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure(
        "TNotebook.Tab",
        padding=(20, 10),
        background=INK["surface"],
        foreground=INK["muted"],
        borderwidth=0,
        font=body,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", INK["surface"])],
        foreground=[("selected", INK["primary"])],
        expand=[("selected", (0, 0, 0, 2))],
    )

    style.configure(
        "TButton",
        padding=(14, 7),
        background=INK["surface"],
        foreground=INK["primary"],
        bordercolor=INK["gridline"],
        focuscolor=INK["surface"],
        borderwidth=1,
        font=body,
    )
    style.map(
        "TButton",
        background=[("active", INK["gridline"])],
        bordercolor=[("active", INK["secondary"])],
    )
    style.configure("TSeparator", background=INK["gridline"])
    return style


def create_landmarks_figure(image, landmarks, roles):
    """The 68 landmarks over the cropped image, with the ones that feed a
    reported descriptor picked out from the ones that feed none."""
    fig = Figure(figsize=(7.6, 7.6), dpi=100, facecolor=INK["surface"], layout="constrained")
    ax = fig.subplots()
    ax.imshow(image)

    feeds = np.array([index in roles for index in range(len(landmarks))])
    ax.scatter(landmarks[~feeds, 0], landmarks[~feeds, 1],
               s=26, c=CONTEXT_COLOR, alpha=0.8, linewidths=0)
    ax.scatter(landmarks[feeds, 0], landmarks[feeds, 1],
               s=40, c=LANDMARK_ACCENT, linewidths=0)
    highlight = ax.scatter([], [], s=190, facecolors="none",
                           edgecolors=INK["primary"], linewidths=1.4)
    ax.axis("off")
    return fig, highlight


def _build_landmarks_tab(parent, image, points, ldm68_idx):
    content = ttk.Frame(parent, padding=24)
    content.pack(fill="both", expand=True)

    roles = landmark_descriptors()
    landmarks = points[ldm68_idx]

    heading = ttk.Label(content, text="Anthropometric points", style="Display.TLabel")
    heading.pack(anchor="w", pady=(0, 4))

    summary = ttk.Label(
        content,
        text=(
            f"{len(roles)} of the {len(landmarks)} landmarks carry the reported descriptors. "
            "The rest come with the 68-point convention and feed none of them."
        ),
        style="Secondary.TLabel",
        wraplength=900,
        justify="left",
    )
    summary.pack(anchor="w", pady=(0, 14))

    figure, highlight = create_landmarks_figure(image, landmarks, roles)
    _embed_figure(content, figure)

    name_var = tk.StringVar(value="Point at a landmark to read it.")
    feeds_var = tk.StringVar(value="")
    ttk.Label(content, textvariable=name_var, style="Value.TLabel").pack(anchor="w", pady=(14, 0))
    ttk.Label(content, textvariable=feeds_var, style="Secondary.TLabel",
              wraplength=900, justify="left").pack(anchor="w", pady=(2, 0))

    # Within this many image pixels a landmark counts as the one being pointed at.
    reach = max(np.ptp(landmarks[:, 0]), np.ptp(landmarks[:, 1])) * 0.04

    def _on_move(event):
        # Only the landmarks that carry a descriptor answer to the pointer; the
        # rest are drawn for context and stay inert.
        nearest = None
        if event.inaxes is not None and event.xdata is not None:
            distances = np.hypot(landmarks[:, 0] - event.xdata, landmarks[:, 1] - event.ydata)
            candidate = int(distances.argmin())
            if distances[candidate] <= reach and candidate in roles:
                nearest = candidate

        if nearest is None:
            name_var.set("Point at a landmark to read it.")
            feeds_var.set("")
            highlight.set_offsets(np.empty((0, 2)))
        else:
            name_var.set(f"{nearest} \u00b7 {LANDMARK_NAMES[nearest]}")
            feeds_var.set("Feeds " + ", ".join(roles[nearest]))
            highlight.set_offsets(landmarks[nearest:nearest + 1])
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", _on_move)
    return content


def _build_reported_grid(parent, measures, columns=3):
    """The reported descriptors as a grid, so the values line up and read as
    the result rather than as another line of text."""
    grid = ttk.Frame(parent)
    grid.pack(fill="x", anchor="w", pady=(0, 18))
    for column in range(columns):
        grid.columnconfigure(column, weight=1, uniform="reported")

    for position, (name, value) in enumerate(measures.items()):
        cell = ttk.Frame(grid, padding=(0, 7, 20, 7))
        cell.grid(row=position // columns, column=position % columns, sticky="w")
        ttk.Label(cell, text=name, style="Secondary.TLabel").pack(anchor="w")
        ttk.Label(cell, text=value, style="Value.TLabel").pack(anchor="w", pady=(2, 0))
    return grid


def _build_part_row(parent, part, root):
    row = ttk.Frame(parent, padding=(0, 12))
    row.pack(fill="x", anchor="w")
    row.columnconfigure(1, weight=1)

    swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0, background=INK["surface"])
    swatch.create_oval(2, 2, 14, 14, fill=part.color, outline="")
    swatch.grid(row=0, column=0, rowspan=2, padx=(0, 12), sticky="n")

    name_label = ttk.Label(row, text=part.display_name, style="Title.TLabel")
    name_label.grid(row=0, column=1, sticky="w")

    summary_text = "    ".join(f"{key}: {value}" for key, value in part.measurements.items())
    summary = ttk.Label(row, text=summary_text, style="Secondary.TLabel", wraplength=760, justify="left")
    summary.grid(row=1, column=1, sticky="w", pady=(2, 0))

    # part.secondary holds the convex-hull measures of the segmentation mask.
    # They were discarded during development for being unstable, so they stay
    # in the CSV and out of this panel.
    next_row = 2

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
    status_label.grid(row=next_row, column=1, columnspan=2, sticky="w", pady=(4, 0))

    return row


def launch_results_viewer(
    image,
    labels,
    results,
    points,
    ldm68_idx=None,
    vertex_labels=None,
    label_names=None,
    parts=None,
    reported_measures=None,
    save_path="results/pipeline_panels.png",
):
    parts = parts or []
    reported_measures = reported_measures or {}

    results_figure = create_results_figure(image, labels, results, points, ldm68_idx)
    output_path = _save_figure(results_figure, save_path)

    root = tk.Tk()
    root.title("AnthroFace3D Viewer")
    # Fit the window to the screen instead of assuming a large display.
    width = min(1440, root.winfo_screenwidth() - 80)
    height = min(960, root.winfo_screenheight() - 80)
    root.geometry(f"{width}x{height}")
    root.minsize(1024, 680)
    _configure_style(root)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    overview_tab = ttk.Frame(notebook)
    landmarks_tab = ttk.Frame(notebook)
    mesh_tab = ttk.Frame(notebook)
    parts_tab = ttk.Frame(notebook)
    notebook.add(overview_tab, text="2D results")
    notebook.add(landmarks_tab, text="Anthropometric points")
    notebook.add(mesh_tab, text="3D viewer")
    notebook.add(parts_tab, text="Measured parts")

    _embed_figure(overview_tab, results_figure, with_toolbar=True)

    if ldm68_idx is not None:
        _build_landmarks_tab(landmarks_tab, image, points, ldm68_idx)

    mesh_content = ttk.Frame(mesh_tab, padding=24)
    mesh_content.pack(fill="both", expand=True)

    title = ttk.Label(mesh_content, text="Interactive 3D Viewer", style="Display.TLabel")
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

    parts_title = ttk.Label(parts_content, text="Reported descriptors", style="Display.TLabel")
    parts_title.pack(anchor="w", pady=(0, 2))

    if reported_measures:
        reported_caption = ttk.Label(
            parts_content,
            text="Dimensionless and taken from the landmarks alone.",
            style="Secondary.TLabel",
        )
        reported_caption.pack(anchor="w", pady=(0, 12))
        _build_reported_grid(parts_content, reported_measures)

    if parts:
        ttk.Separator(parts_content, orient="horizontal").pack(fill="x", pady=(0, 16))

        parts_subtitle = ttk.Label(parts_content, text="Per-part detail", style="Title.TLabel")
        parts_subtitle.pack(anchor="w", pady=(0, 2))

        parts_description = ttk.Label(
            parts_content,
            text="The landmark distances behind the descriptors above. Each part opens in its own 3D view.",
            style="Secondary.TLabel",
            wraplength=900,
            justify="left",
        )
        parts_description.pack(anchor="w", pady=(0, 6))
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
