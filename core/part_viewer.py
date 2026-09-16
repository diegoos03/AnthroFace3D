import tempfile
import webbrowser

import numpy as np
from scipy.spatial import ConvexHull

from core.palette import INK, LANDMARK_ACCENT


def create_part_figure(part):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=part.points[:, 0], y=part.points[:, 1], z=part.points[:, 2],
        mode="markers",
        name=part.display_name,
        marker=dict(size=3, color=part.color, opacity=1.0, line=dict(width=0.0)),
        hovertemplate=f"<b>{part.display_name}</b><extra></extra>",
    ))

    if len(part.points) >= 4:
        hull = ConvexHull(part.points)
        fig.add_trace(go.Mesh3d(
            x=part.points[:, 0], y=part.points[:, 1], z=part.points[:, 2],
            i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
            color=part.color,
            opacity=0.15,
            name=f"{part.display_name} (convex hull)",
            hoverinfo="skip",
            showlegend=False,
        ))

    if part.landmarks:
        landmark_points = np.array(list(part.landmarks.values()))
        fig.add_trace(go.Scatter3d(
            x=landmark_points[:, 0], y=landmark_points[:, 1], z=landmark_points[:, 2],
            mode="markers",
            name="Reference landmarks",
            text=list(part.landmarks.keys()),
            marker=dict(size=6, color=LANDMARK_ACCENT, symbol="diamond", line=dict(width=1, color="white")),
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"{part.display_name} — isolated view", font=dict(color=INK["primary"], size=20)),
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
    )
    return fig


def open_part_in_browser(part):
    fig = create_part_figure(part)
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(html)
        temp_path = temp_file.name

    webbrowser.open(f"file://{temp_path}")
    return temp_path
