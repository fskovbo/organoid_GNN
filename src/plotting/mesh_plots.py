import numpy as np
from plotly.subplots import make_subplots

from organograph.plotting.meshes import plot_organoid_mesh


"""Plotly utilities for mesh-projected quantities."""



def plot_projected_true_vs_pred(
    result,
    *,
    colorscale="RdBu_r",
    center_at_zero=True,
    fig_size=(1200, 550),
    view=None,
    title=None,
):
    """Plot ground-truth and predicted vertex values side-by-side on the same mesh."""
    mesh = result["mesh"]
    mesh_true = np.asarray(result["mesh_true"], dtype=float)
    mesh_pred = np.asarray(result["mesh_pred"], dtype=float)

    all_vals = np.concatenate([mesh_true.ravel(), mesh_pred.ravel()])
    finite = np.isfinite(all_vals)
    if not np.any(finite):
        vmin, vmax = -1.0, 1.0
    else:
        vv = all_vals[finite]
        if center_at_zero:
            m = float(np.max(np.abs(vv)))
            m = 1.0 if m == 0.0 else m
            vmin, vmax = -m, m
        else:
            vmin, vmax = float(np.min(vv)), float(np.max(vv))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0

    fig_true = plot_organoid_mesh(
        mesh,
        vertex_values=mesh_true,
        backend="plotly",
        colorscale=colorscale,
        center_at_zero=center_at_zero,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=True,
        fig_size=fig_size,
        view=view,
    )
    fig_pred = plot_organoid_mesh(
        mesh,
        vertex_values=mesh_pred,
        backend="plotly",
        colorscale=colorscale,
        center_at_zero=center_at_zero,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        fig_size=fig_size,
        view=view,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Ground truth", "Prediction"),
        horizontal_spacing=0.03,
    )
    for tr in fig_true.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in fig_pred.data:
        fig.add_trace(tr, row=1, col=2)

    scene1 = fig_true.layout.scene.to_plotly_json() if fig_true.layout.scene else {}
    scene2 = fig_pred.layout.scene.to_plotly_json() if fig_pred.layout.scene else {}
    fig.update_layout(scene=scene1, scene2=scene2, width=int(fig_size[0]), height=int(fig_size[1]))

    organoid_str = result.get("organoid_str")
    if title is None:
        title = organoid_str if organoid_str is not None else "Projected truth vs prediction"
    fig.update_layout(title=title)
    return fig