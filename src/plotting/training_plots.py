import numpy as np
import matplotlib.pyplot as plt


def plot_train_val_distribution(
    g_train,
    g_val,
    *,
    target_idx=None,
    bins=100,
    density=True,
    alpha=0.5,
    logy=False,
    clip_quantiles=(0.01, 0.99),
):
    """
    Plot train vs validation target distributions.

    - target_idx = int  → single target (old behavior)
    - target_idx = None → plot all targets in separate subplots

    Returns
    -------
    fig, axes
    """

    def extract_targets(graphs):
        ys = []
        for g in graphs:
            y = g.y.detach().cpu().numpy()
            if y.ndim == 1:
                y = y[:, None]
            ys.append(y)
        return np.concatenate(ys, axis=0)  # (N, D)

    Y_train = extract_targets(g_train)
    Y_val   = extract_targets(g_val)

    # Remove NaNs/infs globally
    mask_train = np.all(np.isfinite(Y_train), axis=1)
    mask_val   = np.all(np.isfinite(Y_val), axis=1)
    Y_train = Y_train[mask_train]
    Y_val   = Y_val[mask_val]

    D = Y_train.shape[1]

    # --- Select targets ---
    if target_idx is None:
        target_indices = list(range(D))
    else:
        target_indices = [int(target_idx)]

    n_targets = len(target_indices)

    # --- Figure layout ---
    fig, axes = plt.subplots(
        1, n_targets,
        figsize=(6 * n_targets, 4),
        squeeze=False,
    )
    axes = axes.ravel()

    # --- Loop over targets ---
    for ax, ti in zip(axes, target_indices):

        y_train = Y_train[:, ti]
        y_val   = Y_val[:, ti]

        # --- Clipping ---
        y_all = np.concatenate([y_train, y_val])
        q_low, q_high = np.quantile(y_all, clip_quantiles)

        train_out_low  = np.sum(y_train < q_low)
        train_out_high = np.sum(y_train > q_high)
        val_out_low    = np.sum(y_val < q_low)
        val_out_high   = np.sum(y_val > q_high)

        y_train_clip = np.clip(y_train, q_low, q_high)
        y_val_clip   = np.clip(y_val, q_low, q_high)

        # --- Plot ---
        ax.hist(
            y_train_clip,
            bins=bins,
            range=(q_low, q_high),
            density=density,
            alpha=alpha,
            label="train",
        )

        ax.hist(
            y_val_clip,
            bins=bins,
            range=(q_low, q_high),
            density=density,
            alpha=alpha,
            label="val",
        )

        ax.set_xlabel(f"target {ti} (clipped)")
        ax.set_ylabel("density" if density else "count")
        ax.set_title(f"Target {ti}")

        if logy:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.2)
        ax.legend()

        # --- Print stats per target ---
        print(f"\n=== Target {ti} ===")
        print(f"Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        print(f"Val  : mean={y_val.mean():.4f}, std={y_val.std():.4f}")
        print(f"Clip range: [{q_low:.4f}, {q_high:.4f}]")

        print("Train outliers:")
        print(f"  below: {train_out_low} ({train_out_low/len(y_train):.2%})")
        print(f"  above: {train_out_high} ({train_out_high/len(y_train):.2%})")

        print("Val outliers:")
        print(f"  below: {val_out_low} ({val_out_low/len(y_val):.2%})")
        print(f"  above: {val_out_high} ({val_out_high/len(y_val):.2%})")

    fig.tight_layout()
    return fig, axes



from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm


def add_cov_ellipse(ax, x, y, n_std=1.0, **kwargs):
    """Draw covariance ellipse for points (x, y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        return

    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(vals)

    ell = Ellipse(
        xy=(x.mean(), y.mean()),
        width=width,
        height=height,
        angle=theta,
        fill=False,
        **kwargs,
    )
    ax.add_patch(ell)


def _as_2d_targets(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    if a.ndim == 2:
        return a
    raise ValueError(f"Expected array with shape (N,) or (N,D), got {a.shape}")


def plot_markerwise_true_vs_pred_density(
    y,
    mu,
    X,
    marker_idx_show,
    marker_names_sel,
    *,
    n_pos=None,
    target_idx=None,
    target_labels=None,
    marker_names=None,
    cols=4,
    nbins=50,
):
    """
    Plot markerwise true-vs-predicted density maps.

    Parameters
    ----------
    y, mu : array-like
        Shape (N,) or (N, D).
    X : array-like
        Marker matrix, shape (N, M).
    marker_idx_show : array-like
        Marker indices to plot.
    marker_names_sel : list[str]
        Display names matching marker_idx_show.
    n_pos : array-like or None
        Optional positive counts per marker.
    target_idx : int or None
        If int, plot only that target.
        If None, plot all targets and return one figure per target.
    target_labels : list[str] or dict[int, str] or None
        Labels for target dimensions.
    marker_names : list[str] or None
        Optional full marker-name list, used only if marker_names_sel is None.
    cols : int
        Number of subplot columns.
    nbins : int
        Number of 2D histogram bins.

    Returns
    -------
    figs : list[matplotlib.figure.Figure]
        One figure per plotted target.
    axes_list : list[np.ndarray]
        One flattened axes array per figure.
    """

    Y = _as_2d_targets(y)
    MU = _as_2d_targets(mu)
    X = np.asarray(X)

    if Y.shape != MU.shape:
        raise ValueError(f"y and mu must have matching shapes, got {Y.shape} and {MU.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and y must have same first dimension, got {X.shape[0]} and {Y.shape[0]}")

    D = Y.shape[1]

    if target_idx is None:
        target_indices = list(range(D))
    else:
        target_indices = [int(target_idx)]

    marker_idx_show = list(marker_idx_show)

    if marker_names_sel is None:
        if marker_names is None:
            marker_names_sel = [f"marker {m}" for m in marker_idx_show]
        else:
            marker_names_sel = [
                marker_names[m] if m < len(marker_names) else f"marker {m}"
                for m in marker_idx_show
            ]

    def target_label(ti):
        if target_labels is None:
            return f"target {ti}" if D > 1 else "target"
        if isinstance(target_labels, dict):
            return target_labels.get(ti, f"target {ti}")
        return target_labels[ti] if ti < len(target_labels) else f"target {ti}"

    figs = []
    axes_list = []

    for ti in target_indices:
        y_t = Y[:, ti]
        mu_t = MU[:, ti]
        label_t = target_label(ti)

        rows = int(np.ceil(len(marker_idx_show) / cols)) if len(marker_idx_show) else 1
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 4.6, rows * 4.2),
            squeeze=False,
        )
        axes = axes.ravel()

        all_true = []
        all_pred = []

        for m in marker_idx_show:
            mask = X[:, m] > 0.5
            yt = np.asarray(y_t[mask], dtype=float)
            yp = np.asarray(mu_t[mask], dtype=float)

            good = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[good], yp[good]

            if yt.size > 0:
                all_true.append(yt)
                all_pred.append(yp)

        if len(all_true) == 0:
            raise RuntimeError(f"No valid points found for selected markers for {label_t}.")

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        global_lo = min(all_true.min(), all_pred.min())
        global_hi = max(all_true.max(), all_pred.max())
        pad = 0.05 * (global_hi - global_lo + 1e-12)
        global_lo -= pad
        global_hi += pad

        hist_range = [[global_lo, global_hi], [global_lo, global_hi]]

        for k, m in enumerate(marker_idx_show):
            ax = axes[k]

            mask = X[:, m] > 0.5
            yt = np.asarray(y_t[mask], dtype=float)
            yp = np.asarray(mu_t[mask], dtype=float)

            good = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[good], yp[good]

            if yt.size == 0:
                ax.axis("off")
                continue

            n = yt.size
            mean_t, mean_p = yt.mean(), yp.mean()
            std_t, std_p = yt.std(ddof=0), yp.std(ddof=0)
            rmse = np.sqrt(np.mean((yp - yt) ** 2))
            bias = np.mean(yp - yt)
            r = np.corrcoef(yt, yp)[0, 1] if n > 1 else np.nan

            ax.hist2d(
                yt,
                yp,
                bins=nbins,
                range=hist_range,
                norm=LogNorm(),
                cmin=1,
            )

            ax.plot(
                [global_lo, global_hi],
                [global_lo, global_hi],
                linestyle="--",
                linewidth=1,
            )

            ax.scatter([mean_t], [mean_p], s=60, marker="x", linewidths=2)

            ax.plot([mean_t - std_t, mean_t + std_t], [mean_p, mean_p], linewidth=2)
            ax.plot([mean_t, mean_t], [mean_p - std_p, mean_p + std_p], linewidth=2)

            add_cov_ellipse(ax, yt, yp, n_std=1.0, linewidth=2)
            add_cov_ellipse(ax, yt, yp, n_std=2.0, linewidth=1, alpha=0.7)

            ax.set_xlim(global_lo, global_hi)
            ax.set_ylim(global_lo, global_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)

            if n_pos is None:
                n_pos_label = n
            else:
                n_pos_label = n_pos[m]

            ax.set_title(
                f"{marker_names_sel[k]} (n⁺={n_pos_label})\n"
                f"r={r:.2f}  RMSE={rmse:.3f}  bias={bias:.3f}",
                fontsize=9,
            )
            ax.set_xlabel(f"true {label_t}")
            ax.set_ylabel(f"pred {label_t}")

        for k in range(len(marker_idx_show), len(axes)):
            axes[k].axis("off")

        fig.suptitle(
            f"Validation predictions per marker: true vs predicted — {label_t}\n"
            "Density map with identity line, mean±std cross, and covariance ellipses",
            y=1.02,
            fontsize=14,
        )

        fig.tight_layout()

        figs.append(fig)
        axes_list.append(axes)

    return figs, axes_list