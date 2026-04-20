import numpy as np
import matplotlib.pyplot as plt


"""Reusable matplotlib plots for cluster-level summaries."""



def plot_cluster_marker_heatmaps(marker_means_all, marker_means_center, marker_names, *, cluster_labels=None, figsize=(12, 4), colorbar_label="fraction positive"):
    """Plot side-by-side heatmaps for all nodes and center nodes only."""
    K = marker_means_all.shape[0]
    if cluster_labels is None:
        cluster_labels = [f"C{k}" for k in range(K)]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    im0 = axes[0].imshow(marker_means_all, aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("All nodes")
    axes[0].set_xlabel("marker")
    axes[0].set_ylabel("cluster")
    axes[0].set_xticks(range(len(marker_names)))
    axes[0].set_xticklabels(marker_names, rotation=45, ha="right")
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels(cluster_labels)

    im1 = axes[1].imshow(marker_means_center, aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Center nodes only")
    axes[1].set_xlabel("marker")
    axes[1].set_xticks(range(len(marker_names)))
    axes[1].set_xticklabels(marker_names, rotation=45, ha="right")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label(colorbar_label)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    return fig, axes



def plot_cluster_prediction_boxplots(true_data, pred_data, *, clusters=None, show_outliers=False, figsize=(12, 5)):
    """Plot side-by-side boxplots of true and prediction per cluster."""
    if clusters is None:
        clusters = np.arange(len(true_data))
    offset = 0.18
    pos_true = clusters - offset
    pos_pred = clusters + offset

    fig, ax = plt.subplots(figsize=figsize)
    bp_true = ax.boxplot(true_data, positions=pos_true, widths=0.3, patch_artist=True, showfliers=show_outliers, manage_ticks=False)
    bp_pred = ax.boxplot(pred_data, positions=pos_pred, widths=0.3, patch_artist=True, showfliers=show_outliers, manage_ticks=False)

    for box in bp_true["boxes"]:
        box.set_facecolor("lightblue")
    for box in bp_pred["boxes"]:
        box.set_facecolor("lightgreen")

    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(clusters)
    ax.set_xticklabels([f"C{k}" for k in clusters])
    ax.set_xlabel("cluster")
    ax.set_ylabel("target")
    ax.set_title("True vs prediction  by cluster")
    ax.plot([], [], color="lightblue", linewidth=8, label="true")
    ax.plot([], [], color="lightgreen", linewidth=8, label="predicted")
    ax.legend()
    plt.tight_layout()
    return fig, ax



def plot_cluster_boxplots(df, value_col, cluster_order, *, has_value_mask=None, ylabel=None, title=None, show_outliers=False, facecolor="lightgray", annotate_missing_counts=False, missing_label="count missing"):
    """Plot a per-cluster boxplot, optionally annotating the number of missing nodes."""
    box_data = []
    for k in cluster_order:
        mask = df["cluster"] == k
        if has_value_mask is not None:
            mask = mask & df[has_value_mask]
        box_data.append(df.loc[mask, value_col].values)

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(box_data, showfliers=show_outliers, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor(facecolor)

    ax.set_xticks(np.arange(1, len(cluster_order) + 1))
    ax.set_xticklabels([f"C{k}" for k in cluster_order])
    ax.set_xlabel("cluster")
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(title or f"Distribution of {value_col} by cluster")

    if annotate_missing_counts and has_value_mask is not None:
        whisker_vals = np.array([y for w in bp["whiskers"] for y in w.get_ydata()])
        y_min_whisk = float(np.min(whisker_vals))
        y_max_whisk = float(np.max(whisker_vals))
        yr = y_max_whisk - y_min_whisk if y_max_whisk > y_min_whisk else 1.0
        y_text = y_min_whisk - 0.10 * yr

        valid_mask = np.asarray(df[has_value_mask], dtype=bool)
        labels = np.asarray(df["cluster"], dtype=int)
        for i, k in enumerate(cluster_order, start=1):
            n_missing = int(np.sum((labels == k) & (~valid_mask)))
            ax.text(i, y_text, f"{n_missing}", ha="center", va="center", fontsize=9)

        ax.set_ylim(y_text - 0.05 * yr, y_max_whisk + 0.05 * yr)
        ax.text(len(cluster_order) + 0.4, y_text, missing_label, ha="left", va="center", fontsize=10)

    plt.tight_layout()
    return fig, ax