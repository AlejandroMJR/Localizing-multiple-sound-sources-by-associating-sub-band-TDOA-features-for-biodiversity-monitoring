import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from association import peak_band_vector

# ============================================================
# File / folder utilities
# ============================================================

def clear_folder(folder_path, pattern="*"):
    """
    Delete files inside `folder_path` (non-recursive).

    Parameters
    ----------
    folder_path : str
        Path to the directory to clear.
    pattern : str
        Glob pattern for which files to delete (default: '*').
        Examples: '*.png', '*.pdf'
    """
    if not os.path.isdir(folder_path):
        return

    files = glob.glob(os.path.join(folder_path, pattern))
    for file in files:
        if not os.path.isfile(file):
            continue
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def ensure_dir(folder_path):
    """Create folder if it does not exist."""
    os.makedirs(folder_path, exist_ok=True)


# ============================================================
# Feature utilities
# ============================================================
def build_cluster_color_map(sources, cmap_name="tab10"):
    """
    Returns cid2color: dict {cluster_id: rgba}
    """
    if not sources:
        return {}

    cmap = plt.get_cmap(cmap_name)
    n = len(sources)
    return {cid: cmap(cid % cmap.N) for cid in range(n)}


def build_peak_cluster_map(sources):
    """
    Build lookup (pair, peak_idx) -> cluster_id for coloring / post-processing.

    Parameters
    ----------
    sources : list of dict
        Output of associate_tdoa_peaks_by_band_cosine_constrained(...).
        Each dict has "members", and each member contains:
          - "pair": (i,j)
          - "peak_idx": int

    Returns
    -------
    peak2cid : dict
        Maps ((i,j), peak_idx) -> cluster_id
    """
    peak2cid = {}
    if not sources:
        return peak2cid

    for cid, src in enumerate(sources):
        for m in src.get("members", []):
            peak2cid[(tuple(m["pair"]), int(m["peak_idx"]))] = cid
    return peak2cid

def theoretical_tdoas(src_positions, mic_positions, c=343.0, fs=None):
    src_positions = np.asarray(src_positions)
    mic_positions = np.asarray(mic_positions)
    src_positions = src_positions.reshape(-1, src_positions.shape[-1])
    mic_positions = mic_positions.reshape(-1, mic_positions.shape[-1])

    dists = np.linalg.norm(src_positions[:, None, :] - mic_positions[None, :, :], axis=-1)
    tof = dists / c
    tdoa_seconds = tof[:, :, None] - tof[:, None, :]

    if fs is not None:
        return tdoa_seconds, tdoa_seconds * fs
    return tdoa_seconds, None


# ============================================================
# Geometry plots
# ============================================================

def set_plot_style(base_font=24):
    """Set consistent matplotlib style for plots."""
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 2,
        "legend.fontsize": base_font - 2,
        "axes.linewidth": 1.4,
    })


def savefig(fig, path_no_ext, *, save_pdf=True, save_png=False, dpi=300, close=True):
    """
    Save a matplotlib figure with consistent settings.
    """
    import os
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)

    if save_pdf:
        fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", pad_inches=0.02)
    if save_png:
        fig.savefig(path_no_ext + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    if close:
        plt.close(fig)

def plot_room(micPos, srcPos, roomDims, ax=None,
              mic_label="Microphones", src_label="Sources",
              mic_color="green", src_color="black",
              mic_marker="o", src_marker="x",
              mic_size=100, src_size=100,
              show_legend=True,
              **scatter_kwargs):
    """
    Plot microphones and sources in a 2D room.


    Parameters
    ----------
    micPos : (M,2) array
        Microphone positions.
    srcPos : (S,2) array
        Source positions (can be empty with shape (0,2)).
    roomDims : float or (W,H)
        Room size. If float -> square [0,roomDims]x[0,roomDims].
    ax : matplotlib Axes or None
        If None, creates a new figure/axes.
    scatter_kwargs : dict
        Extra kwargs passed to ax.scatter.
    """
    micPos = np.asarray(micPos, dtype=float)
    srcPos = np.asarray(srcPos, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Room bounds
    if np.isscalar(roomDims):
        xlim = (0.0, float(roomDims))
        ylim = (0.0, float(roomDims))
    else:
        xlim = (0.0, float(roomDims[0]))
        ylim = (0.0, float(roomDims[1]))

    # Microphones
    ax.scatter(
        micPos[:, 0], micPos[:, 1],
        s=mic_size, marker=mic_marker, c=mic_color,
        linewidths=1.5,
        label=mic_label,
        **scatter_kwargs
    )

    # Sources (optional)
    if srcPos.size > 0:
        ax.scatter(
            srcPos[:, 0], srcPos[:, 1],
            s=src_size, marker=src_marker, c=src_color,
            linewidths=1.5,
            label=src_label,
            **scatter_kwargs
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

    return ax


def plot_tdoa_hyperbola(m0, m1, tdoa,
                        fs=None, c=343.0,
                        xlim=(0, 1), ylim=(0, 1),
                        n_points=400, ax=None,
                        color=None, linewidths=1.0, alpha=1.0,
                        **contour_kwargs):
    """
    Plot the 2D TDOA hyperbola for a microphone pair as a zero-level contour.

    Hyperbola definition:
        ||p - m0|| - ||p - m1|| = c * tau

    Parameters
    ----------
    m0, m1 : (2,) array-like
        Microphone positions [x,y] in meters.
    tdoa : float
        If fs is None -> seconds.
        If fs is given -> samples (converted to seconds).
    fs : float or None
        Sampling frequency.
    c : float
        Speed of sound (m/s).
    xlim, ylim : (min,max)
        Plot bounds in meters.
    n_points : int
        Grid resolution used for contouring.
    ax : matplotlib Axes or None
        Axis to plot on.
    color : any matplotlib color or None
        Can be a string ("C0") OR an RGBA array from a colormap.
    linewidths, alpha : float
        Contour style.
    contour_kwargs : dict
        Passed to ax.contour.
    """
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)

    # Convert to seconds if needed
    tau = (float(tdoa) / float(fs)) if fs is not None else float(tdoa)
    delta_d = c * tau  # distance difference in meters

    xs = np.linspace(xlim[0], xlim[1], n_points)
    ys = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], axis=-1)

    d0 = np.linalg.norm(P - m0, axis=-1)
    d1 = np.linalg.norm(P - m1, axis=-1)
    F = d0 - d1 - delta_d

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    default_kwargs = dict(linewidths=linewidths, alpha=alpha)
    if color is not None:
        default_kwargs["colors"] = [color]
    default_kwargs.update(contour_kwargs)

    ax.contour(X, Y, F, levels=[0.0], **default_kwargs)
    return ax


# ============================================================
# Band-support histogram plots
# ============================================================

def plot_tdoa_peak_band_histograms_three_pairs(
    tdoas_hat_pairs,
    pair_left,
    pair_mid,
    pair_right,
    n_bands,
    band_freqs=None,
    out_dir="Figures",
    base_font=24,
    tick_step=15,
    sources=None,
    unassigned_color="0.75",
):
    """
    Plot band-support histograms for peaks from three microphone pairs (3 columns).

    Coloring
    --------
    If `sources` (association output) is provided:
      - peaks belonging to the same cluster are assigned the same color across pairs.
      - peaks not assigned to any cluster are plotted in `unassigned_color`.

    Expected structure
    ------------------
    tdoas_hat_pairs[(i,j)] is a dict with:
      - "peaks": list of peaks
    Each peak must contain "band_counts" or data usable by `peak_band_vector`.

    Returns
    -------
    fig, axes
    """
    ensure_dir(out_dir)

    def _get_peaks(pair):
        info = tdoas_hat_pairs.get(pair)
        if info is None:
            raise ValueError(f"Microphone pair {pair} not found in tdoas_hat_pairs.")
        return info.get("peaks", [])

    peaks_L = _get_peaks(pair_left)
    peaks_M = _get_peaks(pair_mid)
    peaks_R = _get_peaks(pair_right)

    n_rows = max(len(peaks_L), len(peaks_M), len(peaks_R))
    if n_rows == 0:
        raise ValueError("No peaks found for the provided microphone pairs.")

    # x-axis
    x = np.arange(n_bands) if band_freqs is None else np.asarray(band_freqs)
    x_label = "Frequency zone index" if band_freqs is None else "Frequency (Hz)"
    bar_width = 1 if band_freqs is None else (x[1] - x[0] if len(x) > 1 else 1.0)

    # Local rcParams
    with plt.rc_context({
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 2,
        "legend.fontsize": base_font - 2,
        "axes.linewidth": 1.4,
    }):
        fig_w = 20
        fig_h = max(9, 3.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows, 3,
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = np.array([axes])

        # Cluster coloring
        peak2cid = build_peak_cluster_map(sources) if sources is not None else {}
        peak2cid = build_peak_cluster_map(sources) if sources is not None else {}
        cid2color = build_cluster_color_map(sources, cmap_name=("tab10" if len(sources or []) <= 10 else "tab20"))

        def color_for(pair, peak_idx):
            cid = peak2cid.get((tuple(pair), int(peak_idx)))
            if cid is None:
                return unassigned_color
            return cid2color[cid]

        def annotate_peak(ax, k):
            ax.text(
                0.72, 0.92, f"Peak {k}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=base_font,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
            )

        pairs = [pair_left, pair_mid, pair_right]
        peaks_all = [peaks_L, peaks_M, peaks_R]

        for k in range(n_rows):
            for col in range(3):
                ax = axes[k, col]
                pair = pairs[col]
                peaks = peaks_all[col]

                if k >= len(peaks):
                    ax.axis("off")
                    continue

                vec = peak_band_vector(peaks[k], n_bands)
                ax.bar(x, vec, width=bar_width, color=color_for(pair, k), linewidth=0)
                annotate_peak(ax, k)
                ax.grid(alpha=0.25, linewidth=1.0)

        # Column titles
        axes[0, 0].set_title(f"Pair {pair_left}")
        axes[0, 1].set_title(f"Pair {pair_mid}")
        axes[0, 2].set_title(f"Pair {pair_right}")

        # x labels only at bottom row
        for c in range(3):
            axes[-1, c].set_xlabel(x_label)

        # Reduce x tick clutter for band indices
        if band_freqs is None:
            ticks = np.arange(0, n_bands, tick_step)
            for c in range(3):
                axes[-1, c].set_xticks(ticks)

    return fig


# ============================================================
# Hyperbola plots
# ============================================================

def plot_sources_hyperbolas(
    sources,
    mic_positions,
    fs,
    env_size,
    ax=None,
    c=343.0,
    alpha=0.8,
    linewidth=1.0,
    show_legend=True,
):
    """
    Plot hyperbolas for each associated source hypothesis (one color per hypothesis).

    Parameters
    ----------
    sources : list of dict
        Association output. Each source has "members".
        Each member must have:
          - "pair": (i,j)
          - "tau": TDOA in samples
    mic_positions : (M,2)
    fs : float
    env_size : float
    """
    mic_positions = np.asarray(mic_positions, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    colors = build_cluster_color_map(sources, cmap_name=("tab10" if len(sources) <= 10 else "tab20"))

    for s_idx, src in enumerate(sources):
        col = colors[s_idx]

        for member in src.get("members", []):
            i, j = member["pair"]
            tau = member["tau"]

            plot_tdoa_hyperbola(
                m0=mic_positions[i],
                m1=mic_positions[j],
                tdoa=tau,
                fs=fs,
                c=c,
                xlim=(0, env_size),
                ylim=(0, env_size),
                ax=ax,
                color=col,
                linewidths=linewidth,
                alpha=alpha,
            )

        ax.plot([], [], color=col, label=rf"$\mathcal{{A}}_{{{s_idx+1}}}$")

    ax.set_title("Hyperbolas grouped by associated source")
    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_aspect("equal")

    if show_legend:
        ax.legend(loc="best")

    return ax


def plot_truth_vs_estimates(micPos, srcPos, estPos, roomDims, ax=None):
    """
    Plot microphones, true sources, and estimated sources.
    """
    micPos = np.asarray(micPos, float)
    srcPos = np.asarray(srcPos, float)
    estPos = np.asarray(estPos, float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    plot_room(micPos, srcPos, roomDims, ax=ax)

    if estPos.size > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(estPos)))
        ax.scatter(
            estPos[:, 0], estPos[:, 1],
            s=120,
            marker="o",
            linewidths=2,
            alpha=0.6,
            color=colors,
            label="Estimated sources",
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    ax.set_title("True and estimated sources")
    return ax


def plot_all_pairs_hyperbolas(
    tdoas_hat_pairs,
    micPos,
    fs,
    env_size,
    srcPos=None,
    ax=None,
    c=343.0,
    alpha=0.35,
    linewidth=1.0,
    color_by="none",
    max_hyperbolas=None,
):
    """
    Plot hyperbolas for all pairs and all estimated TDOA peaks.

    Expected structure
    ------------------
    tdoas_hat_pairs[(i,j)] contains:
      - "tdoas": list/array of TDOA peaks (in samples)

    Parameters
    ----------
    color_by : "pair" or "none"
        If "pair", each mic pair uses a different color.
    max_hyperbolas : int or None
        If set, stops plotting after this many hyperbolas.
    """
    micPos = np.asarray(micPos, float)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Plot microphones and (optionally) true sources
    if srcPos is None:
        plot_room(micPos, np.zeros((0, 2)), env_size, ax=ax)
    else:
        plot_room(micPos, srcPos, env_size, ax=ax)

    pair_keys = list(tdoas_hat_pairs.keys())
    if color_by == "pair":
        cols = plt.cm.tab10(np.linspace(0, 1, max(len(pair_keys), 1)))
    else:
        cols = [None] * len(pair_keys)

    count = 0
    for idx, (pair, info) in enumerate(tdoas_hat_pairs.items()):
        i, j = pair
        taus = np.asarray(
            info.get("peak_taus", info.get("tdoas", [])),
            dtype=float
        )
        if taus.size == 0:
            continue

        for tau in taus:
            plot_tdoa_hyperbola(
                m0=micPos[i],
                m1=micPos[j],
                tdoa=tau,
                fs=fs,
                c=c,
                xlim=(0, env_size),
                ylim=(0, env_size),
                ax=ax,
                color=(cols[idx] if color_by == "pair" else None),
                linewidths=linewidth,
                alpha=alpha,
            )
            count += 1
            if max_hyperbolas is not None and count >= int(max_hyperbolas):
                ax.set_title(f"Hyperbolas (stopped at {max_hyperbolas})")
                return ax

    ax.set_title("All hyperbolas from all pairs (all estimated TDOA peaks)")
    return ax
