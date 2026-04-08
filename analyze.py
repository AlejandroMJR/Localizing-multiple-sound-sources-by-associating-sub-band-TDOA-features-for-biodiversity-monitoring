import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
# import matplot2tikz
# -----------------------------
def _make_noise_color_map(df, cmap_name="tab10"):
    noises = sorted(df["noisePower"].unique())
    cmap = plt.get_cmap(cmap_name)
    return noises, {noise: cmap(i * 3) for i, noise in enumerate(noises)}

def _make_linestyle_map(values, linestyles=("-", "--", ":", "-.")):
    return {v: linestyles[i % len(linestyles)] for i, v in enumerate(values)}

def _add_dual_legend(ax, noises, noise_color, style_values, style_map,
                     noise_title="", style_title="", noiseLegend=False, styleLegend=False):
    if noiseLegend:
        noise_handles = [
            Line2D([0], [0], color=noise_color[n], lw=2, marker="o",
                   label=fr"$L_{{\mathrm{{noise}}}} = {n}\,\mathrm{{dB\,SPL}}$")
            for n in noises
        ]
        leg1 = ax.legend(handles=noise_handles, title=noise_title,
                         loc="upper left")
        ax.add_artist(leg1)

    if styleLegend:
        style_handles = [
            Line2D([0], [0], color="black", lw=2, linestyle=style_map[v],
                   label=f"S = {v}")
            for v in style_values
        ]
        ax.legend(handles=style_handles, title="",
                  loc="upper right")

def plot_rmse_vs_nodes_multi_sources(df, fixed_sources=(3, 4, 5)):
    fig, ax = plt.subplots(figsize=(12, 8))

    noises, noise_color = _make_noise_color_map(df)
    src_ls = _make_linestyle_map(fixed_sources)

    for noise in noises:
        for ns in fixed_sources:
            sub = df[(df["noisePower"] == noise) & (df["nSources"] == ns)]
            if sub.empty:
                continue

            grp = (sub.groupby("nMics")["rmse_mean"]
                     .mean().reset_index().sort_values("nMics"))

            ax.plot(grp["nMics"], grp["rmse_mean"],
                    color=noise_color[noise], linestyle=src_ls[ns],
                    marker="o", linewidth=2)


    ax.set_xlabel("Number of microphones")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True)

    _add_dual_legend(ax, noises, noise_color, fixed_sources, src_ls, noiseLegend=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("Results/rmse_vs_nMics.pdf",  bbox_inches="tight", pad_inches=0.02)
    # matplot2tikz.save("Results/rmse_vs_nMics.tex",  strict=True, standalone=True)
    plt.show()


def plot_missed_vs_nodes_multi_sources(df, fixed_sources=(3, 4, 5)):
    fig, ax = plt.subplots(figsize=(12, 8))

    noises, noise_color = _make_noise_color_map(df)
    src_ls = _make_linestyle_map(fixed_sources)

    for noise in noises:
        for ns in fixed_sources:
            sub = df[(df["noisePower"] == noise) & (df["nSources"] == ns)]
            if sub.empty:
                continue

            grp = (sub.groupby("nMics")["missed_mean"]
                     .mean().reset_index().sort_values("nMics"))

            ax.plot(grp["nMics"], grp["missed_mean"]/ns*100,
                    color=noise_color[noise], linestyle=src_ls[ns],
                    marker="o", linewidth=2)

    ax.set_xlabel("Number of microphones")
    ax.set_ylabel("Missed detections (%)")
    ax.grid(True)

    _add_dual_legend(ax, noises, noise_color, fixed_sources, src_ls, styleLegend=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("Results/missed_vs_nMics.pdf",  bbox_inches="tight", pad_inches=0.02)
    # matplot2tikz.save("Results/missed_vs_nMics.tex",  strict=True, standalone=True)
    plt.show()


base_font = 24
plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 2,
        "legend.fontsize": base_font - 4,
        "axes.linewidth": 1.4,
    })
noises_to_use = [40, 60]
df = pd.read_csv("Results/monte_carlo_results.csv")
df = df[df["noisePower"].isin(noises_to_use)]
plot_rmse_vs_nodes_multi_sources(df, fixed_sources=(2, 4, 8))
plot_missed_vs_nodes_multi_sources(df, fixed_sources=(2, 4, 8))
