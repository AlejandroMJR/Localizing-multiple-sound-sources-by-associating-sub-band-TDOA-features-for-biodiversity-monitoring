import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from TDOA_extraction import (
    build_frequency_bands,
    extract_subband_tdoas_gccpairs,
    smooth_tdoa_histogram_gaussian_fast,
    matching_pursuit_tdoa_maxSources,
)
from association import (
    associate_tdoa_peaks_by_band_cosine_constrained,
    attach_band_info_to_peaks_fast,
    attach_timefreq_info_to_peaks
)
from simulate_signals import _simulate
from utils import (
    theoretical_tdoas,
    plot_truth_vs_estimates,
    plot_all_pairs_hyperbolas,
    plot_sources_hyperbolas,
    plot_tdoa_peak_band_histograms_three_pairs,
    savefig,
    set_plot_style,
    clear_folder,
    plot_reference_spectrogram,
    plot_highres_spectrogram_with_projected_clusters,
    build_cluster_color_map,
    build_source_tf_maps,
    project_source_maps_to_visual_grid,
)
from posEstimator import estimate_all_sources_positions
from metrics import compute_position_errors



# =============================================================================
# Main
# =============================================================================

def main(args):
    t0 = time.time()
    np.random.seed(args.seed) # 2 is interesting

    # -----------------------------
    # High-level flags / settings
    # -----------------------------

    make_plots = bool(args.make_plots)
    fig_dir = args.fig_dir
    if args.clean_figures:
        clear_folder(fig_dir)

    if make_plots:
        set_plot_style(base_font=args.base_font)

    # -----------------------------
    # Simulate
    # -----------------------------
    fs, micPos, signals, srcPos = _simulate(
        nMics=args.nMics,
        environmentSize=args.environmentSize,
        nSources=args.nSources,
        noisePower=args.noisePower,
        sourcePower=args.sourcePower,
    )


    # -----------------------------
    # Plot reference spectrogram
    # -----------------------------
    signals = signals[:, :int(fs * args.historyTime)]

    mic_ref = 0
    nperseg_vis = 1024
    noverlap_vis = nperseg_vis // 2
    nfft_vis = nperseg_vis
    if make_plots:
        fig, ax = plot_reference_spectrogram(
            signals=signals,
            fs=fs,
            mic_idx=mic_ref,
            t_start=0.0,
            t_end=args.historyTime,
            nperseg=nperseg_vis,
            noverlap=noverlap_vis,
            nfft=nfft_vis,
            title="Recorder 0 reference spectrogram (1024)",
        )

        savefig(fig, f"{fig_dir}/spectrogram_reference_1024",
                save_pdf=True, save_png=False, close=not args.show_plots)

        if args.show_plots:
            plt.show()

    c = 343.0  # speed of sound [m/s]

    # -----------------------------
    # Build mic pairs (k nearest neighbors per mic)
    # -----------------------------
    k = args.k_neighbors
    M = micPos.shape[0]
    pair_set = set()
    for i in range(M):
        di = np.linalg.norm(micPos - micPos[i], axis=1)
        nn = np.argsort(di)[1:k + 1]  # skip itself
        for j in nn:
            a, b = (i, j) if i < j else (j, i)
            pair_set.add((a, b))

    pair_list = sorted(pair_set)
    iu = np.array([p[0] for p in pair_list], dtype=int)
    ju = np.array([p[1] for p in pair_list], dtype=int)
    n_pairs = iu.size

    pair_dists = np.linalg.norm(micPos[iu] - micPos[ju], axis=1)
    maxMicDist = float(pair_dists.max())
    avgMicDist = float(pair_dists.mean())

    maxTDOA = int((maxMicDist / c) * fs)
    avgTDOA = int((avgMicDist / c) * fs)

    # theoretical TDOAs (simulation/diagnostic)
    _, theoretical_tdoas_samples = theoretical_tdoas(srcPos, micPos, fs=fs)

    # -----------------------------
    # STFT parameters
    # -----------------------------
    frameSize = max(maxTDOA * 2, avgTDOA * 4)
    # frameSize = maxTDOA * 2
    timeOverlap = frameSize // 2
    nfft = frameSize
    hop = frameSize - timeOverlap

    # how many frames correspond to historyTime seconds?
    B = int(np.floor(fs * args.historyTime / hop))

    # -----------------------------
    # Sub-band regions
    # -----------------------------
    Omega_size_hz = args.Omega_size
    fmax = fs // 2
    freqOverlap = Omega_size_hz // 2

    freqs_hz, _, Omega_list = build_frequency_bands(
        fs=fs,
        nfft=nfft,
        fmax=fmax,
        omega_size_hz=Omega_size_hz,
        overlap_hz=freqOverlap,
    )
    Omega_list = Omega_list[:-1]  # remove last incomplete band
    n_bands = len(Omega_list)

    # Precompute band weights W[b, f] (Windowing in frequency for each band)
    F_full = (nfft // 2) + 1
    W = np.full((n_bands, F_full), 1e-4, dtype=np.float32)
    for b_idx, Omega_idx in enumerate(Omega_list):
        w_band = np.hanning(len(Omega_idx)).astype(np.float32)
        w_band /= (w_band.max() + 1e-12)
        W[b_idx, Omega_idx] = w_band

    # -----------------------------
    # STFT
    # -----------------------------
    _, _, signalsSTFT = stft(
        signals,
        nperseg=frameSize,
        noverlap=timeOverlap,
        nfft=nfft,
        axis=1,
    )

    # -----------------------------
    # Matching pursuit parameters (in samples)
    # -----------------------------
    max_sources_mp = args.nSources
    Qn = int(2 * maxTDOA / 40) # Source peak-finder Atom
    Qw = int(2 * maxTDOA / 20) # Source peak-contribution Atom

    # -----------------------------
    # Extract local (sub-band) tau observations
    # -----------------------------
    obs_tau, obs_band, obs_frame, lags = extract_subband_tdoas_gccpairs(
        signalsSTFT=signalsSTFT,
        W=W,
        iu=iu,
        ju=ju,
        maxTDOA=maxTDOA,
        nfft=nfft,
        frameSize=frameSize,
        B=B,
        beta=1.0,
        weighting="beta",
        normalize_lag_overlap=True,
        abs_val=True,
    )

    # -----------------------------
    # Per-pair: smooth histogram -> MP peaks -> attach band support
    # -----------------------------
    tdoas_hat_pairs = {}

    L_hist = len(lags)
    t_min = float(lags[0])
    t_max = float(lags[-1] + 1)  # histogram span convention: [t_min, t_max)

    # Gaussian smoothing sigma in samples
    sigma_tau = Qw / 10.0

    for p in range(n_pairs):
        i, j = int(iu[p]), int(ju[p])
        if len(obs_tau[p]) == 0:
            continue

        tau_obs = np.asarray(obs_tau[p], dtype=np.float32)
        bands = np.asarray(obs_band[p], dtype=np.int16)

        hist_ij, bins_ij = smooth_tdoa_histogram_gaussian_fast(
            tau_obs=tau_obs,
            L=L_hist,
            t_min=t_min,
            t_max=t_max,
            sigma_tau=sigma_tau,
        )
        hist_ij = hist_ij / (hist_ij.max() + 1e-12)

        peak_taus, betas, residual, _ = matching_pursuit_tdoa_maxSources(
            y=hist_ij,
            L=L_hist,
            QN=Qn,
            QW=Qw,
            maxSources=max_sources_mp,
            t_min=t_min,
            t_max=t_max,
            verbose=False,
        )

        # peaks = attach_band_info_to_peaks_fast(
        #     tau_obs=tau_obs,
        #     bands=bands,
        #     peak_taus=peak_taus,
        #     tol_tau=Qn // 2,
        # )
        
        frames = np.asarray(obs_frame[p], dtype=np.int32)

        peaks = attach_timefreq_info_to_peaks(
            tau_obs=tau_obs,
            bands=bands,
            frames=frames,
            peak_taus=peak_taus,
            tol_tau=Qn // 2,
            n_bands=n_bands,
            n_frames=signalsSTFT.shape[-1],
        )
        tdoas_hat_pairs[(i, j)] = dict(
            peak_taus=peak_taus,   # MP peak locations (samples)
            peaks=peaks,
            hist=hist_ij,
            bins=bins_ij,
        )

        if make_plots and args.save_pair_histograms:
            fig = plt.figure(figsize=(15, 10))
            plt.subplots_adjust(bottom=0.15, left=0.18)

            plt.plot(bins_ij, hist_ij, label=f"Pair ({i},{j})")

            if peak_taus.size > 0:
                plt.stem(
                    peak_taus,
                    betas,
                    linefmt="r-",
                    markerfmt="ro",
                    basefmt=" ",
                    label="MP peaks",
                )

            colors = plt.cm.tab10(np.linspace(0, 1, theoretical_tdoas_samples.shape[0]))
            for k_src, (tdoa_true, col) in enumerate(zip(theoretical_tdoas_samples[:, i, j], colors)):
                plt.vlines(float(tdoa_true), 0, 1, color=col, linestyles="dashed", label=f"True {k_src}")

            plt.title(f"TDOA histogram + MP peaks - Pair ({i},{j})")
            plt.xlabel("TDOA lag (samples)")
            plt.ylabel("Normalized count")
            plt.legend()

            savefig(fig, f"{fig_dir}/tdoa_hist_pair_{i}_{j}",
                    save_pdf=True, save_png=False, close=not args.show_plots)

            if args.show_plots:
                plt.show()

    # -----------------------------
    # Optional: plot all hyperbolas from all pairs
    # -----------------------------
    if make_plots and args.plot_all_hyperbolas:
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_all_pairs_hyperbolas(tdoas_hat_pairs, micPos, fs, args.environmentSize, srcPos=None, ax=ax) # srcPos off to see the mess of hyperbolas
        savefig(fig, f"{fig_dir}/all_pairs_hyperbolas",
                save_pdf=True, save_png=False, close=not args.show_plots)
        if args.show_plots:
            plt.show()

    # -----------------------------
    # Association
    # -----------------------------
    sources = associate_tdoa_peaks_by_band_cosine_constrained(
        tdoas_hat_pairs,
        n_bands=n_bands,
        cosine_thresh=args.cosine_thresh,
        min_tdoas_per_source=args.min_tdoas_per_source,
        max_sources=max_sources_mp,
    )

    # -----------------------------
    # Optional: Create projected source region maps for visualization
    # (not used for localization, just to visualize where the associated sources "live" in the TF space)
    # -----------------------------


    if make_plots and args.plot_projected_clusters:
        cid2color = build_cluster_color_map(sources)

        source_maps = build_source_tf_maps(
            sources,
            tdoas_hat_pairs,
            n_bands,
            signalsSTFT.shape[-1],
        )

        hop_vis = nperseg_vis - noverlap_vis
        t_start = 0.0
        t_end = signals.shape[1] / fs

        _, times_vis_tmp, _ = stft(
            signals[0],
            fs=fs,
            nperseg=nperseg_vis,
            noverlap=noverlap_vis,
            nfft=nfft_vis,
            boundary=None,
            padded=False,
        )

        n_frames_vis = len(times_vis_tmp)

        projected_maps = project_source_maps_to_visual_grid(
            source_maps=source_maps,
            Omega_list_alg=Omega_list,
            fs=fs,
            nfft_alg=nfft,
            hop_alg=hop,
            n_frames_alg=signalsSTFT.shape[-1],
            nfft_vis=nfft_vis,
            hop_vis=hop_vis,
            n_frames_vis=n_frames_vis,
        )

        fig, ax, _, _, _ = plot_highres_spectrogram_with_projected_clusters(
            signals=signals,
            fs=fs,
            mic_idx=mic_ref,
            cid2color=cid2color,
            projected_maps=projected_maps,
            t_start=t_start,
            t_end=t_end,
            nperseg=nperseg_vis,
            noverlap=noverlap_vis,
            nfft=nfft_vis,
            alpha=0.45,
            cmap="gray_r",
            title="Recorder 0 spectrogram with projected source regions",
        )

        savefig(fig, f"{fig_dir}/spectrogram_highres_projected_clusters",
                save_pdf=True, save_png=False, close=not args.show_plots)

        if args.show_plots:
            plt.show()

    # -----------------------------
    # Optional: 3-pair feature histogram visualization
    # -----------------------------
    if make_plots and args.plot_three_pair_bands:
        available_pairs = list(tdoas_hat_pairs.keys())
        if len(available_pairs) >= 3:
            idx = np.random.choice(len(available_pairs), size=3, replace=False)
            pair_left, pair_mid, pair_right = (available_pairs[idx[0]],
                                              available_pairs[idx[1]],
                                              available_pairs[idx[2]])

            fig = plot_tdoa_peak_band_histograms_three_pairs(
                tdoas_hat_pairs,
                pair_left=pair_left,
                pair_mid=pair_mid,
                pair_right=pair_right,
                n_bands=n_bands,
                sources=sources,
                base_font=args.base_font,
                tick_step=args.tick_step,
            )
            if fig is None:
                fig = plt.gcf()

            savefig(fig, f"{fig_dir}/tdoa_peak_band_histograms_three_pairs",
                    save_pdf=True, save_png=False, close=not args.show_plots)
            if args.show_plots:
                plt.show()
        else:
            print("Not enough pairs for three-pair histogram plot.")

    # -----------------------------
    # Localization (per associated source)
    # -----------------------------
    env_bounds = (
        np.array([0.0, 0.0], float),
        np.array([args.environmentSize, args.environmentSize], float),
    )

    estimates = estimate_all_sources_positions(
        sources=sources,
        mic_positions=micPos,
        fs=fs,
        c=c,
        env_bounds=env_bounds,
        loss="huber",
        f_scale=2.0 / fs,
    )

    estPos = np.array([e["x_hat"] for e in estimates if e["x_hat"] is not None], dtype=float)
    metrics = compute_position_errors(srcPos, estPos, max_match_dist=None)

    print("RMSE:", metrics["rmse"])
    print("MAE :", metrics["mae"])
    print("Unmatched true:", metrics["unmatched_true"])
    print("Unmatched est :", metrics["unmatched_est"])

    # -----------------------------
    # Final visualization: grouped hyperbolas + truth vs estimates
    # -----------------------------
    if make_plots and args.plot_final_overlay:
        fig, ax = plt.subplots(figsize=(11, 11))

        plot_sources_hyperbolas(sources, micPos, fs, args.environmentSize, ax=ax)
        plot_truth_vs_estimates(
            micPos=micPos,
            srcPos=srcPos,
            estPos=estPos,
            roomDims=args.environmentSize,
            ax=ax,
        )

        savefig(fig, f"{fig_dir}/truth_vs_estimates",
                save_pdf=True, save_png=False, close=not args.show_plots)
        if args.show_plots:
            plt.show()

    print(f"done in {time.time() - t0:.2f}s")


# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDOA pipeline (GCC + MP + feature association)")

    # Simulation
    parser.add_argument("--nMics", type=int, default=4)
    parser.add_argument("--environmentSize", type=float, default=30.0)
    parser.add_argument("--nSources", type=int, default=6)
    parser.add_argument("--noisePower", type=float, default=40.0)
    parser.add_argument("--sourcePower", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)

    # Sub-band / history
    parser.add_argument("--Omega_size", type=int, default=800)         # Hz
    parser.add_argument("--historyTime", type=float, default=3.0)      # seconds
    parser.add_argument("--k_neighbors", type=int, default=4)

    # Association
    parser.add_argument("--cosine_thresh", type=float, default=0.7)
    parser.add_argument("--min_tdoas_per_source", type=int, default=3)

    # Plotting / output
    parser.add_argument("--clean_figures", type=int, default=1)
    parser.add_argument("--make_plots", type=int, default=1)
    parser.add_argument("--show_plots", type=int, default=1)
    parser.add_argument("--fig_dir", type=str, default="Figures")
    parser.add_argument("--base_font", type=int, default=24)
    parser.add_argument("--tick_step", type=int, default=15)

    parser.add_argument("--save_pair_histograms", type=int, default=1)
    parser.add_argument("--plot_all_hyperbolas", type=int, default=1)
    parser.add_argument("--plot_projected_clusters", type=int, default=1)
    parser.add_argument("--plot_three_pair_bands", type=int, default=1)
    parser.add_argument("--plot_final_overlay", type=int, default=1)

    args = parser.parse_args()
    main(args)