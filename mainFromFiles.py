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
    extract_subband_tdoas_windowed_search,
    omega_list_to_hz_ranges,
    build_pair_band_reliability_mask,
    matching_pursuit_tdoa
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
    gps_to_local
)
from posEstimator import estimate_all_sources_positions
from metrics import compute_position_errors
import librosa


# =============================================================================
# Main
# =============================================================================

def main(args):
    t0 = time.time()

    # -----------------------------
    # High-level flags / settings
    # -----------------------------

    make_plots = bool(args.make_plots)
    fig_dir = args.fig_dir
    if args.clean_figures:
        clear_folder(fig_dir)

    if make_plots:
        set_plot_style(base_font=args.base_font)

    # ---------------------------
    # Load data (Use this as example)
    # ---------------------------
    fs = 32000
    aud26, _ = librosa.load("/Users/alejandro/Documents/PhD/Data/VizinaGPSDataFirstTest/Vizina_rec/AUD26/245AAA0563FBE586_20260311_073500_SYNC.WAV", sr=fs)
    aud27, _ = librosa.load("/Users/alejandro/Documents/PhD/Data/VizinaGPSDataFirstTest/Vizina_rec/AUD27/249C6006641FC215_20260311_073500_SYNC.WAV", sr=fs)
    aud28, _ = librosa.load("/Users/alejandro/Documents/PhD/Data/VizinaGPSDataFirstTest/Vizina_rec/AUD28/2453AC0564201551_20260311_073500_SYNC.WAV", sr=fs)
    aud29, _ = librosa.load("/Users/alejandro/Documents/PhD/Data/VizinaGPSDataFirstTest/Vizina_rec/AUD29/24E1440163FBE4DB_20260311_073500_SYNC.WAV", sr=fs)
    aud30, _ = librosa.load("/Users/alejandro/Documents/PhD/Data/VizinaGPSDataFirstTest/Vizina_rec/AUD30/245AAA0563FBE51D_20260311_073500_SYNC.WAV", sr=fs)
    signals = np.stack([aud27, aud29], axis=0)
    start = 5260000
    dur = int(args.historyTime * fs)
    signals = signals[:, start:start + dur]

    coords_gps = np.array([
        # [49.8399506, 14.0997564],
        [49.8396086, 14.0996661],
        # [49.8392744, 14.1002700],
        [49.8395239, 14.0991978],
        # [49.8392422, 14.0991011]
    ])    
    micPos = gps_to_local(coords_gps)
    ## Excess area to localize sources
    excess = 80
    micPos = micPos - micPos.min(axis=0) + np.array([excess, excess])
    environmentSize = micPos.max(axis=0) + excess
    environmentSize = max(environmentSize)  # make square
    
    # Filter signals if desired
    # nyquist = fs / 2
    # lowcut = 100
    # highcut = nyquist - 100
    # order = 4
    # low = lowcut / nyquist
    # high = min(highcut / nyquist, 0.99)  # Cap slightly below Nyquist
    #
    # # Design Butterworth bandpass filter
    # b, a = butter(order, [low, high], btype='band')
    #
    # # Apply filter (filtfilt for zero-phase filtering)
    # if signals.ndim == 1:
    #     signals = filtfilt(b, a, signals)
    # else:
    #     signals = filtfilt(b, a, signals, axis=1)
    
    
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
    pair_max_tdoa = np.ceil((pair_dists / c) * fs).astype(int)

    # maxMicDist = float(pair_dists.max())
    # avgMicDist = float(pair_dists.mean())

    # maxTDOA = int((maxMicDist / c) * fs)
    # avgTDOA = int((avgMicDist / c) * fs)

    # -----------------------------
    # STFT parameters
    # -----------------------------
    # frameSize = max(maxTDOA * 2, avgTDOA * 4)
    frameSize = 2048 * 4
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

    valid_mask = build_pair_band_reliability_mask(
        signalsSTFT,
        W,
        iu,
        ju,
        pair_mode="min",
        energy_threshold_mode="percentile",
        energy_percentile=20.0,
        coherence_threshold_mode="percentile",
        coherence_percentile=20.0,
        combine_mode="and",
    )

    # -----------------------------
    # Matching pursuit parameters (in samples)
    # -----------------------------
    Qn = 2 * pair_max_tdoa / 40 # Source peak-finder Atom
    Qw = 2 * pair_max_tdoa / 20 # Source peak-contribution Atom
    Qn = np.array(Qn, dtype=int)
    Qw = np.array(Qw, dtype=int)

    # -----------------------------
    # Extract local (sub-band) tau observations
    # -----------------------------
    band_ranges_hz = omega_list_to_hz_ranges(Omega_list, fs=fs, nfft=nfft)
    obs_tau, obs_band, obs_frame, lags = extract_subband_tdoas_windowed_search(
        signals=signals,
        fs=fs,
        band_ranges_hz=band_ranges_hz,
        iu=iu,
        ju=ju,
        pair_max_tdoa=pair_max_tdoa,
        frameSize=frameSize,
        hop=hop,
        B=B,
        valid_mask=valid_mask,
        taper_hz=0.0,
        normalize=True,
        abs_val=True,
    )

    # -----------------------------
    # Per-pair: smooth histogram -> MP peaks -> attach band support
    # -----------------------------
    tdoas_hat_pairs = {}

    # L_hist = len(lags)
    # t_min = float(lags[0])
    # t_max = float(lags[-1] + 1)  # histogram span convention: [t_min, t_max)


    for p in range(n_pairs):
        i, j = int(iu[p]), int(ju[p])
        if len(obs_tau[p]) == 0:
            continue

        tau_obs = np.asarray(obs_tau[p], dtype=np.float32)
        bands = np.asarray(obs_band[p], dtype=np.int16)

        L_hist = pair_max_tdoa[p]*2 + 1
        t_min = -pair_max_tdoa[p]
        t_max = pair_max_tdoa[p] + 1
        # Gaussian smoothing sigma in samples
        sigma_tau = Qw[p] / 10.0

        hist_ij, bins_ij = smooth_tdoa_histogram_gaussian_fast(
            tau_obs=tau_obs,
            L=L_hist,
            t_min=t_min,
            t_max=t_max,
            sigma_tau=sigma_tau,
        )
        hist_ij = hist_ij / (hist_ij.max() + 1e-12)

        hist_ij = hist_ij - np.median(hist_ij)

        peak_taus, betas, residual, _ = matching_pursuit_tdoa(
            y=hist_ij,
            L=L_hist,
            QN=Qn[p],
            QW=Qw[p],
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
            tol_tau=Qn[p] // 3,
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
        plot_all_pairs_hyperbolas(tdoas_hat_pairs, micPos, fs, environmentSize, srcPos=None, ax=ax) # srcPos off to see the mess of hyperbolas
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
        max_sources=None,
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
        np.array([environmentSize, environmentSize], float),
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


    # -----------------------------
    # Final visualization: grouped hyperbolas + truth vs estimates
    # -----------------------------
    if make_plots and args.plot_final_overlay:
        fig, ax = plt.subplots(figsize=(11, 11))

        plot_sources_hyperbolas(sources, micPos, fs, environmentSize, ax=ax)
        plot_truth_vs_estimates(
            micPos=micPos,
            srcPos=None,
            estPos=estPos,
            roomDims=environmentSize,
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
    parser.add_argument("--nSources", type=int, default=None)

    # Sub-band / history
    parser.add_argument("--Omega_size", type=int, default=500)         # Hz
    parser.add_argument("--historyTime", type=float, default=60.0)      # seconds
    parser.add_argument("--k_neighbors", type=int, default=3)

    # Association
    parser.add_argument("--cosine_thresh", type=float, default=0.7)
    parser.add_argument("--min_tdoas_per_source", type=int, default=1)

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