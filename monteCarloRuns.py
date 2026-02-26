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
)
from simulate_signals import _simulate
from utils import (
    theoretical_tdoas
)
from posEstimator import estimate_all_sources_positions
from metrics import compute_position_errors
import pandas as pd
import os
from itertools import product


def monte_carlo_simulation(
    n_runs,
    nMics,
    nSources,
    environmentSize,
    noisePower,
    sourcePower,
):
    position_errors_all_runs = []
    position_maes_all_runs = []
    missed_sources_all_runs = []
    start_time = time.time()

    for run in range(n_runs):
        print(f"Monte Carlo Run {run + 1}/{n_runs}")

        fs, micPos, signals, srcPos = _simulate(
            nMics=nMics,
            environmentSize=environmentSize,
            nSources=nSources,
            noisePower=noisePower,
            sourcePower=sourcePower,
        )

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

        # Theoretical TDOAs (simulation/diagnostic)
        _, theoretical_tdoas_samples = theoretical_tdoas(srcPos, micPos, fs=fs)

        # -----------------------------
        # STFT parameters
        # -----------------------------
        frameSize = max(maxTDOA * 2, avgTDOA * 4)
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

        # Precompute band weights W[b, f]
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
        # Matching pursuit parameters in samples
        # -----------------------------
        Qn = int(2 * maxTDOA / 40)
        Qw = int(2 * maxTDOA / 20)

        min_tdoas_per_source = [2,3,4,4] # Hihger for more mics, since more pairs => more spurious peaks, so we want more TDOAs to confirm a source

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
                maxSources=nSources,
                t_min=t_min,
                t_max=t_max,
                verbose=False,
            )

            peaks = attach_band_info_to_peaks_fast(
                tau_obs=tau_obs,
                bands=bands,
                peak_taus=peak_taus,
                tol_tau=Qn // 2,
            )

            tdoas_hat_pairs[(i, j)] = dict(
                peak_taus=peak_taus,  # MP peak locations (samples)
                peaks=peaks,  # enriched peaks with band_counts
                hist=hist_ij,
                bins=bins_ij,
            )

        # -----------------------------
        # Association
        # -----------------------------
        sources = associate_tdoa_peaks_by_band_cosine_constrained(
            tdoas_hat_pairs,
            n_bands=n_bands,
            cosine_thresh=args.cosine_thresh,
            min_tdoas_per_source=min_tdoas_per_source[nMics-3],
            max_sources=nSources,
        )

        if len(sources) == 0:
            print("No sources found, skipping this run")
            continue

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

        position_errors_all_runs.append(metrics["rmse"])
        position_maes_all_runs.append(metrics["mae"])
        missed_sources_all_runs.append(len(metrics["unmatched_true"]) + len(metrics["unmatched_est"]))

        print("run done")

    end_time = time.time()
    print(f"\nTotal time for {n_runs} Monte Carlo runs: {end_time - start_time:.2f} seconds")
    return np.array(position_errors_all_runs), np.array(position_maes_all_runs), np.array(missed_sources_all_runs)


def run_one_combo(n_runs, nn, ns, es, npow, spow, seed=0):
    """
    Runs one parameter combination and returns a flat dict (one row).

    where:
      per_source_errors: shape (n_runs, n_sources) or (n_runs, something)
      maes:             shape (n_runs,) or (n_runs, ...)
      missed:           shape (n_runs,) or (n_runs, ...)
    Adjust the summaries as needed.
    """
    t0 = time.time()

    rng_seed = int(seed)  # make deterministic per-combo
    np.random.seed(rng_seed)

    per_source_errors, maes, missed = monte_carlo_simulation(
        n_runs, nn, ns, es, npow, spow
    )

    per_source_errors = np.asarray(per_source_errors)
    maes = np.asarray(maes)
    missed = np.asarray(missed)

    out = {
        "nMics": nn,
        "nSources": ns,
        "environmentSize": es,
        "noisePower": npow,
        "sourcePower": spow,
        "n_runs": n_runs,
        "time_sec": time.time() - t0,
    }

    if per_source_errors.ndim == 2:
        out["rmse_mean_over_sources"] = float(np.nanmean(per_source_errors))
        out["rmse_std_over_sources"]  = float(np.nanstd(per_source_errors))

        mean_per_src = np.nanmean(per_source_errors, axis=0)
        std_per_src  = np.nanstd(per_source_errors, axis=0)
        for k, v in enumerate(mean_per_src):
            out[f"rmse_mean_src{k}"] = float(v)
        for k, v in enumerate(std_per_src):
            out[f"rmse_std_src{k}"] = float(v)
    else:
        out["rmse_mean"] = float(np.nanmean(per_source_errors))
        out["rmse_std"]  = float(np.nanstd(per_source_errors))

    out["mae_mean"] = float(np.nanmean(maes))
    out["mae_std"]  = float(np.nanstd(maes))

    out["missed_mean"] = float(np.nanmean(missed))
    out["missed_std"]  = float(np.nanstd(missed))

    return out


def build_param_grid(nMics, nSources, environmentSizes, noisePowers, sourcePowers):
    for nn, ns, es, npow, spow in product(nMics, nSources, environmentSizes, noisePowers, sourcePowers):
        yield (nn, ns, es, npow, spow)


def run_sweep(
    n_runs,
    nMics,
    nSources,
    environmentSizes,
    noisePowers,
    sourcePowers,
    out_csv="mc_results.csv",
    base_seed=0,
):
    rows = []
    grid = list(build_param_grid(nMics, nSources, environmentSizes, noisePowers, sourcePowers))

    # Optional: resume if file exists
    done_keys = set()
    if os.path.exists(out_csv):
        df_prev = pd.read_csv(out_csv)
        done_keys = set(
            zip(df_prev["nMics"], df_prev["nSources"], df_prev["environmentSize"], df_prev["noisePower"], df_prev["sourcePower"])
        )

    for idx, (nn, ns, es, npow, spow) in enumerate(grid):
        key = (nn, ns, es, npow, spow)
        if key in done_keys:
            continue

        print(f"Running: nMics={nn}, nSources={ns}, env={es}, noise={npow}, source={spow}")

        # deterministic per-combo seed:
        seed = base_seed + idx

        row = run_one_combo(n_runs, nn, ns, es, npow, spow, seed=seed)
        rows.append(row)

        # write incremental
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    # return full dataframe
    df = pd.read_csv(out_csv)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation for Source Localization")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of Monte Carlo runs")
    parser.add_argument("--environmentSize", type=float, default=30, help="Dimensions of the squared environment (meters)")
    parser.add_argument("--noisePower", type=float, default=40, help="Power of the background noise")
    parser.add_argument("--sourcePower", type=float, default=80, help="Power of the source signals")
    parser.add_argument("--k_neighbors", type=int, default=4, help="Number of nearest neighbors for mic pairing")
    parser.add_argument("--historyTime", type=float, default=3.0, help="Time window (seconds) for TDOA observation history")
    parser.add_argument("--Omega_size", type=float, default=800, help="Size of frequency bands (Hz) for sub-band processing")
    parser.add_argument("--cosine_thresh", type=float, default=0.7, help="Cosine similarity threshold for TDOA peak association")
    args = parser.parse_args()

    nMics = [3, 4, 5, 6]
    nSources = [2, 4, 8]
    environmentSizes = [30]
    noisePowers = [40, 60]
    sourcePowers = [80]

    df_results = run_sweep(
        n_runs=50,
        nMics=nMics,
        nSources=nSources,
        environmentSizes=environmentSizes,
        noisePowers=noisePowers,
        sourcePowers=sourcePowers,
        out_csv="monte_carlo_results.csv",
        base_seed=123,
    )

    print("done")
