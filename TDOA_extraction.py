import numpy as np

"""
Convention
----------------------
All TDOA / delay variables (tau) in this module are expressed in samples,
unless explicitly stated otherwise.
 
Pipeline:
  1) Extract local sub-band TDOA observations via GCC argmax.
  2) Aggregate those observations into a smoothed histogram on a tau grid.
  3) Apply matching pursuit on the histogram to obtain peak_taus (samples).
"""


# ============================================================
# Frequency band construction
# ============================================================

def build_frequency_bands(fs, nfft, fmax, omega_size_hz, overlap_hz):
    """
    Build overlapping frequency-band index sets for sub-band processing.

    Parameters
    ----------
    fs : float
        Sampling rate.
    nfft : int
        FFT size (defines frequency bin spacing df = fs/nfft).
    fmax : float
        Maximum frequency considered [Hz] (upper cutoff).
    omega_size_hz : float
        Sub-band width [Hz].
    overlap_hz : float
        Sub-band overlap [Hz].

    Returns
    -------
    freqs_hz : (F,) ndarray
        Frequency axis for rFFT bins in Hz, where F = nfft//2 + 1.
    omega_bins_full : (F,) ndarray
        Angular frequency axis (rad/s) for the same rFFT bins.
    Omega_list : list of 1D ndarrays
        List of arrays of bin indices, one per sub-band.
    """
    F = nfft // 2 + 1
    df = fs / float(nfft)

    freqs_hz = np.arange(F) * df
    omega_bins_full = 2.0 * np.pi * freqs_hz

    omega_bins = int(round(omega_size_hz / df))
    overlap_bins = int(round(overlap_hz / df))
    step_bins = omega_bins - overlap_bins
    if step_bins <= 0:
        raise ValueError("overlap_hz too large: step_bins <= 0.")

    max_bin = int(np.floor(fmax / df))
    max_bin = min(max_bin, F - 1)

    Omega_list = []
    start_bin = 0
    while start_bin <= max_bin:
        end_bin = min(start_bin + omega_bins, max_bin + 1)
        if end_bin > start_bin:
            Omega_list.append(np.arange(start_bin, end_bin, dtype=np.int32))
        start_bin += step_bins

    return freqs_hz, omega_bins_full, Omega_list


# ============================================================
# Histogram smoothing
# ============================================================

def smooth_tdoa_histogram_gaussian_fast(tau_obs, L, t_min, t_max, sigma_tau):
    """
    Smooth a set of discrete TDOA observations using Gaussian kernel evaluation
    on a fixed tau grid of length L.

    Notes
    -----
    This computes:
        y[k] = sum_n exp(-0.5 * ((bin_center[k] - tau_obs[n]) / sigma_tau)^2)

    Parameters
    ----------
    tau_obs : (N,) array-like
        TDOA observations in samples (typically integer lags from GCC).
    L : int
        Number of histogram bins on [t_min, t_max).
    t_min, t_max : float
        Tau range covered by the grid, in samples.
    sigma_tau : float
        Gaussian standard deviation in samples.

    Returns
    -------
    y : (L,) ndarray
        Smoothed histogram values.
    bin_centers : (L,) ndarray
        Tau grid points (bin centers) in samples.
    """
    tau_obs = np.asarray(tau_obs, dtype=float)
    bin_centers = np.linspace(t_min, t_max, L, endpoint=False)

    diff = bin_centers[:, None] - tau_obs[None, :]
    y = np.exp(-0.5 * (diff / sigma_tau) ** 2).sum(axis=1)

    return y, bin_centers


# ============================================================
# Matching pursuit
# ============================================================

def _blackman_atom(L, Q):
    """
    Build a Blackman-window atom of length L centered at the histogram midpoint.

    Parameters
    ----------
    L : int
        Histogram length.
    Q : int
        Atom width in bins. Forced to be odd for symmetry.
    """
    if Q % 2 == 0:
        Q += 1
    w = np.blackman(Q)

    atom = np.zeros(L, dtype=np.float32)
    mid = L // 2
    half = Q // 2
    start = max(0, mid - half)
    end = min(L, mid + half + 1)
    atom[start:end] = w
    return atom


def _shift_atom_linear(atom, shift):
    """
    Shift an atom by an integer amount without wrap-around (linear shift).

    Parameters
    ----------
    atom : (L,) ndarray
    shift : int
        Positive shift moves content to the right.

    Returns
    -------
    out : (L,) ndarray
        Shifted atom with zero padding.
    """
    L = atom.shape[0]
    out = np.zeros_like(atom)
    s = int(shift)

    if s >= 0:
        if s < L:
            out[s:] = atom[: L - s]
    else:
        s = -s
        if s < L:
            out[: L - s] = atom[s:]
    return out


def _energy(vec):
    """Squared L2 norm."""
    return float(np.dot(vec, vec))


# ============================================================
# Matching pursuit peak extraction
# ============================================================

def matching_pursuit_tdoa_maxSources(
    y,
    L,
    QN,
    QW,
    maxSources,
    t_min,
    t_max,
    verbose=False,
):
    """
    Matching pursuit over a linear 1D histogram defined on a tau grid.

    Parameters
    ----------
    y : (L,) ndarray
        Smoothed histogram values (nonnegative).
    L : int
        Histogram length (kept for compatibility; must match len(y)).
    QN, QW : int
        Atom widths in bins:
          - QN: narrow atom for correlation / peak search
          - QW: wide atom for subtraction (suppresses neighborhood after selecting peak)
    maxSources : int
        Maximum number of peaks to extract (iterations).
    t_min, t_max : float
        Tau range (in samples) of the histogram grid.
    verbose : bool
        If True, returns debug traces and prints iteration info.

    Returns
    -------
    peak_taus : (P,) ndarray
        Estimated peak locations in samples (float grid locations).
    betas : (P,) ndarray
        MP coefficients.
    residual : (L,) ndarray
        Residual histogram after subtraction.
    debug : dict or None
        Debug traces if verbose else None.
    """
    y = np.asarray(y, dtype=float).copy()
    if y.ndim != 1 or len(y) != L:
        raise ValueError("y must be 1D of length L")

    cN = _blackman_atom(L, QN)
    cW = _blackman_atom(L, QW)
    E_CN = _energy(cN)

    span = float(t_max - t_min)

    peak_taus, betas = [], []
    residual = y.copy()

    traces = None
    if verbose:
        traces = {"a_list": [], "i_star": [], "beta": [], "y_snapshots": [residual.copy()]}

    for j in range(int(maxSources)):
        # Padding reduces edge bias when correlating near the boundaries
        pad = int(QW)
        res_padded = np.pad(residual, pad, mode="reflect")
        a_full = np.correlate(res_padded, cN, mode="same")
        a = a_full[pad:pad + L]

        i_star = int(np.argmax(a))

        # i_star -> tau on the same grid used in smoothing
        tau_hat = t_min + i_star * span / L
        beta_j = float(a[i_star] / (E_CN + 1e-12))

        # Subtract a wide atom centered at i_star to avoid selecting the same peak again
        cW_shift = _shift_atom_linear(cW, i_star - L // 2)
        residual = np.maximum(residual - cW_shift, 0.0)

        peak_taus.append(tau_hat)
        betas.append(beta_j)

        if verbose:
            print(f"[j={j}] i*={i_star}, tau_hat={tau_hat:.6f} samp, beta={beta_j:.4f}")
            traces["a_list"].append(a.copy())
            traces["i_star"].append(i_star)
            traces["beta"].append(beta_j)
            traces["y_snapshots"].append(residual.copy())

    return np.asarray(peak_taus), np.asarray(betas), residual, traces


# ============================================================
# Pairwise GCC
# ============================================================

def gcc_pairs(
    signals_dft: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    tdoa_max: int = None,
    abs_val: bool = True,
    ifft: bool = True,
    n_dft_bins: int = None,
    beta: float = 1.0,
    weighting: str = "beta",
    coh_exponent: float = 1.0,
    eps: float = 1e-10,
    normalize_lag_overlap: bool = True,
    frame_len: int = None,
):
    """
    Pairwise GCC between microphone pairs (pair_i[k], pair_j[k]).

    Notes
    -----
    - If `ifft=True`, this returns time-domain GCC curves indexed by integer lags (samples).
    - The lag axis is centered so that lag=0 is at the center index before cropping.
    - If `tdoa_max` is provided, the output is cropped to lags:
        [-tdoa_max, ..., tdoa_max-1]  (length 2*tdoa_max)

    Parameters
    ----------
    signals_dft : (M, F) complex ndarray
        rFFT-domain signals per microphone.
    pair_i, pair_j : (P,) int arrays
        Microphone pair indices, P = number of pairs.
    tdoa_max : int or None
        If provided and ifft=True, crop to lags in [-tdoa_max, tdoa_max).
    abs_val : bool
        If True, take abs() of time-domain GCC.
    ifft : bool
        If True, return time-domain GCC via irfft.
        If False, return weighted cross-spectrum (frequency domain).
    n_dft_bins : int or None
        IFFT length. If None, inferred from rFFT length F as n = 2*(F-1).
    weighting : {"beta","phat","gcc","roth","scot","coherence"}
        Weighting scheme for GCC.
    normalize_lag_overlap : bool
        If True and frame_len is provided, divide by (frame_len - |lag|) to account for
        reduced overlap near large lags.
    frame_len : int or None
        Frame length (samples) used for overlap normalization.

    Returns
    -------
    gcc_out : ndarray
        If ifft=True: shape (P, Tlag), real.
        If ifft=False: shape (P, F), complex.
    lags : (Tlag,) ndarray
        Integer lag axis in samples. If cropped, lags are in [-tdoa_max, ..., tdoa_max-1].
    """
    X = signals_dft
    if X.ndim != 2:
        raise ValueError("signals_dft must have shape (M, F)")

    M, F = X.shape
    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)

    # Cross-spectrum for each pair: (P, F)
    cross = X[pair_i] * np.conj(X[pair_j])

    weighting = weighting.lower()
    if weighting in ("roth", "scot", "coherence"):
        auto = (np.abs(X) ** 2).astype(np.float32)  # (M, F)
        Sxx = auto[pair_i]  # (P, F)
        Syy = auto[pair_j]  # (P, F)

    # Weighting. This was used only for testing, all results in the paper are with phat weighting (beta=1).
    if weighting == "beta":
        denom = (np.abs(cross) ** beta) + eps
        G = cross / denom
    elif weighting == "phat":
        denom = np.abs(cross) + eps
        G = cross / denom
    elif weighting == "gcc":
        G = cross
    elif weighting == "roth":
        G = cross / (Sxx + eps)
    elif weighting == "scot":
        denom = np.sqrt(Sxx * Syy) + eps
        G = cross / denom
    elif weighting == "coherence":
        denom = np.sqrt(Sxx * Syy) + eps
        gamma = cross / denom
        if coh_exponent == 1.0:
            G = gamma
        else:
            mag = np.abs(gamma)
            G = (mag ** (coh_exponent - 1.0)) * gamma
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    if not ifft:
        n = (F - 1) * 2
        max_shift = n // 2
        lags = np.arange(-max_shift, max_shift, dtype=int)
        return G, lags

    if n_dft_bins is None:
        n_dft_bins = (F - 1) * 2

    g = np.fft.irfft(G, n=n_dft_bins, axis=-1)

    if abs_val:
        g = np.abs(g)

    # Shift so lag=0 is at the center
    half = g.shape[-1] // 2
    g = np.concatenate((g[..., half:], g[..., :half]), axis=-1)

    # Full lag axis
    n = g.shape[-1]
    max_shift = n // 2
    lags = np.arange(-max_shift, max_shift, dtype=int)

    # Crop if requested
    if tdoa_max is not None:
        tdoa_max = int(tdoa_max)
        center = n // 2
        start = center - tdoa_max
        end = center + tdoa_max
        g = g[..., start:end]  # (P, 2*tdoa_max)
        lags = np.arange(-tdoa_max, tdoa_max, dtype=int)

        if normalize_lag_overlap and frame_len is not None:
            overlaps = frame_len - np.abs(lags)
            overlaps = np.maximum(overlaps, 1)
            g = g / overlaps[None, :]

    return g, lags


# ============================================================
# Sub-band GCC observation extraction
# ============================================================

def extract_subband_tdoas_gccpairs(
    signalsSTFT,      # (M, F, T)
    W,                # (n_bands, F)
    iu, ju,           # arrays of mic-pair indices
    maxTDOA,
    nfft,
    frameSize,
    B=None,
    beta=1.0,
    weighting="beta",
    normalize_lag_overlap=True,
    abs_val=True,
):
    """
    Extract local TDOA observations (tau) per pair from sub-band GCC maxima.

    For each time frame and each sub-band, compute GCC for all requested pairs and
    store the argmax lag.

    Parameters
    ----------
    signalsSTFT : (M, F, T) complex ndarray
        STFT (or any frequency-domain frame representation) per microphone.
    W : (n_bands, F) ndarray
        Band masks in frequency (one row per sub-band).
    iu, ju : (P,) int arrays
        Pair list: p-th pair is (iu[p], ju[p]).
    maxTDOA : int
        Maximum TDOA lag in samples (GCC is cropped to [-maxTDOA, ..., maxTDOA-1]).
    nfft : int
        IFFT length used inside GCC (passed to gcc_pairs as n_dft_bins).
    frameSize : int
        Frame length in samples (used for lag-overlap normalization).
    B : int or None
        If provided, process only the first B frames.
    beta, weighting, normalize_lag_overlap, abs_val
        Parameters forwarded to gcc_pairs.

    Returns
    -------
    obs_tau : list of length P
        obs_tau[p] is a list of raw lag observations (integer samples) for pair p.
    obs_band : list of length P
        obs_band[p] is the aligned band index for each observation in obs_tau[p].
    obs_frame : list of length P
        obs_frame[p] is the aligned frame index for each observation in obs_tau[p].
    lags : (2*maxTDOA,) ndarray
        Lag axis (samples) returned by gcc_pairs.
    """
    M, F, T = signalsSTFT.shape
    iu = np.asarray(iu, dtype=int)
    ju = np.asarray(ju, dtype=int)

    if W is None:
        W = np.ones((1, F), dtype=np.float32)

    P = len(iu)
    n_bands = W.shape[0]

    T_use = T if B is None else min(int(B), T)

    obs_tau = [[] for _ in range(P)]    # raw lag observations (samples)
    obs_band = [[] for _ in range(P)]   # band index for each observation
    obs_frame = [[] for _ in range(P)]  # frame index for each observation

    lags_last = None

    for t in range(T_use):
        Xt = signalsSTFT[:, :, t].astype(np.complex64)

        for b in range(n_bands):
            Xb = Xt * W[b][None, :]

            gcc, lags = gcc_pairs(
                Xb,
                iu,
                ju,
                tdoa_max=maxTDOA,
                beta=beta,
                ifft=True,
                n_dft_bins=nfft,
                weighting=weighting,
                frame_len=frameSize,
                normalize_lag_overlap=normalize_lag_overlap,
                abs_val=abs_val,
            )

            # argmax lag per pair
            kmax = np.argmax(gcc, axis=-1)
            taus = lags[kmax]  # integer lags in samples

            for p in range(P):
                obs_tau[p].append(int(taus[p]))
                obs_band[p].append(int(b))
                obs_frame[p].append(int(t))

            lags_last = lags

    if lags_last is None:
        lags_last = np.arange(-maxTDOA, maxTDOA, dtype=int)

    return obs_tau, obs_band, obs_frame, lags_last

def extract_subband_tdoas_windowed_search(
    signals,
    fs,
    band_ranges_hz,
    iu,
    ju,
    pair_max_tdoa,
    frameSize,
    hop,
    *,
    B=None,
    valid_mask=None,   # (P, BANDS, T)
    taper_hz=0.0,
    normalize=True,
    abs_val=True,
    global_max_tdoa=None,
):
    """
    Extract local TDOA observations using:
      - short analysis window on mic i
      - larger lag-search interval on mic j

    This decouples frameSize from the maximum pair delay.

    Parameters
    ----------
    signals : (M, N) ndarray
        Time-domain microphone signals.
    fs : float
    band_ranges_hz : list[(f_lo, f_hi)]
        Frequency ranges for each sub-band.
    iu, ju : (P,) int arrays
        Pair indices.
    pair_max_tdoa : (P,) int array
        Pair-dependent maximum feasible lag in samples.
    frameSize : int
        Short analysis window size.
    hop : int
        Hop size between frames.
    B : int or None
        Number of frames to use.
    valid_mask : (P, n_bands, T) bool ndarray or None
        Optional mask to skip low-energy zones.
    taper_hz : float
        Optional taper width when building full-signal band masks.
    normalize : bool
        Use normalized cross-correlation.
    abs_val : bool
        Take absolute value before argmax.
    global_max_tdoa : int or None
        Global lag axis returned for downstream histogram code.

    Returns
    -------
    obs_tau : list of length P
    obs_band : list of length P
    obs_frame : list of length P
    lags : ndarray
        Global lag axis for downstream use.
    """
    signals = np.asarray(signals, dtype=np.float32)
    iu = np.asarray(iu, dtype=int)
    ju = np.asarray(ju, dtype=int)
    pair_max_tdoa = np.asarray(pair_max_tdoa, dtype=int)

    if signals.ndim != 2:
        raise ValueError("signals must have shape (M, N)")
    if iu.shape != ju.shape:
        raise ValueError("iu and ju must have the same shape")
    if pair_max_tdoa.shape[0] != iu.shape[0]:
        raise ValueError("pair_max_tdoa must have one entry per pair")

    M, N = signals.shape
    P = iu.size
    n_bands = len(band_ranges_hz)

    if N < frameSize:
        raise ValueError("frameSize is longer than the signals")

    n_frames_total = 1 + (N - frameSize) // hop
    T_use = n_frames_total if B is None else min(int(B), n_frames_total)

    if global_max_tdoa is None:
        global_max_tdoa = int(np.max(pair_max_tdoa))

    obs_tau = [[] for _ in range(P)]
    obs_band = [[] for _ in range(P)]
    obs_frame = [[] for _ in range(P)]

    # Full-signal FFT once
    X_full = np.fft.rfft(signals, axis=1)
    freqs_full = np.fft.rfftfreq(N, d=1.0 / fs)

    win = np.hanning(frameSize).astype(np.float32)

    for b, (f_lo, f_hi) in enumerate(band_ranges_hz):
        mask = build_fullband_rfft_mask(freqs_full, f_lo, f_hi, taper_hz=taper_hz)
        if not np.any(mask > 0):
            continue

        Xb = X_full * mask[None, :]
        xb = np.fft.irfft(Xb, n=N, axis=1).astype(np.float32)

        for t in range(T_use):
            s = t * hop
            e = s + frameSize

            if valid_mask is None:
                active_idx = np.arange(P, dtype=int)
            else:
                active_idx = np.where(valid_mask[:, b, t])[0]

            if active_idx.size == 0:
                continue

            for p in active_idx:
                i = int(iu[p])
                j = int(ju[p])
                maxlag = int(pair_max_tdoa[p])

                x_ref = xb[i, s:e]

                search_start = max(0, s - maxlag)
                search_end = min(N, e + maxlag)
                y_search = xb[j, search_start:search_end]

                lag_min = search_start - s

                # scores, lags_local = _xcorr_curve_short_in_long(
                #     x_ref,
                #     y_searc3h,
                #     lag_min=lag_min,
                #     window=win,
                #     normalize=normalize,
                #     abs_val=abs_val,
                # )
                scores, lags_local = fast_xcorr_short_in_long(
                    x_ref,
                    y_search,
                    lag_min=lag_min,
                    maxlag=maxlag,
                )

                if scores.size == 0:
                    continue

                k = int(np.argmax(np.abs(scores)))
                tau_hat = -int(lags_local[k])

                obs_tau[p].append(tau_hat)
                obs_band[p].append(int(b))
                obs_frame[p].append(int(t))

    lags = np.arange(-global_max_tdoa, global_max_tdoa + 1, dtype=int)
    return obs_tau, obs_band, obs_frame, lags

def fast_xcorr_short_in_long(x_ref, y_search, lag_min, maxlag, eps=1e-12):
    """
    Correlate short x_ref against all valid length-L windows inside y_search.

    Returns scores indexed by lag where lag means:
        lag = (start index of y window) - (start index of x window)

    IMPORTANT:
    This raw lag has the opposite sign of your pipeline convention
    tau_ij = tof_i - tof_j, so the caller should negate it.
    """
    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_search = np.asarray(y_search, dtype=np.float32)

    L = len(x_ref)
    M = len(y_search)

    if M < L:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)

    # Reverse x so convolution gives sliding correlation
    xr = x_ref[::-1]

    nfft = 1
    while nfft < M + L - 1:
        nfft *= 2

    X = np.fft.rfft(xr, n=nfft)
    Y = np.fft.rfft(y_search, n=nfft)
    conv = np.fft.irfft(X * Y, n=nfft)

    # Valid windows: y[k:k+L], k = 0 ... M-L
    scores = conv[L-1 : L-1 + (M - L + 1)]

    lags = lag_min + np.arange(M - L + 1, dtype=int)

    valid = (lags >= -maxlag) & (lags <= maxlag)
    return scores[valid].astype(np.float32), lags[valid]

def build_fullband_rfft_mask(freqs_full, f_lo, f_hi, taper_hz=0.0):
    """
    Build a real-valued band mask on a full-signal rFFT frequency axis.
    """
    mask = np.zeros_like(freqs_full, dtype=np.float32)

    if f_hi <= f_lo:
        return mask

    if taper_hz <= 0:
        mask[(freqs_full >= f_lo) & (freqs_full <= f_hi)] = 1.0
        return mask

    # Cosine ramps
    left0 = max(0.0, f_lo - taper_hz)
    left1 = f_lo
    right0 = f_hi
    right1 = f_hi + taper_hz

    passband = (freqs_full >= left1) & (freqs_full <= right0)
    mask[passband] = 1.0

    left = (freqs_full >= left0) & (freqs_full < left1)
    if np.any(left):
        x = (freqs_full[left] - left0) / max(left1 - left0, 1e-12)
        mask[left] = 0.5 - 0.5 * np.cos(np.pi * x)

    right = (freqs_full > right0) & (freqs_full <= right1)
    if np.any(right):
        x = (freqs_full[right] - right0) / max(right1 - right0, 1e-12)
        mask[right] = 0.5 + 0.5 * np.cos(np.pi * x)

    return mask


def omega_list_to_hz_ranges(Omega_list, fs, nfft):
    """
    Convert Omega_list (FFT-bin bands) into frequency ranges in Hz.

    Returns
    -------
    band_ranges_hz : list of tuples
        [(f_lo, f_hi), ...] for each band.
    """
    freqs = np.arange(nfft // 2 + 1) * (fs / float(nfft))
    band_ranges_hz = []

    for Omega_idx in Omega_list:
        Omega_idx = np.asarray(Omega_idx, dtype=int)
        if Omega_idx.size == 0:
            band_ranges_hz.append((0.0, 0.0))
            continue
        f_lo = float(freqs[Omega_idx.min()])
        f_hi = float(freqs[Omega_idx.max()])
        band_ranges_hz.append((f_lo, f_hi))

    return band_ranges_hz

def build_pair_band_reliability_mask(
    signalsSTFT,
    W,
    iu,
    ju,
    *,
    pair_mode="min",
    energy_threshold_mode="percentile",
    energy_percentile=50.0,
    energy_rel_factor=1.5,
    coherence_threshold_mode="percentile",
    coherence_percentile=50.0,
    coherence_rel_factor=1.0,
    combine_mode="and",
    score_alpha=0.5,
    eps=1e-12,
    return_scores=False,
):
    """
    Build a boolean mask indicating which (pair, band, frame) zones are reliable
    for TDOA extraction, using two cheap cues:

      1) pair-band energy
      2) pair-band normalized coherence

    This is intended as a low-cost replacement for a pure energy mask.

    Parameters
    ----------
    signalsSTFT : (M, F, T) complex ndarray
        STFT of microphone signals.
    W : (B, F) real ndarray
        Frequency-band weights/masks.
    iu, ju : (P,) int ndarray
        Pair index arrays. Pair p is (iu[p], ju[p]).
    pair_mode : {"min", "geom_mean", "mean"}
        How to combine the two microphone energies of a pair:
          - "min"       : conservative, requires both mics to have energy
          - "geom_mean" : softer than min
          - "mean"      : least conservative
    energy_threshold_mode : {"percentile", "median_rel"}
        Thresholding mode for pair energy across time, separately for each (pair, band).
    energy_percentile : float
        Percentile used if energy_threshold_mode="percentile".
    energy_rel_factor : float
        Factor used if energy_threshold_mode="median_rel".
    coherence_threshold_mode : {"percentile", "median_rel"}
        Thresholding mode for pair coherence across time, separately for each (pair, band).
    coherence_percentile : float
        Percentile used if coherence_threshold_mode="percentile".
    coherence_rel_factor : float
        Factor used if coherence_threshold_mode="median_rel".
    combine_mode : {"and", "score"}
        How to combine energy and coherence:
          - "and"   : both masks must be true
          - "score" : threshold a normalized weighted score
    score_alpha : float
        Weight for normalized energy in score mode:
            score = score_alpha * E_norm + (1 - score_alpha) * C_norm
    eps : float
        Small constant for numerical stability.
    return_scores : bool
        If True, also return intermediate energies, coherences, thresholds, and masks.

    Returns
    -------
    valid_mask : (P, B, T) bool ndarray
        True where the zone is reliable enough.
    band_energy_mic : (M, B, T) ndarray, optional
    pair_energy : (P, B, T) ndarray, optional
    pair_coherence : (P, B, T) ndarray, optional
    energy_thresholds : (P, B) ndarray, optional
    coherence_thresholds : (P, B) ndarray, optional
    energy_mask : (P, B, T) bool ndarray, optional
    coherence_mask : (P, B, T) bool ndarray, optional
    score : (P, B, T) ndarray, optional
    """
    signalsSTFT = np.asarray(signalsSTFT)
    W = np.asarray(W, dtype=float)
    iu = np.asarray(iu, dtype=int)
    ju = np.asarray(ju, dtype=int)

    if signalsSTFT.ndim != 3:
        raise ValueError("signalsSTFT must have shape (M, F, T)")
    if W.ndim != 2:
        raise ValueError("W must have shape (B, F)")
    if signalsSTFT.shape[1] != W.shape[1]:
        raise ValueError("signalsSTFT and W must have the same frequency dimension")
    if iu.shape != ju.shape:
        raise ValueError("iu and ju must have the same shape")

    # Frequency-local microphone power: (M, F, T)
    mag2 = np.abs(signalsSTFT) ** 2

    # Band energy per microphone: (M, B, T)
    band_energy_mic = np.einsum("bf,mft->mbt", W, mag2, optimize=True)

    # Pair-band energy: (P, B, T)
    Ei = band_energy_mic[iu]
    Ej = band_energy_mic[ju]

    if pair_mode == "min":
        pair_energy = np.minimum(Ei, Ej)
    elif pair_mode == "geom_mean":
        pair_energy = np.sqrt(np.maximum(Ei * Ej, 0.0))
    elif pair_mode == "mean":
        pair_energy = 0.5 * (Ei + Ej)
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    # Pair-band normalized coherence, aggregated over each band:
    #   |sum_f W X_i X_j*|^2 / [(sum_f W |X_i|^2)(sum_f W |X_j|^2)]
    Xi = signalsSTFT[iu]                      # (P, F, T)
    Xj = signalsSTFT[ju]                      # (P, F, T)
    cross = Xi * np.conj(Xj)                  # (P, F, T)
    cross_band = np.einsum("bf,pft->pbt", W, cross, optimize=True)
    pair_coherence = (np.abs(cross_band) ** 2) / np.maximum(Ei * Ej, eps)
    pair_coherence = np.clip(pair_coherence, 0.0, 1.0)

    def _threshold_over_time(x, mode, percentile_value, rel_factor_value):
        if mode == "percentile":
            return np.percentile(x, percentile_value, axis=2)
        elif mode == "median_rel":
            return rel_factor_value * np.median(x, axis=2)
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")

    energy_thresholds = _threshold_over_time(
        pair_energy,
        energy_threshold_mode,
        energy_percentile,
        energy_rel_factor,
    )
    coherence_thresholds = _threshold_over_time(
        pair_coherence,
        coherence_threshold_mode,
        coherence_percentile,
        coherence_rel_factor,
    )

    energy_mask = pair_energy >= (energy_thresholds[:, :, None] + eps)
    coherence_mask = pair_coherence >= (coherence_thresholds[:, :, None] + eps)

    combine_mode = combine_mode.lower()
    if combine_mode == "and":
        valid_mask = energy_mask & coherence_mask
        score = None
    elif combine_mode == "score":
        energy_norm = pair_energy / np.maximum(energy_thresholds[:, :, None], eps)
        coherence_norm = pair_coherence / np.maximum(coherence_thresholds[:, :, None], eps)
        score = score_alpha * energy_norm + (1.0 - score_alpha) * coherence_norm
        valid_mask = score >= 1.0
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")

    if return_scores:
        return (
            valid_mask,
            band_energy_mic,
            pair_energy,
            pair_coherence,
            energy_thresholds,
            coherence_thresholds,
            energy_mask,
            coherence_mask,
            score,
        )

    return valid_mask

def matching_pursuit_tdoa(
    y,                  # length-L TDOA histogram (smoothed), nonnegative
    L,                  # number of bins (len(y))
    QN, QW,             # odd widths (in bins) for narrow & wide atoms
    t_min, t_max,       # TDOA range covered by the histogram
    verbose=False
):
    """
    Matching pursuit on a linear TDOA histogram over [t_min, t_max], using the
    same atom construction, candidate selection, and residual update logic as
    `matching_pursuit_tdoa_maxSources`, but with an automatic stopping rule.

    Parameters
    ----------
    y : array-like, shape (L,)
        Smoothed TDOA histogram.
    L : int
        Number of bins.
    QN, QW : int
        Atom widths (in bins) for narrow and wide atoms.
    u_min : float
        Kept only for API compatibility.
        It is not used in this function.
    t_min, t_max : float
        Min and max TDOA values represented by the histogram.
    verbose : bool
        Print debug info.

    Returns
    -------
    tdoas_hat : (P,) ndarray
        Estimated TDOA peaks (same units as t_min/t_max).
    betas : (P,) ndarray
        MP coefficients for each detected peak.
    residual : (L,) ndarray
        Final residual histogram.
    debug : dict or None
        Internal traces if verbose, else None.
    """
    y = np.asarray(y, dtype=float).copy()
    if y.ndim != 1 or len(y) != L:
        raise ValueError("y must be 1D of length L")

    # y, _ = remove_baseline(y, sigma_base=10*QW)

    cN = _blackman_atom(L, QN)
    cW = _blackman_atom(L, QW)
    E_CN = _energy(cN)

    span = float(t_max - t_min)

    tdoas_hat, betas = [], []
    residual = y.copy()

    traces = {"a_list": [], "i_star": [], "beta": [], "y_snapshots": [residual.copy()]}

    def rms(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

    j = 1
    a1_max = None
    while True:
        # Same correlation logic as matching_pursuit_tdoa_maxSources
        pad = int(QW)
        res_padded = np.pad(residual, pad, mode="reflect")
        a_full = np.correlate(res_padded, cN, mode="same")
        a = a_full[pad:pad + L]
        traces["a_list"].append(a.copy())

        if not np.any(np.isfinite(a)):
            break

        i_star = int(np.argmax(a))
        traces["i_star"].append(i_star)

        tdoa_hat = t_min + i_star * span / L
        beta_j = float(a[i_star] / (E_CN + 1e-12))
        traces["beta"].append(beta_j)

        if verbose:
            print(f"[j={j}] i*={i_star}, TDOA={tdoa_hat:.6f}, a[i*]={a[i_star]:.4f}, beta={beta_j:.4f}")

        # Automatic stopping rule: compare local peak RMS against residual background.
        half_w = max(1, QW // 2)
        p0 = max(0, i_star - half_w)
        p1 = min(L, i_star + half_w + 1)

        peak_vals = residual[p0:p1]

        noise_mask = np.ones(L, dtype=bool)
        noise_mask[p0:p1] = False
        noise_vals = residual[noise_mask]

        peak_rms = rms(peak_vals)
        noise_rms = rms(noise_vals) if noise_vals.size > 0 else 1e-12
        snr_rms = peak_rms / (noise_rms + 1e-12)

        if verbose:
            print(f"  peak_rms={peak_rms:.4g}, noise_rms={noise_rms:.4g}, ratio={snr_rms:.2f}")

        if j == 1:
            a1_max = a[i_star]

        if snr_rms < 3.0 and a[i_star] / (a1_max + 1e-12) < 0.2:
            break
        # if a[i_star] / (a1_max + 1e-12) < 0.2:
        #     break
        # Same residual update logic as matching_pursuit_tdoa_maxSources
        cW_shift = _shift_atom_linear(cW, i_star - L // 2)
        residual = np.maximum(residual - cW_shift, 0.0)

        traces["y_snapshots"].append(residual.copy())

        tdoas_hat.append(tdoa_hat)
        betas.append(beta_j)
        j += 1

    debug = traces if verbose else None
    return np.asarray(tdoas_hat), np.asarray(betas), residual, debug