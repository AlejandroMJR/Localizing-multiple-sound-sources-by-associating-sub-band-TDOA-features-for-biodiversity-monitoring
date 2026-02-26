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
