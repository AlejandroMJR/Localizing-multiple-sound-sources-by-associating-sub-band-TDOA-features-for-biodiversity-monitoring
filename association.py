import numpy as np

"""
Convention
----------------------
All TDOA / delay variables (tau) in this module are expressed in samples.

This module covers:
  - Feature construction from sub-band support (band-count feature vectors)
  - Association of TDOA peaks 
"""


# ============================================================
# Feature construction
# ============================================================

def attach_band_info_to_peaks_fast(tau_obs, bands, peak_taus, tol_tau):
    """
    Attach band-support information to each detected peak tau.

    For each peak tau_hat, this collects nearby observations (within tol_tau)
    and builds a band-support histogram (feature vector):
        band_counts[b] = number of observations assigned to band b.

    Parameters
    ----------
    tau_obs : (N,) array-like
        Local TDOA observations in samples.
    bands : (N,) array-like of int
        Band index for each observation in tau_obs.
    peak_taus : (P,) array-like
        Peak locations in samples (from matching pursuit).
    tol_tau : float
        Assignment tolerance in samples (W in the paper):
            an observation tau is assigned to tau_hat if |tau - tau_hat| <= tol_tau.

    Returns
    -------
    peaks : list of dict
        One dict per peak, with:
          - "tau" : float (samples)
          - "bands" : list[int] unique band indices supporting the peak
          - "band_counts" : dict {band_id: count}
    """
    tau_obs = np.asarray(tau_obs, dtype=float)
    bands = np.asarray(bands, dtype=int)
    peak_taus = np.asarray(peak_taus, dtype=float)

    if tau_obs.size != bands.size:
        raise ValueError("tau_obs and bands must have the same length.")

    peaks = []
    if peak_taus.size == 0:
        return peaks

    # If there are no observations, return empty support for each peak
    if tau_obs.size == 0:
        for tau_hat in peak_taus:
            peaks.append({"tau": float(tau_hat), "bands": [], "band_counts": {}})
        return peaks

    for tau_hat in peak_taus:
        close = np.abs(tau_obs - tau_hat) <= tol_tau
        b = bands[close]

        if b.size == 0:
            peaks.append({"tau": float(tau_hat), "bands": [], "band_counts": {}})
            continue

        uniq, cnt = np.unique(b, return_counts=True)
        peaks.append(
            {
                "tau": float(tau_hat),  # samples
                "bands": uniq.tolist(),
                "band_counts": dict(zip(uniq.tolist(), cnt.tolist())),
            }
        )

    return peaks

def attach_timefreq_info_to_peaks(
    tau_obs,
    bands,
    frames,
    peak_taus,
    tol_tau,
    n_bands=None,
    n_frames=None,
):
    """
    Old-format-compatible peak attachment.

    Keeps the old keys:
        - "tau"
        - "bands"
        - "band_counts"

    Adds new keys:
        - "frames"
        - "tf_cells"
        - "tf_support_map" (optional)
        - "obs_idx"
        - "n_support"
    """
    tau_obs = np.asarray(tau_obs)
    bands = np.asarray(bands)
    frames = np.asarray(frames)
    peak_taus = np.asarray(peak_taus)

    peaks = []

    for k, tau_hat in enumerate(peak_taus):
        mask = np.abs(tau_obs - tau_hat) <= tol_tau
        idx = np.flatnonzero(mask)

        peak_bands = bands[idx].copy()
        peak_frames = frames[idx].copy()

        band_counts = {}
        if peak_bands.size > 0:
            uniq_b, cnt_b = np.unique(peak_bands, return_counts=True)
            band_counts = {int(b): int(c) for b, c in zip(uniq_b, cnt_b)}

        peak_dict = {
            "peak_idx": k,
            "tau": float(tau_hat),          # keep OLD expected key
            "bands": peak_bands,            # keep OLD expected key
            "band_counts": band_counts,     # keep OLD expected key

            # new TF info
            "frames": peak_frames,
            "obs_idx": idx,
            "n_support": int(idx.size),
            "tf_cells": np.stack([peak_bands, peak_frames], axis=1) if idx.size > 0 else np.zeros((0, 2), dtype=int),
        }

        if n_bands is not None and n_frames is not None:
            tf_support_map = np.zeros((n_bands, n_frames), dtype=np.float32)
            for b, t in peak_dict["tf_cells"]:
                if 0 <= b < n_bands and 0 <= t < n_frames:
                    tf_support_map[b, t] += 1.0
            peak_dict["tf_support_map"] = tf_support_map

        peaks.append(peak_dict)

    return peaks

def peak_band_vector(peak, n_bands):
    """
    Convert peak["band_counts"] into a dense band-support vector.

    Parameters
    ----------
    peak : dict
        Must contain 'band_counts' (dict {band_id: count}).
    n_bands : int
        Total number of bands (feature dimension).

    Returns
    -------
    vec : (n_bands,) ndarray
        Dense band-support vector (float).
    """
    vec = np.zeros(n_bands, dtype=float)
    for b, cnt in peak.get("band_counts", {}).items():
        b = int(b)
        if 0 <= b < n_bands:
            vec[b] = float(cnt)
    return vec


# ============================================================
# Association
# ============================================================

def associate_tdoa_peaks_by_band_cosine_constrained(
    tdoas_hat_pairs,
    n_bands,
    cosine_thresh=0.5,
    max_sources=None,
    min_tdoas_per_source=2,
):
    """
    Associate (mic_pair, peak) candidates across microphone pairs using cosine similarity
    between band-support feature vectors, under a physical constraint:
        a source can induce at most one TDOA peak per microphone pair.

    Overview
    --------
    1) Create one node per peak (per microphone pair).
    2) Compute cosine similarity between feature vectors for all cross-pair node pairs.
    3) Keep only similarities >= cosine_thresh, and only across different mic pairs.
    4) Greedily merge clusters (from strongest to weakest link) while enforcing:
         - no cluster may contain two peaks from the same mic pair.
    5) Return clusters as source hypotheses.

    Parameters
    ----------
    tdoas_hat_pairs : dict
        (i,j) -> dict with key "peaks" (list of dicts).
        Each peak dict is expected to contain:
          - "tau" (samples)
          - "band_counts" (dict) or be compatible with peak_band_vector
    n_bands : int
        Feature dimension (total number of bands).
    cosine_thresh : float
        Minimum cosine similarity to consider two peaks compatible (beta in the paper).
    max_sources : int or None
        If set, keep only the largest `max_sources` clusters (S in the paper).
    min_tdoas_per_source : int
        Discard clusters with fewer than this many members (Discards groups form by only one TDOA).

    Returns
    -------
    sources : list of dict
        Each dict corresponds to one source hypothesis and contains:
          - "members": list of dict with keys:
              "pair", "peak_idx", "tau", "bands", "band_counts", "vec"
    """
    # -------------------------
    # 1) Build nodes: one per peak
    # -------------------------
    nodes = []
    for (i, j), info in tdoas_hat_pairs.items():
        for p_idx, peak in enumerate(info.get("peaks", [])):
            vec = peak_band_vector(peak, n_bands)
            nodes.append(((i, j), p_idx, peak, vec))

    N = len(nodes)
    if N == 0:
        return []

    vecs = np.stack([n[3] for n in nodes], axis=0).astype(float)
    norms = np.linalg.norm(vecs, axis=1) + 1e-12
    pair_of = [nodes[u][0] for u in range(N)]  # mic pair label for node u

    def cosine(u, v):
        return float(np.dot(vecs[u], vecs[v]) / (norms[u] * norms[v]))

    # -------------------------
    # 2) Prospective pairs (above threshold, cross-pair only)
    # -------------------------
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if pair_of[u] == pair_of[v]:
                continue  # cannot associate two peaks from the same mic pair
            s = cosine(u, v)
            if s >= cosine_thresh:
                edges.append((s, u, v))

    # If no edges survive, each node becomes its own cluster
    if len(edges) == 0:
        comps = [[u] for u in range(N)]
    else:
        # -------------------------
        # 3) Constrained greedy merging
        # -------------------------
        edges.sort(reverse=True, key=lambda x: x[0])

        parent = np.arange(N, dtype=int)
        size = np.ones(N, dtype=int)
        used_pairs = [set([pair_of[u]]) for u in range(N)]

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False

            # Physical constraint: one peak per mic pair inside each cluster
            if used_pairs[ra] & used_pairs[rb]:
                return False

            # Union by size
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            used_pairs[ra] |= used_pairs[rb]
            return True

        for _, u, v in edges:
            union(u, v)

        buckets = {}
        for u in range(N):
            r = find(u)
            buckets.setdefault(r, []).append(u)
        comps = list(buckets.values())

    # -------------------------
    # 4) Build output sources
    # -------------------------
    def make_source(indices):
        members = []
        for idx in indices:
            pair, p_idx, peak, vec = nodes[idx]
            members.append(
                {
                    "pair": pair,
                    "peak_idx": p_idx,
                    "tau": float(peak["tau"]),  # samples
                    "bands": peak.get("bands", []),
                    "band_counts": peak.get("band_counts", {}),
                    "vec": vec,
                }
            )
        return {"members": members}

    sources = [make_source(c) for c in comps]

    # Keep only largest sources if requested
    if max_sources is not None and len(sources) > max_sources:
        sources = sorted(sources, key=lambda s: len(s["members"]), reverse=True)[:max_sources]

    # Enforce minimum number of supporting pairs
    sources = [s for s in sources if len(s["members"]) >= min_tdoas_per_source]

    return sources
