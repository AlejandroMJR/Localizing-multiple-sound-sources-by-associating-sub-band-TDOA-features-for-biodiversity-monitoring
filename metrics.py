import numpy as np
from scipy.optimize import linear_sum_assignment

def match_estimates_to_truth(srcPos, estPos, max_match_dist=None):
    """
    Match estimated source positions to ground-truth source positions using
    minimum total Euclidean distance. This is necessary since we don't know
    which estimate corresponds to which true source.

    Parameters
    ----------
    srcPos : (N_true, D) array
        True source positions.
    estPos : (N_est, D) array
        Estimated source positions.
    max_match_dist : float or None
        If not None, any matched pair with distance > max_match_dist is treated as invalid.

    Returns
    -------
    matching : list of tuples (true_idx, est_idx, dist)
        Optimal assignment pairs (may be filtered by max_match_dist if provided).
    unmatched_true : list of int
        True sources not matched (only happens if N_est != N_true or filtering removes matches).
    unmatched_est : list of int
        Estimates not matched.
    cost_matrix : (N_true, N_est) array
        Distance matrix used for matching.
    """
    srcPos = np.asarray(srcPos, dtype=float)
    estPos = np.asarray(estPos, dtype=float)

    N_true = srcPos.shape[0]
    N_est  = estPos.shape[0]

    # cost matrix: distances
    diff = srcPos[:, None, :] - estPos[None, :, :]
    C = np.linalg.norm(diff, axis=-1)  # (N_true, N_est)

    row_ind, col_ind = linear_sum_assignment(C)

    matching = []
    matched_true = set()
    matched_est = set()

    for ti, ei in zip(row_ind, col_ind):
        d = float(C[ti, ei])
        if (max_match_dist is None) or (d <= max_match_dist):
            matching.append((int(ti), int(ei), d))
            matched_true.add(int(ti))
            matched_est.add(int(ei))

    unmatched_true = [i for i in range(N_true) if i not in matched_true]
    unmatched_est  = [j for j in range(N_est)  if j not in matched_est]

    return matching, unmatched_true, unmatched_est, C


def compute_position_errors(srcPos, estPos, max_match_dist=None):
    """
    Compute per-source errors after optimal matching.

    Returns a dict with metrics.
    """
    matching, unmatched_true, unmatched_est, C = match_estimates_to_truth(
        srcPos, estPos, max_match_dist=max_match_dist
    )

    if len(matching) == 0:
        return {
            "matching": [],
            "rmse": np.nan,
            "mae": np.nan,
            "errors": np.array([]),
            "unmatched_true": unmatched_true,
            "unmatched_est": unmatched_est,
        }

    errors = np.array([d for (_, _, d) in matching], dtype=float)
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(errors))

    return {
        "matching": matching,          # list of (true_idx, est_idx, dist)
        "errors": errors,              # matched distances
        "rmse": rmse,
        "mae": mae,
        "unmatched_true": unmatched_true,
        "unmatched_est": unmatched_est,
        "cost_matrix": C,
    }

