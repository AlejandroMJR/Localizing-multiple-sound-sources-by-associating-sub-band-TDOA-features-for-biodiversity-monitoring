import numpy as np
from scipy.optimize import least_squares


def estimate_source_position_from_tdoas(
    mic_positions,
    tdoa_measurements,
    fs,
    c=343.0,
    x0=None,
    bounds=None,
    loss="huber",
    f_scale=0.002,
    max_nfev=2000,
):
    """
    Estimate a single source position from associated TDOA constraints.

    Model:
        (||x - m_i|| - ||x - m_j||) / c  =  tau_ij   (seconds)
    """
    mics = np.asarray(mic_positions, dtype=float)
    M, D = mics.shape

    # Parse measurements into arrays
    pairs_i = []
    pairs_j = []
    tau_sec = []

    for meas in tdoa_measurements:
        i = int(meas["i"]) if isinstance(meas, dict) else int(meas[0])
        j = int(meas["j"]) if isinstance(meas, dict) else int(meas[1])

        if isinstance(meas, dict):
            if "tau_seconds" in meas:
                tau = float(meas["tau_seconds"])
            else:
                tau = float(meas["tau_samples"]) / float(fs)
        else:
            tau = float(meas[2]) / float(fs)

        pairs_i.append(i)
        pairs_j.append(j)
        tau_sec.append(tau)

    pairs_i = np.asarray(pairs_i, dtype=int)
    pairs_j = np.asarray(pairs_j, dtype=int)
    tau_sec = np.asarray(tau_sec, dtype=float)

    # if tau_sec.size < D + 1: ### Already handled by association step
    #     raise ValueError(
    #         f"Need more TDOA constraints. Got {tau_sec.size}, recommended >= {D+1}."
    #     )

    # Initial guess
    if x0 is None:
        x0 = np.mean(mics, axis=0)

    # Bounds
    if bounds is None:
        lb = -np.inf * np.ones(D)
        ub = np.inf * np.ones(D)
        bounds = (lb, ub)

    # Residual function (seconds)
    def residuals(x):
        x = np.asarray(x, dtype=float)
        di = np.linalg.norm(x - mics[pairs_i], axis=1)
        dj = np.linalg.norm(x - mics[pairs_j], axis=1)
        tau_pred = (di - dj) / float(c)
        return tau_pred - tau_sec

    res = least_squares(
        residuals,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        jac="2-point",
    )

    info = {
        "success": bool(res.success),
        "message": res.message,
        "cost": float(res.cost),
        "residuals_sec": res.fun,
        "jacobian": res.jac,
        "nfev": res.nfev,
    }
    return res.x, info


def estimate_all_sources_positions(
    sources,
    mic_positions,
    fs,
    c=343.0,
    env_bounds=None,
    loss="huber",
    f_scale=0.002,
):
    """
    Estimate positions for all associated sources

    Parameters
    ----------
    sources : list of dict
         Clustered/associated sources.
    mic_positions : (M,D)
    fs : float
    env_bounds : tuple(lb, ub) or None
        Bounds for solver. For 2D square env:
          lb=[0,0], ub=[env_size, env_size]
    loss, f_scale : see estimate_source_position_from_tdoas

    Returns
    -------
    estimates : list of dict
        Each entry:
          {
            "x_hat": (D,),
            "info": solver_info,
            "n_constraints": K
          }
    """
    estimates = []
    for src in sources:
        tdoa_meas = []
        for mem in src.get("members", []):
            i, j = mem["pair"]
            tdoa_meas.append({"i": int(i), "j": int(j), "tau_samples": float(mem["tau"])})

        if len(tdoa_meas) == 0:
            estimates.append({"x_hat": None, "info": {"success": False, "message": "No members"}, "n_constraints": 0})
            continue

        x_hat, info = estimate_source_position_from_tdoas(
            mic_positions=mic_positions,
            tdoa_measurements=tdoa_meas,
            fs=fs,
            c=c,
            x0=None,
            bounds=env_bounds,
            loss=loss,
            f_scale=f_scale,
        )

        estimates.append(
            {"x_hat": x_hat, "info": info, "n_constraints": len(tdoa_meas)}
        )

    return estimates
