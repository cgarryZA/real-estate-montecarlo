import numpy as np

def summarize(final_portfolio_values: np.ndarray):
    """
    Summarize the final portfolio values from multiple simulation paths.
    """
    return {
        "mean": float(np.mean(final_portfolio_values)),
        "median": float(np.median(final_portfolio_values)),
        "p05": float(np.percentile(final_portfolio_values, 5)),
        "p95": float(np.percentile(final_portfolio_values, 95)),
        "std": float(np.std(final_portfolio_values)),
    }


def path_stats_from_portfolios(portfolio_paths: np.ndarray,
                               steps_per_year: int,
                               risk_free: float = 0.0):
    """
    portfolio_paths: shape (n_paths, n_steps) with level values (£)
    We convert levels -> returns -> annualise -> Sharpe.

    Returns dict of:
      - ann_return: array (n_paths,)
      - ann_vol: array (n_paths,)
      - sharpe: array (n_paths,)
    """
    n_paths, n_steps = portfolio_paths.shape

    # convert to simple returns per step
    # r_t = (V_t / V_{t-1}) - 1
    rets = portfolio_paths[:, 1:] / portfolio_paths[:, :-1] - 1.0  # (n_paths, n_steps-1)

    # per-path average step return and step vol
    mean_step = rets.mean(axis=1)
    std_step = rets.std(axis=1, ddof=1)

    # annualise
    ann_return = (1.0 + mean_step) ** steps_per_year - 1.0
    ann_vol = std_step * np.sqrt(steps_per_year)

    # Sharpe
    excess = ann_return - risk_free
    sharpe = np.zeros_like(excess)
    # avoid divide by zero
    nz = ann_vol > 0
    sharpe[nz] = excess[nz] / ann_vol[nz]

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }
