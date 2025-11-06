# analyze.py

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
